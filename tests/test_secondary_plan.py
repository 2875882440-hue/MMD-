import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as N
import pytest

from motionbridge.secondary_motion import _strict_pmx
from motionbridge.secondary_plan import compile_profile, prepare
from motionbridge.secondary_topology import protected_closure, skin_weights, material_partition
from motionbridge.secondary_units import position, rotation, rigid_parameters, joint_parameters, mask_allows

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def profile():
    return json.loads((ROOT/'profiles/secondary_motion/rin.json').read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def real_plan():
    profile = json.loads((ROOT/'profiles/secondary_motion/rin.json').read_text(encoding='utf-8'))
    return compile_profile(profile, ROOT)


def test_coordinate_adapter_matches_mmd_import_axes_and_rotation_order():
    assert position([1, 2, 3], .1) == pytest.approx([.1, .3, .2])
    assert rotation([.2, .4, .8]) == [-.2, -.8, -.4]
    rigid = N(type=1, size=[1, 2, 3], location=[0, 0, 0], rotation=[0, 0, 0], mass=2,
              velocity_attenuation=.5, rotation_attenuation=.4, bounce=.2, friction=.3,
              collision_group_number=1, collision_group_mask=4)
    p = rigid_parameters(rigid, .1)
    assert p['shape']['half_extents_m'] == pytest.approx([.1, .3, .2])
    assert p['rotation_order'] == 'YXZ' and p['mass_import_value'] == 2
    rigid.type = 2
    assert rigid_parameters(rigid, .1)['shape']['total_height_m'] == pytest.approx(.4)


def test_joint_negative_axis_conversion_swaps_angular_limits():
    joint = N(mode=0, location=[1, 2, 3], rotation=[.1, .2, .3], minimum_location=[-1, -2, -3],
              maximum_location=[1, 2, 3], minimum_rotation=[-.1, -.2, -.3], maximum_rotation=[.4, .5, .6],
              spring_constant=[10, 20, 30], spring_rotation_constant=[40, 50, 60])
    result = joint_parameters(joint, .08)
    assert result['angular_min_rad'] == [-.4, -.6, -.5]
    assert result['angular_max_rad'] == [.1, .3, .2]
    assert result['linear_spring_import_value'] == [10, 30, 20]


def test_mask_bits_exclude_and_both_partners_must_allow():
    assert mask_allows(0, 0, 2, 0)
    assert not mask_allows(0, 4, 2, 0)
    assert not mask_allows(2, 0, 0, 4)
    with pytest.raises(ValueError): mask_allows(16, 0, 0, 0)


@pytest.mark.parametrize('kind,values,expected', [(0, [], {0:1}), (1, [.3], {0:.3, 1:.7}),
    (2, [.1,.2,.3,.4], {0:.1,1:.2,2:.3,3:.4}), (3, N(weight=.25), {0:.25,1:.75})])
def test_supported_skin_weights(kind, values, expected):
    bone_ids = [0] if kind == 0 else [0,1,2,3] if kind == 2 else [0,1]
    assert skin_weights(N(type=kind, bones=bone_ids, weights=values), 4) == pytest.approx(expected)


def test_non_normalized_or_missing_skin_weights_are_rejected():
    with pytest.raises(ValueError): skin_weights(N(type=2,bones=[0,1,2,3],weights=[.1]*4), 4)
    with pytest.raises(ValueError): skin_weights(N(type=0,bones=[99],weights=[]), 4)


def test_append_and_ik_dependencies_protect_more_than_parent_chains():
    bones = [N(name=name, parent=None) for name in ('body','donor','IK','target','link','independent')]
    bones[0].hasAdditionalRotate = True; bones[0].additionalTransform = (1, .5)
    bones[2].isIK = True; bones[2].target = 3; bones[2].ik_links = [N(target=4)]
    bones[1].parent = 4
    assert protected_closure(bones, ['body']) == {0,1,2,3,4}


def test_invalid_parent_cycles_are_rejected():
    with pytest.raises(ValueError, match='环'):
        protected_closure([N(name='a',parent=1),N(name='b',parent=0)], ['a'])


def test_partition_rejects_unknown_and_duplicate_materials():
    model = N(materials=[N(name='cloth',vertex_count=3)], faces=[[0,1,2]], vertices=[0]*3)
    with pytest.raises(ValueError): material_partition(model, [])
    with pytest.raises(ValueError): material_partition(model, [
        {'id':'a','role':'cloth','materials':['cloth']}, {'id':'b','role':'hair','materials':['cloth']}])


def test_real_asset_faces_weights_and_native_physics_are_fully_accounted_for(real_plan):
    assert len(real_plan['rigid_bodies']) == 344 and len(real_plan['joints']) == 500
    faces = [face for r in real_plan['regions'] for face in r['face_ids']]
    assert len(faces) == len(set(faces)) == 47384
    assert set(real_plan['selected_secondary_bone_ids']).isdisjoint(real_plan['protected_bone_ids'])
    protected_vertices = {v for r in real_plan['regions'] if r['role'] in ('body','boots') for v in r['vertex_ids']}
    for vertex in protected_vertices:
        assert set(map(int, real_plan['geometry']['skin_weights'][vertex])).isdisjoint(real_plan['selected_secondary_bone_ids'])
    assert all(not c['requires_explicit_root_attachment'] for c in real_plan['components'])
    assert real_plan['enabled'] is real_plan['simulation_performed'] is real_plan['solver_ready'] is False


def test_pins_reflect_remaining_kinematic_skin_weight(real_plan):
    hair = next(r for r in real_plan['regions'] if r['id']=='hair')
    assert hair['fixed_vertex_count'] > 0 and hair['free_vertex_count'] > 0 and hair['transition_vertex_count'] > 0
    selected = set(real_plan['selected_secondary_bone_ids'])
    for vertex, pin in zip(hair['vertex_ids'],hair['pin_weights']):
        weights = real_plan['geometry']['skin_weights'][vertex]
        assert pin == pytest.approx(1 - sum(w for b,w in weights.items() if int(b) in selected), abs=1e-6)


def test_changed_asset_fingerprint_is_rejected(profile):
    profile['model']['sha256'] = '0'*64
    with pytest.raises(ValueError,match='指纹'): compile_profile(profile,ROOT)


def test_late_joint_truncation_cannot_be_filled_with_zero_defaults(profile,tmp_path):
    path = tmp_path/'last-joint-truncated.pmx'
    path.write_bytes((ROOT/profile['model']['path']).read_bytes()[:-40])
    with pytest.raises(ValueError,match='不完整|损坏'): _strict_pmx(path)


def test_output_fence_and_existing_outputs_are_protected(tmp_path):
    with pytest.raises(ValueError,match='独立子目录'):
        prepare(ROOT/'profiles/secondary_motion/rin.json', ROOT, ROOT/'assets/wrong')
    # Root is isolated to exercise refusal without writing real output.
    folder = tmp_path/'output/secondary_motion/already-exists'; folder.mkdir(parents=True)
    with pytest.raises(FileExistsError): prepare('unused.json', tmp_path, folder)


def test_compiling_never_modifies_native_model(profile):
    path = ROOT/profile['model']['path']
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    compile_profile(profile,ROOT)
    assert before == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('change', ['empty_token','body_dynamics','unknown_alias'])
def test_malformed_roles_and_bindings_are_rejected(profile, change):
    if change == 'empty_token': profile['candidate_name_tokens']['hair'] = ['']
    if change == 'body_dynamics': profile['regions'][-1]['deformation_policy'] = 'bone_dynamics_candidate'
    if change == 'unknown_alias': profile['bindings']['bone_aliases'] = {'not-a-bone':'anything'}
    with pytest.raises(ValueError): compile_profile(profile, ROOT)


def test_floor_mask_gaps_and_loose_vertices_remain_visible_diagnostics(real_plan):
    warnings = {w['code']:w for w in real_plan['warnings']}
    assert warnings['native_masks_exclude_floor']['excluded_candidate_rigids'] == 262
    assert real_plan['collision_policy']['actual_contacts_tested'] is False
    assert len(real_plan['geometry']['unreferenced_vertex_ids']) == 8
