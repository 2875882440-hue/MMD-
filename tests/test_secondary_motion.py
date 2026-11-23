import copy,json,hashlib
from pathlib import Path
import pytest
from motionbridge.secondary_motion import inspect_profile,validate_profile

ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture
def profile():
    return json.loads((ROOT/'profiles/secondary_motion/rin.json').read_text(encoding='utf-8'))

def test_real_native_physics_is_inspected_without_changing_model(profile):
    source=ROOT/profile['model']['path'];before=hashlib.sha256(source.read_bytes()).hexdigest()
    report=inspect_profile(profile,ROOT)
    assert report['native_physics']['rigid_body_count']==344
    assert report['native_physics']['joint_count']==500
    assert report['native_physics']['mode_counts']=={'kinematic_follow':32,'dynamic_bone':116,'dynamic':196}
    assert report['simulation_performed'] is False
    assert report['primary_motion_protection']['missing']==[]
    assert report['model_sha256']==before==hashlib.sha256(source.read_bytes()).hexdigest()

@pytest.mark.parametrize('enabled',[True,1,'false'])
def test_unimplemented_solver_cannot_be_enabled(profile,enabled):
    profile['enabled']=enabled
    with pytest.raises(NotImplementedError,match='尚未提供动力学求解'):
        validate_profile(profile,ROOT)

@pytest.mark.parametrize('scale',[0,-.08,float('nan'),float('inf'),True])
def test_invalid_physical_scales_are_rejected(profile,scale):
    profile['model']['scale']=scale
    with pytest.raises(ValueError,match='缩放'):validate_profile(profile,ROOT)

def test_model_cannot_escape_current_development_copy(profile):
    profile['model']['path']='../MotionBridge/other.pmx'
    with pytest.raises(ValueError,match='当前开发副本'):validate_profile(profile,ROOT)

def test_primary_motion_protection_is_mandatory(profile):
    profile['protect_primary_motion']=False
    with pytest.raises(ValueError,match='保护主身体'):validate_profile(profile,ROOT)

def test_dynamic_hair_cannot_silently_drive_a_primary_bone(profile):
    profile['primary_bones'].append('後髪1')
    report=inspect_profile(profile,ROOT)
    assert any(v['code']=='dynamic_physics_on_primary_bone' and v['bone']=='後髪1' for v in report['blocking_issues'])

def test_truncated_native_physics_is_not_accepted_as_partial_data(profile,tmp_path):
    broken=tmp_path/'truncated.pmx';broken.write_bytes((ROOT/profile['model']['path']).read_bytes()[:128])
    profile['model']['path']=str(broken)
    with pytest.raises(ValueError,match='不完整|损坏'):inspect_profile(profile,ROOT)

def test_dynamic_ancestor_is_reported_even_when_primary_itself_is_not_dynamic(profile,monkeypatch):
    from types import SimpleNamespace as N
    from motionbridge import secondary_motion
    profile['primary_bones']=['BodyChild']
    rigid=N(name='DynamicParent',bone=0,mode=1,size=[.1,.1,.1],location=[0.,0.,0.],rotation=[0.,0.,0.],
        mass=1.,velocity_attenuation=.5,rotation_attenuation=.5,type=0,collision_group_number=0,collision_group_mask=0)
    parsed=N(bones=[N(name='HairParent',parent=None),N(name='BodyChild',parent=0)],rigids=[rigid],joints=[],morphs=[])
    monkeypatch.setattr(secondary_motion,'_strict_pmx',lambda path:parsed)
    result=inspect_profile(profile,ROOT)
    assert 'HairParent' in result['primary_motion_protection']['protected_ancestor_bones']
    assert any(v['code']=='dynamic_physics_on_primary_ancestor' for v in result['blocking_issues'])
