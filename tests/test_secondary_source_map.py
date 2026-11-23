import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from motionbridge.secondary_segmentation import _sha, build_inventory
from motionbridge.secondary_source_map import SourceSurfaceMap, load_model_document


def fixture():
    from test_secondary_segmentation import sample, reviewed
    p=sample();p['protected_bone_ids']=[0];p['rigid_bodies']=[];p['joints']=[]
    p['bones']=[{'id':0,'name':'body'}]
    inv=build_inventory(p);p['surface_segmentation']={'inventory':inv}
    r=reviewed(inv);r['parts'][1]['category']='display_overlay';r['overlaps'][0]['kind']='display_duplicate'
    m={'schema_version':1,'model_sha256':p['model_sha256'],'geometry_sha256':inv['geometry_sha256'],'review_sha256':_sha(r),
       'aliases':[{'group_id':inv['coincident_face_groups'][0]['id'],'display_face':1,'canonical_face':0,
                   'vertex_pairs':[[3,0],[5,2],[4,1]]}],
       'parts':[{'part_id':part['id'],'source_bone_ids':[0],'native_rigid_ids':[],'native_joint_ids':[],
                 'attachment_reference':'immutable_source_skin_and_authored_graph'} for part in r['parts']]}
    return p,r,m


def test_alias_deltas_propagate_without_mutating_source():
    p,r,m=fixture();saved=copy.deepcopy((p,r,m));mapping=SourceSurfaceMap(p,r,m)
    source=np.array(p['geometry']['positions_m'],float);original=source.copy();delta=np.zeros_like(source);delta[0]=[0,0,.01]
    result=mapping.expand_delta(source,delta)
    assert np.array_equal(result[0],result[3]) and result[0,2]==.01
    assert np.array_equal(source,original) and (p,r,m)==saved
    assert mapping.summary()['physical_faces']==1 and mapping.summary()['source_faces']==2
    assert not mapping.summary()['simulation_ready']


@pytest.mark.parametrize('change',['fingerprint','incomplete_alias','bad_pair','repeat_pair','wrong_bones','wrong_joints','final_cloth','missing_part'])
def test_rejects_stale_or_invented_model_binding(change):
    p,r,m=fixture()
    if change=='fingerprint':m['geometry_sha256']='wrong'
    elif change=='incomplete_alias':m['aliases']=[]
    elif change=='bad_pair':m['aliases'][0]['vertex_pairs'][0]=[3,2]
    elif change=='repeat_pair':m['aliases'][0]['vertex_pairs'][1]=[3,0]
    elif change=='wrong_bones':m['parts'][0]['source_bone_ids']=[]
    elif change=='wrong_joints':m['parts'][0]['native_joint_ids']=[0]
    elif change=='final_cloth':m['parts'][0]['attachment_reference']='final_cloth'
    elif change=='missing_part':m['parts'].pop()
    with pytest.raises(ValueError):SourceSurfaceMap(p,r,m)


def test_each_current_pose_is_checked_instead_of_trusting_a_static_certificate():
    p,r,m=fixture();mapping=SourceSurfaceMap(p,r,m);x=mapping.rest.copy();x[3,0]+=.003
    with pytest.raises(ValueError,match='当前姿态'):mapping.expand_delta(x,np.zeros_like(x))


def test_changed_rest_geometry_cannot_reuse_inventory_fingerprint():
    p,r,m=fixture();p['geometry']['positions_m'][1][0]+=0.2
    with pytest.raises(ValueError,match='源几何'):SourceSurfaceMap(p,r,m)


def test_shared_protection_applies_to_display_alias_dof():
    p,r,m=fixture();p['regions'].append({'role':'body','face_ids':[],'vertex_ids':[3]})
    mapping=SourceSurfaceMap(p,r,m);delta=np.zeros_like(mapping.rest);delta[0,2]=.001
    assert set(mapping.fixed_vertices)=={0,3}
    with pytest.raises(ValueError,match='受保护'):mapping.expand_delta(mapping.rest,delta)


@pytest.mark.parametrize('change',['noncanonical','nan','size'])
def test_invalid_delta_cannot_be_silently_discarded(change):
    p,r,m=fixture();mapping=SourceSurfaceMap(p,r,m);delta=np.zeros_like(mapping.rest)
    if change=='noncanonical':delta[3,0]=.01
    elif change=='nan':delta[0,0]=float('nan')
    elif change=='size':delta=delta[:-1]
    with pytest.raises(ValueError):mapping.expand_delta(mapping.rest,delta)


def test_model_document_is_fenced_and_hash_bound(tmp_path):
    path=tmp_path/'review.json';path.write_text('{"example":1}',encoding='utf8')
    ref={'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    assert load_model_document(tmp_path,ref)=={'example':1}
    path.write_text('{"example":2}',encoding='utf8')
    with pytest.raises(ValueError,match='指纹'):load_model_document(tmp_path,ref)
    ref['path']='../outside.json'
    with pytest.raises(ValueError,match='开发副本'):load_model_document(tmp_path,ref)


def test_real_rin_has_complete_semantics_and_retains_weight_variant_faces():
    from motionbridge.secondary_plan import compile_profile
    root=Path(__file__).resolve().parents[1];profile=json.loads((root/'profiles/secondary_motion/rin.json').read_text(encoding='utf8'))
    original=copy.deepcopy(profile);p=compile_profile(profile,root)
    m=load_model_document(root,profile['source_surface_map']);r=p['surface_segmentation']['review']
    mapping=SourceSurfaceMap(p,r,m)
    assert profile==original
    assert len(mapping.triangles)==47384 and len(mapping.physical_face_ids)==37972
    assert len(mapping.alias_rows)==9412 and len(mapping.parts)==187
    differing=[g for g in p['surface_segmentation']['inventory']['coincident_face_groups'] if not g['matching_skin_weights']]
    assert len(differing)==6
    assert all(set(g['face_ids'])<=set(mapping.physical_face_ids) for g in differing)
    assert not p['source_surface_map']['dynamic_attachment_constraints_complete']
    cuff=next(part for part in r['parts'] if part['id']=='left_cuff_button_a')
    assert len(cuff['face_ids'])==38
    assert {p['geometry']['face_material_ids'][f] for f in cuff['face_ids']}=={11,16}
    ribbons=[part for part in r['parts'] if part['id'] in ('right_long_ribbon','m11-f12689')]
    assert len(ribbons)==2 and all(len(part['face_ids'])==97 and part['category']=='flexible_trim' for part in ribbons)
