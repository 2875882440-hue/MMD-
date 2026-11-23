import copy
import hashlib
import json
from pathlib import Path

import pytest
from motionbridge.secondary_segmentation import build_inventory,review_gate,require_review
from test_secondary_workbench import work


def sample():
    # Oppositely wound, coincident faces with separate vertices/materials.
    return {'model_sha256':'asset-a','bones':[{'name':'body'}],
        'geometry':{'positions_m':[[0,0,0],[1,0,0],[0,1,0],[0,0,0],[1,0,0],[0,1,0]],
            'triangles':[[0,1,2],[3,5,4]],'face_material_ids':[0,1],
            'material_names':['cloth','metal'],'skin_weights':[{'0':1.} for _ in range(6)]},
        'regions':[{'role':'cloth','face_ids':[0],'vertex_ids':[0,1,2]},
                   {'role':'accessory','face_ids':[1],'vertex_ids':[3,4,5]}]}


def reviewed(inv):
    return {'schema_version':1,'model_sha256':inv['model_sha256'],'geometry_sha256':inv['geometry_sha256'],
        'parts':[{'id':'cloth','category':'outer_cloth','layer':1,'face_ids':[0],'evidence':'fixture layer review'},
                 {'id':'trim','category':'flexible_trim','face_ids':[1],'evidence':'fixture trim review'}],
        'overlaps':[{'group_id':g['id'],'kind':'separate_layers','evidence':'fixture authored relation'} for g in inv['coincident_face_groups']]}


def test_coincident_geometry_never_welds_or_infers_rigid_parts():
    p=sample();before=copy.deepcopy(p);i=build_inventory(p)
    assert p==before and len(i['components'])==2
    assert len(i['coincident_vertex_groups'])==3 and len(i['coincident_face_groups'])==1
    assert i['coincident_face_groups'][0]['matching_skin_weights']
    assert not i['coincident_face_groups'][0]['morph_equivalence_verified']
    assert not i['topology_changed'] and not i['simulation_performed']
    assert not review_gate(i)['accepted']


def test_vertex_only_touching_triangles_are_separate_components():
    p=sample();p['geometry']['triangles'][1]=[0,4,5];p['geometry']['face_material_ids']=[0,0]
    assert len(build_inventory(p)['components'])==2


def test_shared_edge_merges_only_within_material():
    p=sample();p['geometry']['triangles'][1]=[0,2,4]
    assert len(build_inventory(p)['components'])==2
    p['geometry']['face_material_ids']=[0,0]
    assert len(build_inventory(p)['components'])==1


def test_stable_source_face_ids_do_not_make_changed_geometry_review_valid():
    p=sample();a=build_inventory(p);r=reviewed(a);p['geometry']['positions_m'][0][0]=.1;b=build_inventory(p)
    assert a['components'][0]['id']==b['components'][0]['id']
    with pytest.raises(ValueError,match='指纹'):review_gate(b,r)


def test_changed_weights_and_asset_invalidate_review():
    p=sample();a=build_inventory(p);p['geometry']['skin_weights'][3]={'0':.9}
    b=build_inventory(p);assert not b['coincident_face_groups'][0]['matching_skin_weights']
    with pytest.raises(ValueError):review_gate(b,reviewed(a))
    p=sample();p['model_sha256']='asset-b'
    assert build_inventory(p)['components'][0]['id']!=a['components'][0]['id']


@pytest.mark.parametrize('mutation',['missing_face','missing_relation'])
def test_incomplete_semantics_cannot_pass(mutation):
    i=build_inventory(sample());r=reviewed(i)
    r['parts' if mutation=='missing_face' else 'overlaps'].pop()
    assert not review_gate(i,r)['accepted']


@pytest.mark.parametrize('mutation',['duplicate_face','bad_face','duplicate_part','no_evidence','bad_layer','unknown_category','stale_relation'])
def test_invalid_explicit_review_is_rejected(mutation):
    i=build_inventory(sample());r=reviewed(i)
    if mutation=='duplicate_face':r['parts'][1]['face_ids']=[0]
    if mutation=='bad_face':r['parts'][1]['face_ids']=[999]
    if mutation=='duplicate_part':r['parts'][1]['id']='cloth'
    if mutation=='no_evidence':r['parts'][1]['evidence']=''
    if mutation=='bad_layer':r['parts'][0]['layer']=-1
    if mutation=='unknown_category':r['parts'][1]['category']='auto'
    if mutation=='stale_relation':r['overlaps'][0]['group_id']='stale'
    with pytest.raises(ValueError):review_gate(i,r)


def test_protected_body_cannot_be_relabelled_as_cloth():
    p=sample();p['regions'][0]['role']='body';i=build_inventory(p);r=reviewed(i)
    with pytest.raises(ValueError,match='受保护'):review_gate(i,r)
    r['parts'][0]['category']='protected_surface';assert review_gate(i,r)['accepted']


def test_shared_protected_vertex_cannot_become_rigid():
    p=sample();p['regions'][0]['role']='body';p['geometry']['triangles'][1]=[0,4,5]
    p['regions'][1]['vertex_ids']=[0,4,5];i=build_inventory(p);r=reviewed(i)
    r['parts'][0]['category']='protected_surface';r['parts'][1]['category']='rigid_ornament'
    with pytest.raises(ValueError,match='连接边界'):review_gate(i,r)


def test_full_semantic_review_is_not_contact_or_simulation_acceptance():
    i=build_inventory(sample());g=review_gate(i,reviewed(i));assert g['accepted']
    assert not g['collision_rules_verified'] and not g['attachment_verified'] and not g['simulation_accepted']


def test_required_review_blocks_before_native_pipeline_or_output(tmp_path,monkeypatch):
    from motionbridge import secondary_pipeline as pipe
    from motionbridge import secondary_plan
    root=tmp_path;source=root/'input.blend';source.write_bytes(b'untouched')
    profiles=root/'profiles/secondary_motion';profiles.mkdir(parents=True)
    (profiles/'test.json').write_text(json.dumps({'surface_segmentation':{'required_for_surface_dynamics':True}}))
    p={'surface_segmentation':{'required_for_surface_dynamics':True,'review_gate':{'accepted':False,'unclassified_faces':2,'unreviewed_overlap_groups':1}}}
    monkeypatch.setattr(secondary_plan,'compile_profile',lambda *args:p)
    monkeypatch.setattr(pipe,'RunSession',lambda *args:pytest.fail('native session must not start'))
    with pytest.raises(ValueError,match='分类未通过'):pipe.run_pipeline(root,source,root/'output/secondary_motion/test','test')
    assert not (root/'output').exists() and source.read_bytes()==b'untouched'


def test_required_flag_cannot_pass_without_review():
    with pytest.raises(ValueError):require_review({'surface_segmentation':{'required_for_surface_dynamics':True}})
    require_review({'surface_segmentation':{'required_for_surface_dynamics':True,'review_gate':{'accepted':True}}})


def test_real_rin_inventory_exposes_coarse_material_ambiguity():
    from motionbridge.secondary_plan import compile_profile
    root=Path(__file__).resolve().parents[1]
    profile=json.loads((root/'profiles/secondary_motion/rin.json').read_text(encoding='utf8'))
    plan=compile_profile(profile,root);s=plan['surface_segmentation'];i=s['inventory']
    assert i['face_count']==47384 and len(i['components'])==1395
    assert len(i['coincident_face_groups'])==9418
    assert s['required_for_surface_dynamics'] and s['review_gate']['accepted']
    assert not plan['source_surface_map']['simulation_ready']
    assigned=[f for c in i['components'] for f in c['face_ids']]
    assert len(assigned)==len(set(assigned))==i['face_count']
    assert len(plan['protected_bone_ids'])==48


@pytest.mark.parametrize('compiled',['pending','invalid'])
def test_catalog_and_start_explain_block_without_blocking_hair(work,monkeypatch,compiled):
    from test_secondary_workbench import start
    from motionbridge import secondary_plan
    from motionbridge.secondary_cache import atomic_json,read_json
    manager,client,scene,pool=work;path=manager.root/'profiles/secondary_motion/rin.json'
    profile=read_json(path);profile['surface_segmentation']={'required_for_surface_dynamics':True};atomic_json(path,profile)
    def compile(*args):
        if compiled=='invalid':raise ValueError('分类指纹不匹配')
        return {'surface_segmentation':{'required_for_surface_dynamics':True,'review_gate':{'accepted':False,'unclassified_faces':2,'unreviewed_overlap_groups':1}}}
    monkeypatch.setattr(secondary_plan,'compile_profile',compile)
    catalog=manager.profiles()[0]
    assert catalog['ready'] and catalog['hair'] and not catalog['surface']
    response=start(work,mode='surface')
    assert response.status_code==409 and '分类' in response.json()['detail'] and not pool.calls
    assert start(work,mode='hair').status_code==202
