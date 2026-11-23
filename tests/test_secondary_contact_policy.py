import copy
import json
from pathlib import Path

import pytest
from test_secondary_workbench import work
from motionbridge.secondary_segmentation import build_inventory,review_gate,_sha,require_review
from motionbridge.secondary_contact_policy import (POLICY,contact_rule,policy_definition,
    validate_bindings,compile_contact_policy,require_contact_policy)


def fixture_data():
    p={'model_sha256':'asset','protected_bone_ids':[0],'bones':[{'name':'body'}],
       'geometry':{'positions_m':[[0,0,0],[1,0,0],[0,1,0],[0,0,.01],[1,0,.01],[0,1,.01]],
       'triangles':[[0,1,2],[3,4,5]],'face_material_ids':[0,1],'material_names':['cloth','ornament'],
       'skin_weights':[{'0':1.} for _ in range(6)]},
       'regions':[{'role':'cloth','face_ids':[0],'vertex_ids':[0,1,2]},
                  {'role':'accessory','face_ids':[1],'vertex_ids':[3,4,5]}]}
    inv=build_inventory(p)
    review={'schema_version':1,'model_sha256':'asset','geometry_sha256':inv['geometry_sha256'],
        'parts':[{'id':'coat','category':'outer_cloth','layer':1,'face_ids':[0],'evidence':'synthetic cloth'},
                 {'id':'button','category':'rigid_ornament','face_ids':[1],'evidence':'synthetic solid'}],'overlaps':[]}
    p['surface_segmentation']={'inventory':inv,'review_gate':review_gate(inv,review),'required_for_surface_dynamics':True}
    b={'schema_version':1,'model_sha256':'asset','geometry_sha256':inv['geometry_sha256'],'review_sha256':_sha(review),
       'attachments':[{'part_id':'coat','mode':'source_root','anchors':[{'vertex_id':0}],
           'limits':{'translation_m':0,'rotation_rad':0},'evidence':'original source root'},
           {'part_id':'button','mode':'surface_rigid','target_part':'coat',
            'anchors':[{'vertex_id':i+3,'target_face_id':0,'barycentric':[int(i==j) for j in range(3)],'rest_offset_m':[0,0,.01]} for i in range(3)],
            'limits':{'translation_m':.002,'rotation_rad':.1},'evidence':'explicit surface patch'}],
       'aliases':[],'seams':[]}
    return p,review,b


def test_pair_matrix_is_exhaustive_symmetric_and_never_excludes_raw_audit():
    pairs=policy_definition()['pair_rules'];assert len(pairs)==36
    assert len({tuple(p['categories']) for p in pairs})==36
    for p in pairs:
        a,b=p['categories'];assert contact_rule(a,b)==contact_rule(b,a)
        assert not p['raw_audit_excluded'] and not p['rigid_vertices_independently_movable']


@pytest.mark.parametrize('category',['inner_cloth','outer_cloth','hair','rigid_ornament','flexible_trim'])
def test_body_and_ground_never_receive_contact_corrections(category):
    for fixed in ['protected_surface','ground']:
        r=contact_rule(category,fixed)
        assert r['response']=='dynamic_side_only' and r['potentially_movable_categories']==[category]


def test_fixed_fixed_conflict_is_reported_and_not_root_lifted():
    assert contact_rule('ground','protected_surface')['response']=='report_fixed_conflict'
    assert 'no_root_lift' in policy_definition()['ground']


def test_cloth_ornament_joint_correction_has_no_one_way_postsolve_follow():
    r=contact_rule('outer_cloth','rigid_ornament')
    assert r['response']=='joint_response_in_free_dofs' and r['respect_attachment_dofs']
    assert 'never_postsolve_one_way_copy' in policy_definition()['surface_attachment_response']


def test_display_surface_needs_mapping_and_unknown_category_is_not_ignored():
    assert contact_rule('display_overlay','hair')['response']=='canonical_surface_only_after_verified_mapping'
    with pytest.raises(ValueError):contact_rule('unclassified','hair')


def test_explicit_patch_keeps_original_gap_and_is_only_structurally_valid():
    p,r,b=fixture_data();before=copy.deepcopy((p,r,b));result=validate_bindings(p,r,b)
    assert (p,r,b)==before and result['structurally_valid'] and result['attachments']==2
    assert not result['asset_mapping_verified'] and result['requires_native_geometry_and_motion_proof']


@pytest.mark.parametrize('mutation',['missing_root','nearest_neighbor','stale','foreign_vertex','foreign_face',
    'bad_barycentric','snap_offset','nan_limit','negative_limit','unbounded_angle','moving_source_pin',
    'zero_area_child','zero_area_target','wrong_bone','unknown_target','missing_evidence'])
def test_invalid_attachments_fail_closed(mutation):
    p,r,b=fixture_data();row=b['attachments'][1]
    if mutation=='missing_root':b['attachments'].pop(0)
    if mutation=='nearest_neighbor':row['mode']='nearest_cloth'
    if mutation=='stale':b['review_sha256']='old'
    if mutation=='foreign_vertex':row['anchors'][0]['vertex_id']=0
    if mutation=='foreign_face':row['anchors'][0]['target_face_id']=1
    if mutation=='bad_barycentric':row['anchors'][0]['barycentric']=[1,1,1]
    if mutation=='snap_offset':row['anchors'][0]['rest_offset_m']=[0,0,0]
    if mutation=='nan_limit':row['limits']['rotation_rad']=float('nan')
    if mutation=='negative_limit':row['limits']['translation_m']=-1
    if mutation=='unbounded_angle':row['limits']['rotation_rad']=4
    if mutation=='moving_source_pin':b['attachments'][0]['limits']['translation_m']=.1
    if mutation=='zero_area_child':
        p['geometry']['positions_m'][5]=[2,0,.01];row['anchors'][2].update(barycentric=[0,0,1],rest_offset_m=[2,-1,.01])
    if mutation=='zero_area_target':
        for a in row['anchors']:
            a['barycentric']=[1,0,0];a['rest_offset_m']=p['geometry']['positions_m'][a['vertex_id']]
    if mutation=='wrong_bone':
        row.update(mode='bone_socket',bone_id=99,limits={'translation_m':0,'rotation_rad':0},anchors=[{'vertex_id':v} for v in [3,4,5]])
        del row['target_part']
    if mutation=='unknown_target':row['target_part']='unknown'
    if mutation=='missing_evidence':row['evidence']=''
    with pytest.raises(ValueError):validate_bindings(p,r,b)


def test_attachment_ownership_cycle_is_not_a_collision_loop():
    p,r,b=fixture_data();r['parts'][0]['category']='flexible_trim';r['parts'][1]['category']='flexible_trim'
    p['surface_segmentation']['review_gate']=review_gate(p['surface_segmentation']['inventory'],r)
    b['review_sha256']=_sha(r)
    b['attachments'][1]['mode']='surface_seam'
    b['attachments'][0].update(mode='surface_seam',target_part='button',anchors=[
        {'vertex_id':i,'target_face_id':1,'barycentric':[int(i==j) for j in range(3)],'rest_offset_m':[0,0,-.01]} for i in range(3)])
    with pytest.raises(ValueError,match='成环'):validate_bindings(p,r,b)


def test_valid_bone_socket_is_fixed_and_source_bone_is_not_rewritten():
    p,r,b=fixture_data();row=b['attachments'][1]
    row.update(mode='bone_socket',bone_id=0,limits={'translation_m':0,'rotation_rad':0},anchors=[{'vertex_id':v} for v in [3,4,5]])
    del row['target_part']
    assert validate_bindings(p,r,b)['structurally_valid']


@pytest.mark.parametrize('mutation',['whole_material','too_deep','wrong_neighborhood','duplicate','nan'])
def test_seam_exemptions_cannot_disable_a_whole_contact_channel(mutation):
    p,r,b=fixture_data()
    seam={'child_part':'button','target_part':'coat','face_pairs':[[1,0]],'max_depth_m':.001,'max_slip_m':.001,'evidence':'local seam'}
    b['seams']=[seam]
    if mutation=='whole_material':seam['material']='all'
    if mutation=='too_deep':seam['max_depth_m']=.05
    if mutation=='wrong_neighborhood':seam['face_pairs']=[[0,1]]
    if mutation=='duplicate':seam['face_pairs']=[[1,0],[1,0]]
    if mutation=='nan':seam['max_slip_m']=float('nan')
    with pytest.raises(ValueError):validate_bindings(p,r,b)


def test_local_seam_is_bounded_and_does_not_change_raw_audit_policy():
    p,r,b=fixture_data();b['seams']=[{'child_part':'button','target_part':'coat','face_pairs':[[1,0]],'max_depth_m':.001,'max_slip_m':.001,'evidence':'local seam'}]
    assert validate_bindings(p,r,b)['seam_face_pairs']==1
    assert all(not p['raw_audit_excluded'] for p in policy_definition()['pair_rules'])


def test_duplicate_with_different_weights_cannot_be_silently_welded():
    p,r,b=fixture_data();p['geometry']['positions_m'][3:]=copy.deepcopy(p['geometry']['positions_m'][:3])
    p['geometry']['skin_weights'][3]={'0':.9};inv=build_inventory(p)
    r['geometry_sha256']=inv['geometry_sha256'];r['parts'][1]['category']='display_overlay'
    r['overlaps']=[{'group_id':inv['coincident_face_groups'][0]['id'],'kind':'display_duplicate','evidence':'unsafe duplicate'}]
    p['surface_segmentation'].update(inventory=inv,review_gate=review_gate(inv,r))
    b.update(geometry_sha256=inv['geometry_sha256'],review_sha256=_sha(r))
    b['attachments'].pop();b['aliases']=[{'group_id':r['overlaps'][0]['group_id'],'display_face':1,'canonical_face':0,'evidence':'test'}]
    with pytest.raises(ValueError,match='不同蒙皮'):validate_bindings(p,r,b)


def test_semantic_pass_and_valid_bindings_still_cannot_unlock_old_solver():
    p,r,b=fixture_data();profile={'surface_segmentation':{'review':r},'secondary_contact_rules':{'schema_version':1,'policy':POLICY,'bindings':b}}
    result=compile_contact_policy(p,profile);p['secondary_contact_rules']=result
    assert result['rules_defined'] and result['asset_bindings']['structurally_valid']
    assert not result['simulation_ready'] and not result['solver_implemented']
    with pytest.raises(ValueError,match='旧饰品'):require_review(p)
    # Even an edited status flag must not bypass the absent solver adapter.
    p['secondary_contact_rules']['simulation_ready']=True
    with pytest.raises(ValueError):require_contact_policy(p)


def test_real_rin_definition_and_classification_do_not_unlock_missing_constraints():
    from motionbridge.secondary_plan import compile_profile
    root=Path(__file__).resolve().parents[1];profile=json.loads((root/'profiles/secondary_motion/rin.json').read_text(encoding='utf8'))
    p=compile_profile(profile,root);rules=p['secondary_contact_rules']
    assert rules['rules_defined'] and len(rules['definition']['pair_rules'])==36
    assert rules['asset_bindings'] is None and not rules['simulation_ready']
    assert 'model_attachments_and_surface_mapping_missing' in rules['blockers']
    with pytest.raises(ValueError,match='碰撞|附着|接触'):require_review(p)


def test_shared_protected_anchor_cannot_be_moved_with_garment():
    p,r,b=fixture_data()
    p['regions'].append({'role':'body','face_ids':[],'vertex_ids':[3]})
    with pytest.raises(ValueError,match='受保护顶点'):validate_bindings(p,r,b)


@pytest.mark.parametrize('mapped',[True,False])
def test_display_mapping_is_explicit_and_does_not_certify_dynamic_equivalence(mapped):
    p,r,b=fixture_data();p['geometry']['positions_m'][3:]=copy.deepcopy(p['geometry']['positions_m'][:3])
    inv=build_inventory(p);r['geometry_sha256']=inv['geometry_sha256'];r['parts'][1]['category']='display_overlay'
    gid=inv['coincident_face_groups'][0]['id']
    r['overlaps']=[{'group_id':gid,'kind':'display_duplicate','evidence':'fixture equivalent geometry'}]
    p['surface_segmentation'].update(inventory=inv,review_gate=review_gate(inv,r))
    b.update(geometry_sha256=inv['geometry_sha256'],review_sha256=_sha(r));b['attachments'].pop()
    if mapped:
        b['aliases']=[{'group_id':gid,'display_face':1,'canonical_face':0,'evidence':'fixture mapping'}]
        status=validate_bindings(p,r,b)
        assert status['aliases']==1 and not status['asset_mapping_verified']
    else:
        with pytest.raises(ValueError,match='映射'):validate_bindings(p,r,b)


def test_definition_can_precede_segmentation_but_cannot_enable_execution():
    p,_,_=fixture_data();del p['surface_segmentation']
    profile={'secondary_contact_rules':{'schema_version':1,'policy':POLICY,'bindings':None}}
    result=compile_contact_policy(p,profile)
    assert result['rules_defined'] and 'semantic_segmentation_not_accepted' in result['blockers']
    profile['secondary_contact_rules']['enable_solver']=True
    with pytest.raises(ValueError):compile_contact_policy(p,profile)


@pytest.mark.parametrize('include_segmentation',[True,False])
def test_workbench_cannot_unlock_old_solver_after_rule_declaration(work,monkeypatch,include_segmentation):
    from motionbridge import secondary_plan
    from motionbridge.secondary_cache import read_json,atomic_json
    from test_secondary_workbench import start
    manager,client,_,pool=work;path=manager.root/'profiles/secondary_motion/rin.json'
    profile=read_json(path);profile['secondary_contact_rules']={'schema_version':1,'policy':POLICY,'bindings':None}
    atomic_json(path,profile)
    plan={'secondary_contact_rules':{'rules_defined':True,'simulation_ready':False}}
    if include_segmentation:plan['surface_segmentation']={'required_for_surface_dynamics':True,'review_gate':{'accepted':True}}
    monkeypatch.setattr(secondary_plan,'compile_profile',lambda *args:plan)
    catalog=manager.profiles()[0]
    assert catalog['ready'] and catalog['hair'] and not catalog['surface']
    response=start(work,mode='surface')
    assert response.status_code==409 and '联合求解' in response.json()['detail'] and not pool.calls
    assert start(work,mode='hair').status_code==202
