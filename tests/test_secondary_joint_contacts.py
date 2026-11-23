import copy
import itertools
import numpy as np
import pytest
from attachment_fixture import make_fixture
from motionbridge.secondary_joint_contacts import JointContactSystem, segment_closest, joint_solver_status
from motionbridge.secondary_attachment_driver import exp_rotation


def fixture(mode='surface_rigid',floor=-.2,body=False):
    p,r,b=make_fixture(mode)
    rules=[{'part_a':'ornament','part_b':'cloth','side_a':1,'side_b':-1,'clearance_m':.01},
           {'part_a':'cloth','part_b':'protected','side_a':1,'side_b':-1,'clearance_m':.01},
           {'part_a':'ornament','part_b':'protected','side_a':1,'side_b':-1,'clearance_m':.01}]
    system=JointContactSystem(p,r,b,rules,floor_z=floor)
    source=system.attachment.rest.copy()
    if body:source[9:12]=[[-.3,-.3,-.03],[.3,-.3,-.03],[0,.4,-.03]]
    kwargs={'rigid_inverse_mass':{'ornament':1.},'rigid_inverse_inertia':{'ornament':np.eye(3)*.1}} if mode=='surface_rigid' else {}
    if mode=='bone_socket':kwargs={'bone_rest':{0:np.eye(4)},'bone_current':{0:np.eye(4)}}
    return system,system.begin_frame(source,np.ones(len(source)),**kwargs)


def move(frame,delta):
    frame.x[:3]+=delta
    if 'ornament' in frame.rigid:frame.rigid['ornament']['center']+=delta
    else:frame.x[3:6]+=delta


def assert_pass(report):
    assert report['discrete_constraints_converged'],report
    assert report['candidate_committed'] and not report['rolled_back']
    assert not report['physics_accepted'] and not report['continuous_collision_checked']
    assert not report['original_before_quality_gate_checked']
    assert report['candidate_audit']['proper_crossings']==0


def test_rest_does_not_move_or_claim_production_quality():
    s,f=fixture();before=f.positions();r=s.solve(f);assert_pass(r)
    assert r['iterations_used']==0
    np.testing.assert_array_equal(f.positions(),before)
    assert not joint_solver_status()['production_ready']


@pytest.mark.parametrize('mode',['surface_rigid','surface_seam'])
def test_body_contact_and_attachment_are_solved_on_same_state(mode):
    s,f=fixture(mode,body=True);original=f.source.copy();move(f,[0,0,-.05]);incoming=f.positions()
    r=s.solve(f,iterations=128,tolerance_m=2e-5);assert_pass(r)
    assert r['initial_prediction_audit']['maximum_contact_deficit_m']>.02
    assert r['candidate_audit']['maximum_contact_deficit_m']<=2e-5
    assert np.linalg.norm(f.positions()[:3]-incoming[:3])>.02
    assert np.linalg.norm(f.positions()[3:6]-incoming[3:6])>.01
    np.testing.assert_array_equal(f.source,original)
    np.testing.assert_array_equal(f.positions()[9:12],original[9:12])


@pytest.mark.parametrize('mode',['surface_rigid','surface_seam'])
def test_floor_rechecks_attachment_and_lengths_after_last_push(mode):
    s,f=fixture(mode,floor=0.);move(f,[0,0,-.04]);r=s.solve(f,iterations=128,tolerance_m=2e-5)
    assert_pass(r);assert f.positions()[:,2].min()>=-2e-5
    assert r['candidate_audit']['attachment_residual_m']<=2e-5
    assert r['candidate_audit']['edge_length_error_m']<=2e-5
    if mode=='surface_rigid':
        x=f.positions()[3:6];rest=s.attachment.rest[3:6]
        np.testing.assert_allclose(np.linalg.norm(x[:,None]-x[None],axis=-1),np.linalg.norm(rest[:,None]-rest[None],axis=-1),atol=1e-12)


def test_unsatisfiable_fixed_floor_rolls_back_entire_prediction():
    s,f=fixture(floor=.03);move(f,[.01,0,-.01]);before=f.positions().copy();rigid=copy.deepcopy(f.rigid)
    r=s.solve(f,iterations=3)
    assert not r['candidate_committed'] and r['rolled_back'] and r['iterations_used']==3
    assert r['candidate_audit']['signed_deficits_m']['ground']>=.03
    np.testing.assert_array_equal(f.positions(),before)
    np.testing.assert_array_equal(f.rigid['ornament']['center'],rigid['ornament']['center'])


def test_iteration_budget_failure_does_not_publish_partial_improvement():
    s,f=fixture(body=True);move(f,[0,0,-.05]);before=f.positions().copy()
    r=s.solve(f,iterations=1,tolerance_m=1e-9)
    assert r['rolled_back'] and not r['candidate_committed'];np.testing.assert_array_equal(f.positions(),before)


def test_fixed_bone_socket_cannot_be_pushed_by_cloth():
    s,f=fixture('bone_socket',floor=0.);center=f.rigid['ornament']['center'].copy();f.x[:3,2]-=.04
    r=s.solve(f,iterations=128,tolerance_m=2e-5);assert_pass(r)
    np.testing.assert_array_equal(f.rigid['ornament']['center'],center)


def test_predictor_cannot_change_protected_body():
    s,f=fixture();f.x[9,0]+=.01
    with pytest.raises(ValueError,match='fixed'):s.solve(f)


def test_wrong_system_frame_rejected():
    s,f=fixture();other,_=fixture()
    with pytest.raises(ValueError,match='different'):other.solve(f)


@pytest.mark.parametrize('budget',[0,-1,257,True,1.5])
def test_invalid_iteration_budget(budget):
    s,f=fixture()
    with pytest.raises(ValueError):s.solve(f,iterations=budget)


@pytest.mark.parametrize('tol',[0,-1,float('nan'),float('inf'),True])
def test_invalid_tolerance(tol):
    s,f=fixture()
    with pytest.raises(ValueError):s.solve(f,tolerance_m=tol)


def test_missing_pair_cannot_hide_accessory_body_or_hair_channel():
    p,r,b=make_fixture()
    with pytest.raises(ValueError,match='Missing physical'):JointContactSystem(p,r,b,[])


def test_feature_budget_stops_before_quadratic_allocation():
    p,r,b=make_fixture()
    rule={'part_a':'ornament','part_b':'cloth','side_a':1,'side_b':-1,'clearance_m':.01}
    with pytest.raises(ValueError,match='budget'):JointContactSystem(p,r,b,[rule],max_feature_tests=1)


@pytest.mark.parametrize('sides',[(0,1),(True,-1),(1.,-1)])
def test_no_silent_orientation_inference(sides):
    p,r,b=make_fixture();row={'part_a':'ornament','part_b':'cloth','side_a':sides[0],'side_b':sides[1],'clearance_m':.01}
    with pytest.raises(ValueError,match='sides'):JointContactSystem(p,r,b,[row])


def test_exact_crossing_not_reported_as_clear_when_vertices_miss_faces():
    s,f=fixture('surface_seam');f.x[3:6]=[[-.3,0,-.1],[.3,0,-.1],[0,0,.1]]
    report=s.audit(f,[])
    assert report['proper_crossings']>0


def test_coplanar_overlap_is_caught_by_distance_not_only_proper_crossings():
    s,f=fixture('surface_seam');f.x[3:6]=f.x[:3]
    report=s.audit(f,[])
    assert report['proper_crossings']==0 and report['maximum_contact_deficit_m']>=.01


def test_degenerate_geometry_failure_restores_all_secondary_state():
    s,f=fixture('surface_seam');f.x[0:3]=[0,0,0];before=f.positions().copy()
    with pytest.raises(ValueError,match='退化'):s.solve(f)
    np.testing.assert_array_equal(f.positions(),before)


def test_edge_edge_closest_features_are_interior_and_exact():
    a,b,c,d=np.array([[-1,0,0],[1,0,0],[0,-1,.003],[0,1,.003]],float)
    s,t,p,q=segment_closest(a,b,c,d)
    assert s==.5 and t==.5;assert np.linalg.norm(p-q)==pytest.approx(.003)


def test_parallel_edges_choose_finite_segment_not_infinite_line():
    a,b,c,d=np.array([[0,0,0],[1,0,0],[2,.01,0],[3,.01,0]],float)
    s,t,p,q=segment_closest(a,b,c,d)
    assert s==1 and t==0;assert np.linalg.norm(p-q)==pytest.approx(np.sqrt(1.0001))


def test_two_sided_contact_gradient_moves_target_as_well_as_point():
    s,f=fixture('surface_seam');f.x[3,2]=.005
    before=f.positions().copy();c={'kind':'vf','ids':(3,0,1,2),'side':1,'gap':.01}
    s._project(f,c)
    assert f.x[3,2]>before[3,2] and f.x[0,2]<before[0,2]
    np.testing.assert_allclose((f.x-before).sum(0),[0,0,0],atol=1e-10)


def test_assembly_guard_stays_closed_despite_discrete_core():
    from motionbridge.secondary_contact_policy import require_contact_policy
    with pytest.raises(ValueError,match='禁止回退'):require_contact_policy({'secondary_contact_rules':{'simulation_ready':True}})


def test_hair_is_a_real_triangle_contact_channel_not_an_excluded_category():
    from motionbridge.secondary_segmentation import build_inventory,review_gate,_sha
    p,r,b=make_fixture()
    g=p['geometry'];g['positions_m'].extend([[-.1,-.1,.055],[.1,-.1,.055],[0,.1,.055],[.7,0,.055],[.8,0,.055],[.7,.1,.055]])
    g['triangles'].extend([[12,13,14],[15,16,17]]);g['skin_weights'].extend([{'0':1.} for _ in range(6)])
    g['material_names'].append('fixture_hair');g['face_material_ids'].extend([3,3])
    p['regions'].append({'role':'hair','face_ids':[4,5],'vertex_ids':list(range(12,18))})
    r['parts'].append({'id':'hair','category':'hair','face_ids':[4,5],'evidence':'explicit engineering hair patch'})
    inventory=build_inventory(p);r['geometry_sha256']=inventory['geometry_sha256']
    p['surface_segmentation'].update(inventory=inventory,review_gate=review_gate(inventory,r))
    b.update(geometry_sha256=inventory['geometry_sha256'],review_sha256=_sha(r))
    b['attachments'].append({'part_id':'hair','mode':'source_root','anchors':[{'vertex_id':15}],
                            'limits':{'translation_m':0,'rotation_rad':0},'evidence':'fixture root'})
    heights={'cloth':0,'protected':-.03,'ornament':.02,'hair':.055}
    rules=[{'part_a':a,'part_b':bb,'side_a':1 if heights[a]>heights[bb] else -1,
            'side_b':1 if heights[bb]>heights[a] else -1,'clearance_m':.01} for a,bb in itertools.combinations(heights,2)]
    s=JointContactSystem(p,r,b,rules,floor_z=-.2)
    f=s.begin_frame(s.attachment.rest,np.ones(18),rigid_inverse_mass={'ornament':1.},rigid_inverse_inertia={'ornament':np.eye(3)*.1})
    f.x[12:15,2]-=.03;incoming=f.positions().copy()
    report=s.solve(f,iterations=128,tolerance_m=2e-5);assert_pass(report)
    channel='hair/rigid_ornament'
    assert report['initial_prediction_audit']['signed_deficits_m'][channel]>.004
    assert report['candidate_audit']['signed_deficits_m'][channel]<=2e-5
    assert np.linalg.norm(f.positions()[12:15]-incoming[12:15])>.001
    assert np.linalg.norm(f.positions()[3:6]-incoming[3:6])>.001


def test_edge_edge_projection_shares_reaction():
    s,f=fixture('surface_seam');f.x[:2]=[[-1,0,0],[1,0,0]];f.x[3:5]=[[0,-1,.003],[0,1,.003]]
    before=f.x.copy();c={'kind':'ee','ids':(0,1,3,4),'s':.5,'t':.5,'normal':np.array([0,0,-1.]),'gap':.01}
    s._project(f,c)
    assert f.x[0,2]<before[0,2] and f.x[3,2]>before[3,2]
    assert s._value(f,c)>=-1e-10
    np.testing.assert_allclose((f.x-before).sum(0),0.,atol=1e-10)


def test_exception_after_projection_rolls_back_not_just_initial_validation(monkeypatch):
    s,f=fixture(floor=0.);move(f,[0,0,-.04]);before=f.positions().copy();audit=s.audit;calls=[]
    def failing_audit(*args):
        calls.append(1)
        if len(calls)>1:raise ValueError('injected post-projection audit failure')
        return audit(*args)
    monkeypatch.setattr(s,'audit',failing_audit)
    with pytest.raises(ValueError,match='post-projection'):s.solve(f)
    np.testing.assert_array_equal(f.positions(),before)


def test_structurally_valid_local_seam_is_not_silently_ignored():
    p,r,b=make_fixture();b['seams']=[{'child_part':'ornament','target_part':'cloth','face_pairs':[[1,0]],
       'max_depth_m':.001,'max_slip_m':.001,'evidence':'declared local seam'}]
    with pytest.raises(ValueError,match='alias/seam'):JointContactSystem(p,r,b,[])


def test_policy_reports_discrete_core_without_unlocking_production():
    from motionbridge.secondary_contact_policy import compile_contact_policy,POLICY
    p,r,b=make_fixture()
    result=compile_contact_policy(p,{'surface_segmentation':{'review':r},'secondary_contact_rules':{'schema_version':1,'policy':POLICY,'bindings':b}})
    assert result['joint_solver']['discrete_core_implemented']
    assert not result['solver_implemented'] and not result['simulation_ready']


def test_no_self_collision_or_continuous_claim_from_single_frame_success():
    s,f=fixture();report=s.solve(f)
    assert not report['self_collision_checked'] and not report['continuous_collision_checked']
    assert report['scope']=='explicitly_oriented_distinct_part_pairs_and_floor'


def test_contact_features_include_edge_endpoints_after_nearest_feature_switch():
    s,f=fixture('surface_seam');f.x[3:6]+=[.002,0,-.015]
    contacts=[c for c in s._features(f) if c['kind']=='ee']
    assert any(c['s'] in (0.,1.) or c['t'] in (0.,1.) for c in contacts)
