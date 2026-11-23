from dataclasses import replace
import copy
import numpy as np
import pytest
from test_secondary_joint_contacts import fixture,move
from motionbridge.secondary_prewarm import WarmupSettings,run_prewarm,diagnose_initial_state,prewarm_status


def settings(**kwargs):
    return replace(WarmupSettings(dt=1/30,max_steps=30,ramp_steps=0,window_steps=15,minimum_full_force_steps=15),**kwargs)


def assert_pass(report):
    assert report['core_prewarm_passed'],report
    assert report['candidate_committed'] and not report['rolled_back']
    assert report['prewarm_convergence']['passed'] and report['force_balance_probe']['passed']
    assert not report['physics_accepted'] and not report['original_before_quality_gate_checked']


def test_rest_without_force_requires_a_real_trailing_window():
    s,f=fixture();before=f.positions().copy();report=run_prewarm(s,f,settings=settings(),acceleration=[0,0,0])
    assert_pass(report);assert report['steps_completed']==15
    np.testing.assert_allclose(f.positions(),before,atol=1e-15)
    assert report['prewarm_convergence']['p95_limit_m_s']==.02
    assert report['prewarm_convergence']['max_limit_m_s']==.1


def test_floor_supported_cloth_and_rigid_settle_under_real_gravity():
    s,f=fixture(floor=0.);before=f.source.copy();report=run_prewarm(s,f,settings=settings())
    assert_pass(report)
    assert report['force_balance_probe']['damping_per_second']==0.
    assert f.positions()[:,2].min()>=-1e-5
    np.testing.assert_array_equal(f.positions()[[6,9,10,11]],before[[6,9,10,11]])
    np.testing.assert_array_equal(f.source,before)


def test_initial_overlap_repaired_once_and_shift_reported_separately():
    s,f=fixture(floor=0.);move(f,[0,0,-.03]);before=f.positions().copy()
    report=run_prewarm(s,f,settings=settings(),acceleration=[0,0,0]);assert_pass(report)
    assert .029<report['initial_shift_max_m']<.031
    assert report['initial_diagnosis']['source_state_audit']['maximum_contact_deficit_m']>=.03
    assert np.linalg.norm(f.positions()-before)>.02


def test_initial_large_correction_rejected_without_expanding_budget():
    s,f=fixture(floor=0.);move(f,[0,0,-.03]);before=f.positions().copy()
    report=run_prewarm(s,f,settings=settings(max_initial_shift_m=.005),acceleration=[0,0,0])
    assert report['reason']=='initial_correction_budget_exceeded' and report['steps_completed']==0
    np.testing.assert_array_equal(f.positions(),before)


def test_fixed_ground_conflict_stops_before_projection(monkeypatch):
    s,f=fixture(floor=.03);before=f.positions().copy()
    def forbidden(*args,**kwargs):raise AssertionError('Fixed conflict should stop before solving')
    monkeypatch.setattr(s,'solve',forbidden)
    report=run_prewarm(s,f,settings=settings())
    assert report['reason']=='initial_fixed_constraint_conflict' and report['initial_projection'] is None
    assert not report['initial_diagnosis']['pin_release_performed']
    np.testing.assert_array_equal(f.positions(),before)


def test_damping_cannot_disguise_unsupported_gravity_as_equilibrium():
    s,f=fixture(floor=-10.);before=f.positions().copy()
    # With a very small dt and huge damping, terminal drift passes both speed
    # limits; the independent undamped force-response probe must still fail.
    cfg=settings(dt=1/1000,max_steps=501,window_steps=500,minimum_full_force_steps=500,damping_per_second=100000.)
    report=run_prewarm(s,f,settings=cfg)
    assert report['velocity_convergence']['passed'],report
    assert not report['prewarm_convergence']['passed']
    assert report['reason']=='full_force_equilibrium_not_reached'
    assert report['force_balance_probe']['maximum_unbalanced_acceleration_m_s2']>9.
    np.testing.assert_array_equal(f.positions(),before)


def test_quiet_ramp_does_not_count_as_full_force_window():
    s,f=fixture();report=run_prewarm(s,f,settings=settings(ramp_steps=5,max_steps=20),acceleration=[0,0,0])
    assert_pass(report);assert report['steps_completed']==20 and report['full_force_steps']==15
    assert all(not row['full_force_window_eligible'] for row in report['history'][:5])


def test_initial_velocity_is_not_reset_to_manufacture_quiet_frames():
    s,f=fixture(floor=-10.);before=f.positions().copy();v=np.zeros_like(f.x);v[:3,1]=.2
    report=run_prewarm(s,f,settings=settings(damping_per_second=0.),acceleration=[0,0,0],
        initial_vertex_velocity=v,initial_rigid_velocity={'ornament':[0,.2,0]})
    assert report['reason']=='warmup_budget_exhausted',report
    assert report['prewarm_convergence']['maximum_vertex_speed_m_s']>.19
    np.testing.assert_array_equal(f.positions(),before)


def test_cancellation_mid_warmup_restores_initial_prediction():
    s,f=fixture(floor=0.);move(f,[0,0,-.03]);before=f.positions().copy();calls=[]
    def cancelled():calls.append(1);return len(calls)>4
    report=run_prewarm(s,f,settings=settings(),cancelled=cancelled)
    assert report['reason']=='cancelled' and report['steps_completed']==3
    np.testing.assert_array_equal(f.positions(),before)


def test_projection_failure_no_retry_and_whole_warmup_rollback(monkeypatch):
    s,f=fixture(floor=0.);before=f.positions().copy();original=s.solve;calls=[]
    def fail_after_initial(frame,**kwargs):
        calls.append(1)
        if len(calls)>1:return {'discrete_constraints_converged':False,'reason':'controlled conflict'}
        return original(frame,**kwargs)
    monkeypatch.setattr(s,'solve',fail_after_initial)
    report=run_prewarm(s,f,settings=settings())
    assert report['reason']=='warmup_joint_projection_failed' and len(calls)==2
    np.testing.assert_array_equal(f.positions(),before)


def test_exception_after_prediction_restores_state(monkeypatch):
    s,f=fixture(floor=0.);before=f.positions().copy();original=s.solve;calls=[]
    def fail_after_initial(frame,**kwargs):
        calls.append(1)
        if len(calls)>1:raise ValueError('injected failure')
        return original(frame,**kwargs)
    monkeypatch.setattr(s,'solve',fail_after_initial)
    with pytest.raises(ValueError,match='injected'):run_prewarm(s,f,settings=settings())
    np.testing.assert_array_equal(f.positions(),before)


@pytest.mark.parametrize('change',[{'dt':0},{'dt':.1},{'dt':float('nan')},{'max_steps':True},{'max_steps':601},
    {'ramp_steps':-1},{'window_steps':1},{'window_steps':5},{'minimum_full_force_steps':14},
    {'max_steps':14},{'damping_per_second':-1},{'projection_tolerance_m':0},
    {'max_initial_shift_m':0},{'force_balance_limit_m_s2':float('inf')},{'projection_iterations':257}])
def test_invalid_settings_rejected(change):
    with pytest.raises(ValueError):settings(**change).validate()


def test_protected_vertex_cannot_receive_velocity():
    s,f=fixture();v=np.zeros_like(f.x);v[9,0]=1
    with pytest.raises(ValueError,match='independent'):run_prewarm(s,f,settings=settings(),initial_vertex_velocity=v)


def test_fixed_socket_cannot_receive_linear_velocity():
    s,f=fixture('bone_socket')
    with pytest.raises(ValueError,match='Fixed rigid'):run_prewarm(s,f,settings=settings(),initial_rigid_velocity={'ornament':[1,0,0]})


def test_excessive_angular_step_rolls_back_instead_of_aliasing_velocity():
    s,f=fixture();before=f.positions().copy()
    with pytest.raises(ValueError,match='Angular step'):run_prewarm(s,f,settings=settings(),initial_rigid_angular_velocity={'ornament':[100,0,0]})
    np.testing.assert_array_equal(f.positions(),before)


def test_diagnostic_does_not_claim_to_classify_every_conflict():
    s,f=fixture();report=diagnose_initial_state(s,f)
    assert not report['all_conflicts_classified'] and not report['fixed_conflicts_found']
    assert not prewarm_status()['production_ready'] and not prewarm_status()['native_model_validated']


def test_held_pose_stability_does_not_unlock_production_or_replace_before_gate():
    from motionbridge.secondary_contact_policy import require_contact_policy
    s,f=fixture();report=run_prewarm(s,f,settings=settings(),acceleration=[0,0,0]);assert_pass(report)
    assert not report['self_collision_checked'] and not report['continuous_collision_checked']
    with pytest.raises(ValueError):require_contact_policy({'secondary_contact_rules':{'simulation_ready':True}})


def test_one_quiet_final_sample_does_not_erase_earlier_window_motion():
    s,f=fixture(floor=-10.);v=np.zeros_like(f.x);v[:3,1]=.2
    report=run_prewarm(s,f,settings=settings(damping_per_second=20.),acceleration=[0,0,0],
        initial_vertex_velocity=v,initial_rigid_velocity={'ornament':[0,.2,0]})
    assert_pass(report)
    assert report['history'][14]['maximum_vertex_speed_m_s']<.02
    assert report['steps_completed']>15


def test_balance_probe_restores_candidate_exactly():
    from motionbridge.secondary_prewarm import _balance_probe,_moving_vertices
    s,f=fixture(floor=-10.);before=f.positions().copy();rigid=copy.deepcopy(f.rigid)
    report=_balance_probe(s,f,np.array([0,0,-9.81]),settings(),_moving_vertices(f))
    assert not report['passed']
    np.testing.assert_array_equal(f.positions(),before)
    np.testing.assert_array_equal(f.rigid['ornament']['rotation'],rigid['ornament']['rotation'])


def test_initial_attachment_with_no_free_dofs_is_diagnosed():
    s,_=fixture();f=s.begin_frame(s.attachment.rest,np.zeros(len(s.attachment.rest)),
        rigid_inverse_mass={'ornament':0.},rigid_inverse_inertia={'ornament':np.zeros((3,3))})
    f.x[:3]+=[.01,0,0]
    report=diagnose_initial_state(s,f)
    assert any(c['kind']=='fixed_attachment' for c in report['definite_fixed_conflicts'])


def test_angular_speed_measurement_uses_rigid_point_velocity_not_center_only():
    from motionbridge.secondary_prewarm import _speeds,_moving_vertices
    s,f=fixture();ids=_moving_vertices(f)
    speeds=_speeds(f,f.positions(),np.zeros_like(f.x),{'ornament':np.zeros(3)},
                   {'ornament':np.array([0,0,2.])},1/30,ids)
    assert speeds.max()>.2


def test_input_velocity_arrays_are_not_modified():
    s,f=fixture();v=np.zeros_like(f.x);linear={'ornament':np.zeros(3)};angular={'ornament':np.zeros(3)}
    run_prewarm(s,f,settings=settings(),acceleration=[0,0,0],initial_vertex_velocity=v,
        initial_rigid_velocity=linear,initial_rigid_angular_velocity=angular)
    np.testing.assert_array_equal(v,0.);np.testing.assert_array_equal(linear['ornament'],0.)
    np.testing.assert_array_equal(angular['ornament'],0.)


def test_compile_metadata_keeps_native_gate_closed():
    from attachment_fixture import make_fixture
    from motionbridge.secondary_contact_policy import compile_contact_policy,POLICY
    p,r,b=make_fixture();result=compile_contact_policy(p,{'surface_segmentation':{'review':r},
        'secondary_contact_rules':{'schema_version':1,'policy':POLICY,'bindings':b}})
    assert result['prewarm']['core_implemented'] and not result['prewarm']['native_model_validated']
    assert not result['simulation_ready'] and not result['solver_implemented']


@pytest.mark.parametrize('failure',['fixed','balance','projection'])
def test_existing_prewarm_stop_gate_rejects_every_new_failure(failure,monkeypatch):
    from motionbridge.secondary_guard import prewarm_stop_gate
    import motionbridge.secondary_prewarm as module
    s,f=fixture(floor=.03 if failure=='fixed' else -.2)
    if failure=='balance':
        monkeypatch.setattr(module,'_balance_probe',lambda *args:{'passed':False})
    if failure=='projection':
        monkeypatch.setattr(s,'solve',lambda *args,**kwargs:{'discrete_constraints_converged':False})
    report=run_prewarm(s,f,settings=settings(),acceleration=[0,0,0])
    assert report['rolled_back'] and not report['prewarm_convergence']['passed']
    gate=prewarm_stop_gate(report,{'prewarm_p95_limit_m_s':.02,'prewarm_max_limit_m_s':.1})
    assert not gate['accepted'] and gate['stop_before']=='hair_and_surface_assembly'
