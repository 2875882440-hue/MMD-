import copy,json
from pathlib import Path
import numpy as np
import pytest
from motionbridge.secondary_hair import (HairSolver,select_hair,validate_settings,proxy_contacts,
                                         euler_matrix,matrix_euler,limit_rotation,swing)

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def settings():
    return json.loads((ROOT/'profiles/secondary_motion/rin.json').read_text(encoding='utf-8'))['hair_dynamics']


def node(parent=-1):
    return dict(parent=parent,external_parent='head',local_head=[0,0,0] if parent<0 else [0,.2,0],
                local_rotation=np.eye(3).tolist(),joint_frame_parent=np.eye(3).tolist(),length=.2,
                radius=.004,angular_min=[-.5,-.5,-.5],angular_max=[.5,.5,.5])


@pytest.mark.parametrize('key,value',[('substeps',True),('substeps',0),('collision_iterations',0),
    ('warmup_seconds',0),('drag_per_second',float('nan')),('rest_acceleration',-1),
    ('collision_margin_m',float('inf')),('strand_radius_scale',0),('gravity_m_s2',[0,0,-100]),
    ('inertia_strength',-1),('inertia_strength',float('nan'))])
def test_invalid_physics_settings_fail(settings,key,value):
    settings[key]=value
    with pytest.raises(ValueError):validate_settings(settings)


def test_no_force_stationary_driver_is_exactly_unchanged(settings):
    solver=HairSolver([node()],settings)
    for _ in range(120):
        heads,rots,tails,angles=solver.step({'head':np.eye(4)},None,1/180,force_enabled=False)
    assert tails[0]==pytest.approx([0,.2,0],abs=1e-12)
    assert rots[0]==pytest.approx(np.eye(3),abs=1e-12)


def test_gravity_changes_orientation_with_fixed_body_and_length(settings):
    solver=HairSolver([node()],settings)
    driver=np.eye(4);before=driver.copy()
    for _ in range(360):heads,rots,tails,angles=solver.step({'head':driver},None,1/180)
    assert tails[0,2]<-.01
    assert np.linalg.norm(tails[0]-heads[0])==pytest.approx(.2,abs=1e-12)
    assert np.max(np.abs(angles))<=.50000001
    assert np.array_equal(driver,before)
    assert np.linalg.norm(solver.velocities)<.005


def test_moving_parent_produces_inertia_not_a_static_pose_offset(settings):
    settings['gravity_m_s2']=[0,0,0]
    solver=HairSolver([node()],settings);driver=np.eye(4)
    for _ in range(60):solver.step({'head':driver},None,1/180)
    driver[0,3]=.08
    h,r,t,a=solver.step({'head':driver},None,1/180)
    assert t[0,0]<h[0,0]-.0001
    for _ in range(360):h,r,t,a=solver.step({'head':driver},None,1/180)
    assert t[0,0]==pytest.approx(h[0,0],abs=.0001)


def test_parent_chain_remains_connected_and_preserves_both_lengths(settings):
    solver=HairSolver([node(),node(0)],settings)
    for _ in range(200):h,r,t,a=solver.step({'head':np.eye(4)},None,1/180)
    assert h[1]==pytest.approx(t[0],abs=1e-10)
    assert np.linalg.norm(t-h,axis=1)==pytest.approx([.2,.2],abs=1e-10)


@pytest.mark.parametrize('angles',[[.2,.3,-.4],[-.5,.1,.6],[0,0,0]])
def test_angular_limit_is_checked_in_the_joint_frame(angles):
    frame=euler_matrix([.4,-.2,.1]); reference=euler_matrix([-.1,.3,.2])
    candidate=frame@euler_matrix(angles)@frame.T@reference
    limited,actual=limit_rotation(candidate,reference,frame,[-.1]*3,[.1]*3)
    assert np.max(np.abs(actual))<=.1000001
    independently_measured=matrix_euler(frame.T@limited@reference.T@frame)
    assert independently_measured==pytest.approx(np.clip(angles,-.1,.1),abs=1e-10)


@pytest.mark.parametrize('kind,size,point,expected',[(0,[1,0,0],[2,0,0],1),
    (2,[1,2,0],[0,0,3],1),(1,[1,2,3],[0,0,0],-1),(1,[1,2,3],[2,2,3],1)])
def test_analytic_body_proxy_distances_and_normals(kind,size,point,expected):
    distance,normal=proxy_contacts(np.array([point],float),np.zeros((1,3)),np.eye(3)[None],
                                   np.array([kind]),np.array([size],float))
    assert distance[0,0]==pytest.approx(expected)
    assert np.linalg.norm(normal[0,0])==pytest.approx(1)


def test_collision_force_moves_free_tip_out_of_a_body_sphere(settings):
    settings['gravity_m_s2']=[0,0,0]
    s=HairSolver([node()],settings)
    body=dict(centers=np.array([[.015,.19,0]]),rotations=np.eye(3)[None],kinds=np.array([0]),sizes=np.array([[.025,0,0]]))
    for _ in range(120):h,r,t,a=s.step({'head':np.eye(4)},body,1/180)
    assert t[0,0]<-.005 and s.contacts_applied>0
    assert np.linalg.norm(t[0]-h[0])==pytest.approx(.2,abs=1e-10)


def test_native_selection_never_includes_protected_or_accessory_bones(settings):
    from motionbridge.secondary_plan import compile_profile
    profile=json.loads((ROOT/'profiles/secondary_motion/rin.json').read_text(encoding='utf-8'))
    plan=compile_profile(profile,ROOT)
    ids=select_hair(plan,settings)
    assert len(ids)==84
    assert set(ids).isdisjoint(plan['protected_bone_ids'])
    assert all('飾' not in plan['bones'][i]['name'] for i in ids)
    settings['include_bone_tokens']=['上半身']
    with pytest.raises(ValueError):select_hair(plan,settings)


def test_invalid_chain_and_time_step_are_rejected(settings):
    with pytest.raises(ValueError):HairSolver([node(0)],settings)
    solver=HairSolver([node()],settings)
    with pytest.raises(ValueError):solver.step({'head':np.eye(4)},None,.5)


def test_long_float32_chains_do_not_amplify_rotation_drift(settings):
    nodes=[node()]+[node(i-1) for i in range(1,12)]
    for n in nodes:n['local_rotation']=(np.eye(3)*1.0000001).tolist()
    s=HairSolver(nodes,settings)
    for _ in range(20):h,r,t,a=s.step({'head':np.eye(4)},None,1/180)
    for rotation in r:
        assert rotation.T@rotation==pytest.approx(np.eye(3),abs=1e-12)
    assert s.maximum_length_error<1e-12


def test_long_chain_warmup_converges_without_endpoint_velocity_explosion(settings):
    s=HairSolver([node()]+[node(i-1) for i in range(1,10)],settings)
    for _ in range(540):s.step({'head':np.eye(4)},None,1/180)
    assert np.max(np.linalg.norm(s.velocities,axis=1))<.005
    assert np.max(np.linalg.norm(s.angular_velocities,axis=1))<.005


@pytest.mark.parametrize('bad',['not_enabled','outside_source','outside_output','existing_output','profile_traversal'])
def test_hair_cli_requires_opt_in_and_cannot_write_old_projects(tmp_path,monkeypatch,bad):
    import importlib.util
    spec=importlib.util.spec_from_file_location('hair_cli',ROOT/'scripts/simulate_secondary_hair.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    local=tmp_path/'development';local.mkdir();monkeypatch.setattr(module,'ROOT',local)
    scene=local/'model.blend';scene.write_bytes(b'fake scene for argument validation only')
    blender=local/'blender.exe';blender.write_bytes(b'not executed')
    output=local/'output/secondary_motion/new';enabled=True;profile='rin'
    if bad=='not_enabled':enabled=False
    if bad=='outside_source':scene=tmp_path/'original.blend';scene.write_bytes(b'unchanged')
    if bad=='outside_output':output=tmp_path/'original/output'
    if bad=='existing_output':output.mkdir(parents=True)
    if bad=='profile_traversal':profile='../rin'
    with pytest.raises(ValueError):module.build_command(profile,scene,output,enabled,blender)


def test_contact_and_force_cannot_exceed_the_combined_joint_speed_guard(settings):
    solver=HairSolver([node()],settings);driver=np.eye(4);last=np.zeros(3);dt=1/180
    body=dict(centers=np.array([[.015,.19,0]]),rotations=np.eye(3)[None],kinds=np.array([0]),sizes=np.array([[.025,0,0]]))
    for index in range(80):
        driver[0,3]=.02*np.sin(index*.3)
        h,r,t,a=solver.step({'head':driver},body,dt)
        assert np.linalg.norm(a[0]-last)<=(settings['max_joint_speed_rad_s']+settings['contact_joint_speed_rad_s'])*dt+1e-10
        last=a[0].copy()


def test_static_body_contact_reaction_converges_without_chatter(settings):
    settings['gravity_m_s2']=[0,0,0]
    solver=HairSolver([node()],settings)
    body=dict(centers=np.array([[.015,.19,0]]),rotations=np.eye(3)[None],kinds=np.array([0]),sizes=np.array([[.025,0,0]]))
    for _ in range(600):solver.step({'head':np.eye(4)},body,1/180)
    assert np.linalg.norm(solver.velocities)<.005
