import json
from pathlib import Path
import numpy as np
import pytest
from motionbridge.secondary_coupling import CoupledHairSolver,native_joint_residual
from motionbridge.secondary_hair import euler_matrix
from motionbridge.secondary_surface import anchor_unattached_components,weld_driver_surface
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def settings():
    p=json.loads((ROOT/'profiles/secondary_motion/rin.json').read_text(encoding='utf-8'))['hair_dynamics']
    p['gravity_m_s2']=[0,0,0];p['rest_acceleration']=0
    return p

def node(head,rotation=None,external='body'):
    return dict(parent=-1,external_parent=external,local_head=head,local_rotation=(np.eye(3) if rotation is None else rotation).tolist(),
                joint_frame_parent=np.eye(3).tolist(),length=.2,radius=.008,angular_min=[-.8]*3,angular_max=[.8]*3)

def joint(a=0,b=1):
    return dict(a=a,b=b,anchor_a=[.1,.1,0],anchor_b=[-.1,.1,0],frame_a=np.eye(3).tolist(),frame_b=np.eye(3).tolist(),
                linear_min=[0,0,0],linear_max=[0,0,0],angular_min=[0,0,0],angular_max=[0,0,0])

def test_native_sliding_joint_does_not_get_incorrectly_welded_to_zero():
    heads=np.array([[0,0,1],[.4,.1,1.05]]);rotations=np.tile(np.eye(3),(2,1,1));j=joint()
    j.update(anchor_a=[0,0,0],anchor_b=[0,0,0],linear_min=[0,-.8,-.8],linear_max=[.8,.8,.8])
    linear,angular,*_=native_joint_residual(heads,rotations,j)
    assert linear==pytest.approx([0,0,0]);assert angular==pytest.approx([0,0,0])

def test_cross_joint_reduces_real_anchor_and_rotation_error_without_body_feedback(settings):
    nodes=[node([-.1,0,1]),node([.1,0,1])];h=np.array([[-.1,0,1],[.1,0,1]])
    solver=CoupledHairSolver(nodes,settings,[joint()],h,h+[0,.2,0]);solver.base.angles[1,2]=.25
    driver=np.eye(4);before=driver.copy();initial=solver.residuals({'body':driver})
    for _ in range(50):solver.step({'body':driver},None,1/180)
    final=solver.residuals({'body':driver})
    assert final['cross_linear_violation_max_m']<initial['cross_linear_violation_max_m']
    assert final['cross_angular_violation_max_rad']<initial['cross_angular_violation_max_rad']
    assert np.array_equal(driver,before)

def test_capsule_self_collision_separates_crossing_strands(settings):
    nodes=[node([0,-.1,1]),node([-.1,0,1],euler_matrix([0,0,-np.pi/2]))]
    rest_h=np.array([[0,0,2],[1,0,2]]);rest_t=rest_h+[0,.2,0]
    solver=CoupledHairSolver(nodes,settings,[],rest_h,rest_t)
    initial=solver.residuals({'body':np.eye(4)})
    for _ in range(90):solver.step({'body':np.eye(4)},None,1/180)
    final=solver.residuals({'body':np.eye(4)})
    assert solver.self_contacts>0
    assert final['self_capsule_depth_max_m']<initial['self_capsule_depth_max_m']*.5

def test_authored_self_overlap_exclusions_are_explicit(settings):
    nodes=[node([0,0,1]),node([0,0,1])];h=np.array([[0,0,1],[0,0,1.]])
    solver=CoupledHairSolver(nodes,settings,[],h,h+[0,.2,0])
    assert solver.authored_overlap_exclusions==[[0,1]]
    assert len(solver.pairs)==0

def test_floor_reaction_bends_a_strand_without_moving_its_root(settings):
    n=node([0,0,.05],euler_matrix([-np.pi/2,0,0]));n['angular_min']=[-1.5]*3;n['angular_max']=[1.5]*3
    h=np.array([[0,0,.05]]);t=h+[0,0,-.2];solver=CoupledHairSolver([n],settings,[],h,t)
    initial=solver.residuals({'body':np.eye(4)})['floor_depth_max_m']
    for _ in range(350):heads,rots,tails,_=solver.step({'body':np.eye(4)},None,1/180)
    assert heads[0]==pytest.approx([0,0,.05])
    assert solver.floor_contacts>0
    assert solver.residuals({'body':np.eye(4)})['floor_depth_max_m']<initial*.1

def test_model_neutral_anchor_rule_does_not_depend_on_video_pose():
    rest=np.array([[0,0,0],[1,0,0],[1,0,1],[0,0,1.]])
    pins,report=anchor_unattached_components(rest,[[0,1,2],[0,2,3]],[0]*4)
    assert np.sum(pins==1)>=3 and len(report)==1
    assert pins[2]==pins[3]==1
    unchanged,_=anchor_unattached_components(rest,[[0,1,2],[0,2,3]],pins)
    assert np.array_equal(pins,unchanged)

def test_weld_keeps_authored_triangles_and_does_not_join_different_skin():
    rest=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,0.]])
    reps,faces,pins,inverse=weld_driver_surface(rest,[[0,1,2],[3,1,2]],[0,0,0,1],['same']*4)
    assert len(reps)==3 and len(faces)==1 and pins[inverse[0]]==1
    reps,_,_,_=weld_driver_surface(rest,[[0,1,2],[3,1,2]],[0]*4,['a','x','y','b'])
    assert len(reps)==4


def test_fixed_cross_joint_forms_rigid_children_but_parent_can_move(settings):
    root=node([0,0,1]);a=node([-.1,.1,0]);b=node([.1,.1,0]);a['parent']=b['parent']=0
    j=joint(1,2);h=np.array([[0,0,1],[-.1,.1,1],[.1,.1,1]])
    solver=CoupledHairSolver([root,a,b],settings,[j],h,h+[0,.2,0])
    solver.base.angles[0,0]=.2
    solver.step({'body':np.eye(4)},None,1/180)
    assert set(solver.weld_locked_bones)=={1,2}
    assert solver.base.angles[0,0]==pytest.approx(.2)
    assert solver.base.angles[1:]==pytest.approx(np.zeros((2,3)))
    assert solver.residuals({'body':np.eye(4)})['cross_linear_violation_max_m']<1e-10


def test_floor_solver_respects_model_ground_height(settings):
    n=node([0,0,1.05],euler_matrix([-np.pi/2,0,0]));n['angular_min']=[-1.5]*3;n['angular_max']=[1.5]*3
    h=np.array([[0,0,1.05]]);t=h+[0,0,-.2]
    ground=CoupledHairSolver([n],settings,[],h,t,floor_z=1.)
    no_ground=CoupledHairSolver([n],settings,[],h,t,floor_z=0.)
    assert ground.residuals({'body':np.eye(4)})['floor_depth_max_m']>.1
    assert no_ground.residuals({'body':np.eye(4)})['floor_depth_max_m']==0
    for _ in range(350):ground.step({'body':np.eye(4)},None,1/180)
    assert ground.floor_contacts>0
    assert ground.residuals({'body':np.eye(4)})['floor_depth_max_m']<.015
