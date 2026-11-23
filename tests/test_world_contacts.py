import copy
import numpy as np
from motionbridge.global_motion import SMPL22_NAMES,SMPL22_PARENTS
from motionbridge.exporters import default_rest_positions,JOINT_NAMES,solve_rotations
from motionbridge.stability import fk_positions
from motionbridge.world_contacts import refine_world_contacts,ground_evidence


def fixture_motion(jump=False,travel=False):
    n=150;fps=30
    standard=dict(zip(JOINT_NAMES,default_rest_positions()))
    standard['Spine2']=(standard['Spine']+standard['Chest'])/2
    rest=np.array([standard[name] for name in SMPL22_NAMES])
    root=np.zeros((n,3));root[:,1]=1+np.linspace(0,.15,n)
    if travel:root[:,0]=np.linspace(0,.4,n)
    height=np.zeros(n)
    if jump:height[60:80]=.3*np.sin(np.linspace(0,np.pi,20));root[:,1]+=height
    rotations=np.tile(np.eye(3),(n,22,1,1))
    positions=fk_positions(rotations,SMPL22_PARENTS,rest,root)
    motion={'fps':fps,'num_frames':n,'joint_names':SMPL22_NAMES,'parents':SMPL22_PARENTS,
            'rest_positions':rest.tolist(),'world_rotations':rotations.tolist(),
            'root_mode':'estimated_world_translation','root_motion':True,'positions_are_solved':True,
            'frames':[{'positions':p.tolist(),'time':i/fps,'confidence':[0]*22} for i,p in enumerate(positions)],
            'settings':{},'warnings':[]}
    kp=np.zeros((n,17,3));kp[:,:,0]=320;kp[:,:,1]=300;kp[:,:,2]=.95
    kp[:,0,1]=100-height*400;kp[:,[15,16],1]=700-height[:,None]*400
    kp[:,15,0]=290;kp[:,16,0]=350
    return motion,kp


def test_stance_removes_slow_drift_without_resetting_horizontal_root():
    motion,kp=fixture_motion(travel=True);before=copy.deepcopy(motion)
    result=refine_world_contacts(motion,kp)
    _,wr,p=solve_rotations(result);_,raw,old=solve_rotations(motion)
    assert np.array_equal(p[:,0][:,[0,2]],old[:,0][:,[0,2]])
    assert np.array_equal(wr[:,0],raw[:,0])
    assert p[-1,0,1]-p[0,0,1]<.025
    assert result['world_contact_refinement']['max_height_correction_m']>.10
    assert motion==before


def test_airborne_interval_is_released_and_jump_height_survives():
    motion,kp=fixture_motion(jump=True)
    result=refine_world_contacts(motion,kp)
    weights=np.array(result['foot_contacts']['weights'])
    assert not weights[64:76].any()
    p=np.array([f['positions'] for f in result['frames']])[:,0,1]
    assert p[60:80].max()-np.median(p[40:55])>.27


def test_unreliable_feet_do_not_invent_ground_or_correction():
    motion,kp=fixture_motion();kp[:,[15,16],2]=.1
    result=refine_world_contacts(motion,kp)
    assert result['world_contact_refinement']['floor_toe_proxy_m'] is None
    assert result['world_contact_refinement']['max_height_correction_m']==0
    assert np.allclose([f['positions'] for f in result['frames']], [f['positions'] for f in motion['frames']])


def test_contact_refinement_is_idempotent():
    motion,kp=fixture_motion();once=refine_world_contacts(motion,kp)
    assert refine_world_contacts(once,kp)==once


def test_ground_height_evidence_allows_travel_but_releases_airborne_feet():
    _,kp=fixture_motion(jump=True)
    kp[:,[15,16],0]+=np.arange(len(kp))[:,None]*10
    weights=ground_evidence(kp,30)
    assert weights[10:40].min()==1
    assert not weights[64:76].any()


def test_target_sole_baseline_preserves_jump_and_unknown_ground():
    from motionbridge.world_retarget import target_ground_offsets
    n=150;drift=np.linspace(0,.04,n);jump=np.zeros(n);jump[60:80]=.3*np.sin(np.linspace(0,np.pi,20))
    soles=np.tile((drift+jump)[:,None],(1,2));weights=np.ones((n,2));weights[58:82]=0
    offsets=target_ground_offsets(soles,weights,30,.65)
    grounded=soles[:,0]+offsets
    assert abs(grounded[-1]-.003)<.005
    assert grounded[60:80].max()-np.median(grounded[40:55])>.29
    assert abs(offsets[70])<1e-5
    assert np.array_equal(target_ground_offsets(soles,weights*0,30,.65),np.zeros(n))
