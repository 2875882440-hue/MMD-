"""Bounded arm/head proxy avoidance with a previous-torso-result safeguard."""
import copy
import numpy as np
from .body_collision import avoid_arm_torso,arm_clearance,torso_space,_unit,_smooth


def clearance_report(values):
    values=np.asarray(values,float)
    return {'penetrating_frames':int((values.min(axis=1)<-.001).sum()),
            'min_proxy_clearance_m':float(values.min()),
            'total_proxy_deficit_m':float(np.maximum(-values,0).sum())}


def avoid_arm_head(points,center,basis,radii,radius,fps,torso,candidate=None):
    """Head ellipsoid / arm samples, capped to 20 degrees per arm segment.

    Backtrack correction strength if it worsens the old torso's minimum or
    total sample deficit, or increases the head deficit. Smooth the backoff
    weights downwards so rejected corrections do not leak into neighbours.
    Unresolvable poses stay close to the original and are reported, not hidden.
    """
    points=np.asarray(points,float)
    if candidate is None:
        candidate,_=avoid_arm_torso(points,center,basis,radii,radius,fps,max_angle_deg=20.)
    before=arm_clearance(points,center,basis,radii,radius)
    torso_before=arm_clearance(points,*torso)
    lengths=np.linalg.norm(np.diff(points,axis=1),axis=-1,keepdims=True)
    old=_unit(np.diff(points,axis=1));delta=_unit(np.diff(candidate,axis=1))-old
    weight=np.ones(len(points))
    def build():
        segments=_unit(old+delta*weight[:,None,None])*lengths
        out=points.copy();out[:,1]=out[:,0]+segments[:,0];out[:,2]=out[:,1]+segments[:,1]
        unchanged=np.max(abs(delta),axis=(1,2))<1e-12
        out[unchanged|(weight==0)]=points[unchanged|(weight==0)]
        return out
    def bad_rows(out):
        body=arm_clearance(out,*torso);head=arm_clearance(out,center,basis,radii,radius)
        return ((body.min(1)<np.minimum(torso_before.min(1),0)-1e-6)
            |(np.maximum(-body,0).sum(1)>np.maximum(-torso_before,0).sum(1)+1e-6)
            |(np.maximum(-head,0).sum(1)>np.maximum(-before,0).sum(1)+1e-6))
    for _ in range(18):
        out=build();bad=bad_rows(out)
        if not bad.any():break
        weight[bad]*=.5
        weight=np.minimum(weight,_smooth(weight,fps))
    out=build();bad=bad_rows(out)
    weight[bad]=0.;out=build()
    after=arm_clearance(out,center,basis,radii,radius)
    body_after=arm_clearance(out,*torso)
    angles=np.rad2deg(np.arccos(np.clip(np.sum(old*_unit(np.diff(out,axis=1)),axis=-1),-1,1)))
    displacement=np.linalg.norm(out-points,axis=-1)
    return out,{'before':clearance_report(before),'after':clearance_report(after),
        'torso_before':clearance_report(torso_before),'torso_after':clearance_report(body_after),
        'torso_safeguard_passed':bool(not bad_rows(out).any()),
        'backoff_frames':int((weight<.999).sum()),'rejected_frames':int((weight==0).sum()),
        'max_joint_adjustment_m':float(displacement.max()),'segment_adjustment_limit_deg':20.,
        'max_segment_adjustment_deg':float(angles.max()),'corrected_frames':int((displacement.max(1)>1e-6).sum())}


def source_head_space(poses,world):
    # Exactly the head ellipsoid used by the built-in Blender mannequin, Y-up.
    return poses[:,15]+world[:,15]@np.array([0.,.055,0.]),world[:,15],np.array([.095,.12,.085])


def constrain_head_collision(motion):
    from .exporters import solve_rotations,_swing
    from .stability import fk_positions
    from .global_motion import SMPL22_NAMES,SMPL22_PARENTS
    result=copy.deepcopy(motion)
    if result.get('head_collision',{}).get('enabled'):return result
    if motion.get('root_mode')!='estimated_world_translation' or motion['joint_names']!=SMPL22_NAMES or motion['parents']!=SMPL22_PARENTS:
        raise ValueError('Head collision requires SMPL22 world motion')
    _,world,poses=solve_rotations(motion);world=world.copy();rest=np.asarray(motion['rest_positions'])
    torso=torso_space(poses[:,0],poses[:,16],poses[:,17],rest[0],rest[16],rest[17])
    center,basis,radii=source_head_space(poses,world)
    radius=float(np.linalg.norm(rest[16]-rest[17])*.12)
    reports={}
    for label,indices in [('left',[16,18,20]),('right',[17,19,21])]:
        solved,reports[label]=avoid_arm_head(poses[:,indices],center,basis,radii,radius,motion['fps'],torso)
        for f in range(len(poses)):
            for a,b,k in [(indices[0],indices[1],0),(indices[1],indices[2],1)]:
                delta=_swing(poses[f,b]-poses[f,a],solved[f,k+1]-solved[f,k]);world[f,a]=delta@world[f,a]
                if k==1:world[f,indices[2]]=delta@world[f,indices[2]]
    final=fk_positions(world,motion['parents'],rest,poses[:,0]);result['world_rotations']=world.tolist()
    for f,p in zip(result['frames'],final):f['positions']=p.tolist()
    result['head_collision']={'enabled':True,'method':'bounded_head_proxy_torso_guard_v1','radii_m':radii.tolist(),
        'arm_radius_m':radius,'sides':reports,'root_head_and_legs_unchanged':True,
        'note':'Mannequin head ellipsoid plus arm samples; target recalibrates its face proxy. Not mesh/cloth collision or capture accuracy. Touching gestures may move.'}
    result.setdefault('warnings',[]).append('头脸避碰采用头部几何代理和20度局部修正上限，可能改变摸脸手势；不含发饰、头发、衣裙或真实手指碰撞。')
    solve_rotations(result)
    return result
