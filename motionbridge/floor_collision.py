"""Unilateral flat-ground constraints. No timestamps or clip-specific poses.

Source feet use the built-in mannequin's conservative ellipsoid/sphere bounds.
Targets can use evaluated skin geometry with the same positive lift envelope.
"""
import copy
import numpy as np


def lift_envelope(required, fps, max_speed=.35):
    """Smallest nonnegative Lipschitz majorant; never smooth below collision."""
    values=np.asarray(required,float)
    if values.ndim!=1 or not np.isfinite(values).all() or not np.isfinite([fps,max_speed]).all() or fps<=0 or max_speed<=0:
        raise ValueError('Invalid floor correction track')
    result=np.maximum(values,0).copy();step=max_speed/fps
    for i in range(1,len(result)):result[i]=max(result[i],result[i-1]-step)
    for i in range(len(result)-2,-1,-1):result[i]=max(result[i],result[i+1]-step)
    return result


def foot_proxy_heights(poses, world, rest):
    """Y-up minima of foot/toe joint spheres and mannequin ellipsoids."""
    p=np.asarray(poses);w=np.asarray(world);rest=np.asarray(rest);result=[]
    for foot,toe in [(7,10),(8,11)]:
        axis=rest[toe]-rest[foot];length=np.linalg.norm(axis);direction=axis/max(length,1e-12)
        # Ellipsoid axial half-length and transverse radii, matching exporter.
        vertical=(w[:,foot]@direction)[:,1]
        extent=np.sqrt(.038**2*(1-vertical**2)+max(.04,length*.52)**2*vertical**2)
        segment=(p[:,foot,1]+p[:,toe,1])*.5-extent
        center=p[:,toe]+w[:,toe]@np.array([0.,.01,.025])
        toe_extent=np.linalg.norm(w[:,toe,1,:]*np.array([.055,.045,.11]),axis=1)
        result.append(np.minimum.reduce([p[:,foot,1]-.045,p[:,toe,1]-.045,segment,center[:,1]-toe_extent]))
    return np.stack(result,axis=1)


def constrain_world_floor(motion, floor_y=0., clearance=.001):
    from .exporters import solve_rotations,_swing
    from .stability import fk_positions
    from .refinement import fit_two_bone
    result=copy.deepcopy(motion)
    if result.get('floor_collision',{}).get('enabled'):return result
    if motion.get('root_mode')!='estimated_world_translation' or len(motion['parents'])!=22:
        raise ValueError('Floor collision requires SMPL22 world motion')
    if not np.isfinite([floor_y,clearance]).all() or clearance<0:raise ValueError('Invalid floor')
    _,world,poses=solve_rotations(motion);world=world.copy();rest=np.asarray(motion['rest_positions'])
    roots=poses[:,0].copy();parents=motion['parents'];before=foot_proxy_heights(poses,world,rest)
    threshold=floor_y+clearance;corrections=np.maximum(threshold-before,0.)
    for side,(hip,knee,foot) in enumerate([(1,4,7),(2,5,8)]):
        la=np.linalg.norm(rest[knee]-rest[hip]);lb=np.linalg.norm(rest[foot]-rest[knee])
        for f in np.flatnonzero(corrections[:,side]>1e-8):
            goal=poses[f,foot].copy();goal[1]+=corrections[f,side]
            middle,end=fit_two_bone(poses[f,hip],goal,poses[f,knee],la,lb,poses[f,knee])
            world[f,hip]=_swing(world[f,hip]@(rest[knee]-rest[hip]),middle-poses[f,hip])@world[f,hip]
            world[f,knee]=_swing(world[f,knee]@(rest[foot]-rest[knee]),end-middle)@world[f,knee]
    solved=fk_positions(world,parents,rest,roots)
    after_ik=foot_proxy_heights(solved,world,rest)
    lift=lift_envelope(np.maximum(threshold-after_ik.min(axis=1),0),motion['fps'])
    roots[:,1]+=lift;final=fk_positions(world,parents,rest,roots)
    after=foot_proxy_heights(final,world,rest)
    if np.min(after)<threshold-1e-6:raise RuntimeError('Floor projection failed')
    result['world_rotations']=world.tolist()
    for frame,points in zip(result['frames'],final):frame['positions']=points.tolist()
    result['floor_collision']={'enabled':True,'method':'unilateral_foot_proxy_ik_v1','floor_y_m':float(floor_y),
        'clearance_m':clearance,'proxy':'built-in mannequin foot/toe spheres and ellipsoids; not physical shoe truth',
        'corrected_foot_samples':int((corrections>1e-8).sum()),'min_before_m':float(before.min()),
        'min_after_m':float(after.min()),'root_lift_m':lift.tolist(),'max_root_lift_m':float(lift.max()),
        'root_horizontal_unchanged':True,'root_rotation_unchanged':True,
        'note':'Flat Y=0 model reference is assumed. Only upward collision correction; no airborne down-clamp. Not terrain or full-body physics.'}
    if 'global_motion' in result:result['global_motion']['root_translation_range_m']=np.ptp(roots,axis=0).tolist()
    result.setdefault('warnings',[]).append('脚底防穿按水平地面与脚部几何代理求解，必要时向上补偿根高度；不支持台阶、斜坡或全身碰撞。')
    solve_rotations(result)
    return result
