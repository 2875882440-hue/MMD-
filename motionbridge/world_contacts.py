"""Conservative contact hypotheses for GVHMR world motion, without root XY reset.

Slow vertical drift is estimated only from image-supported stance intervals.
No contact is forced in every frame. Ground, scale and confidence remain estimates.
"""
import copy
import numpy as np
from scipy.ndimage import gaussian_filter1d, percentile_filter


def ground_evidence(keypoints, fps):
    """Ground-height evidence allows horizontal foot travel; it never locks XY.

    Distinct from stationary support: walking/turning can touch ground without
    satisfying a foot-lock velocity threshold. Both are still hypotheses.
    """
    kp=np.asarray(keypoints,float);n=len(kp);weights=np.zeros((n,2))
    if n<3:return weights
    height=np.maximum(np.ptp(kp[:,:,1],axis=1),1.)
    height=percentile_filter(height,50,size=max(3,round(fps*.5))|1,mode='nearest')
    feet=kp[:,[15,16],1];lower=feet.max(axis=1)
    floor=percentile_filter(lower,85,size=max(3,round(fps*1.2))|1,mode='nearest')
    vertical=np.abs(np.gradient(feet,axis=0))*fps/height[:,None]
    ok=(kp[:,[15,16],2]>=.7)&(feet>=floor[:,None]-.035*height[:,None])
    ok&=(feet>=lower[:,None]-.05*height[:,None])&(vertical<.28)
    ramp=max(2,round(fps*.08));minimum=max(3,round(fps*.1))
    for s in range(2):
        edges=np.diff(np.r_[False,ok[:,s],False].astype(int))
        for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
            if b-a>=minimum:
                k=np.arange(b-a);weights[a:b,s]=np.minimum(1.,np.minimum((k+1)/ramp,(b-a-k)/ramp))
    return weights


def contact_candidates(poses, keypoints, fps, leg_length):
    poses=np.asarray(poses,float);kp=np.asarray(keypoints,float)
    n=len(poses);weights=np.zeros((n,2));segments=[]
    if n<max(3,round(fps*.2)): return weights,segments
    if kp.shape!=(n,17,3) or not np.isfinite(kp).all() or not np.isfinite(poses).all():
        raise ValueError('Invalid synchronized contact observations')
    # Robust body-image height; camera-space crop is not used as a ground plane.
    height=np.maximum(np.max(kp[:,:,:2][:,:,1],axis=1)-np.min(kp[:,:,:2][:,:,1],axis=1),1.)
    height=percentile_filter(height,50,size=max(3,round(fps*.5))|1,mode='nearest')
    foot_image=kp[:,[15,16],:2]
    lower=foot_image[:,:,1].max(axis=1)
    floor=percentile_filter(lower,85,size=max(3,round(fps*1.2))|1,mode='nearest')
    visible=(kp[:,[15,16],2]>=.7)
    image_speed=np.linalg.norm(np.gradient(foot_image,axis=0),axis=-1)*fps/height[:,None]
    foot=poses[:,[7,8]]
    velocity=np.gradient(gaussian_filter1d(foot,max(.5,fps*.04),axis=0),axis=0)*fps
    horizontal=np.linalg.norm(velocity[:,:,[0,2]],axis=-1)
    vertical=np.abs(velocity[:,:,1])
    near=(foot_image[:,:,1]>=floor[:,None]-.045*height[:,None])
    near&=foot_image[:,:,1]>=lower[:,None]-.065*height[:,None]
    eligible=visible&near&(image_speed<.22)&(horizontal<.35*leg_length)&(vertical<.22*leg_length)
    minimum=max(3,round(fps*.2));ramp=max(2,round(fps*.12))
    for side in range(2):
        start=None
        for f in range(n+1):
            ok=f<n and bool(eligible[f,side])
            if ok and start is not None:
                ok=np.linalg.norm(foot[f,side,[0,2]]-foot[start,side,[0,2]])<.10*leg_length
            if start is not None and not ok:
                end=f
                if end-start>=minimum:
                    k=np.arange(end-start)
                    weights[start:end,side]=np.minimum(1.,np.minimum((k+1)/ramp,(end-start-k)/ramp))
                    segments.append({'side':side,'start':start,'end':end})
                start=None
            if ok and start is None:start=f
    return weights,segments


def refine_world_contacts(motion,keypoints):
    from .exporters import validate_motion,solve_rotations,_swing
    from .stability import fk_positions
    from .refinement import fit_two_bone
    names,parents,rest,poses,fps=validate_motion(motion)
    if motion.get('root_mode')!='estimated_world_translation' or len(names)!=22:
        raise ValueError('World contact refinement requires SMPL22 world motion')
    result=copy.deepcopy(motion)
    if motion.get('world_contact_refinement',{}).get('enabled'):return result
    _,world,poses=solve_rotations(motion);world=world.copy()
    leg=float(np.mean([np.linalg.norm(rest[b]-rest[a])+np.linalg.norm(rest[c]-rest[b])
                       for a,b,c in [(1,4,7),(2,5,8)]]))
    weights,segments=contact_candidates(poses,keypoints,fps,leg)
    support=weights>=.999
    floor_tracks=poses[:,[10,11],1] # toe joint proxy, not a measured sole
    observation=np.full(len(poses),np.nan)
    for f in range(len(poses)):
        if support[f].any():observation[f]=np.median(floor_tracks[f,support[f]])
    reliable=np.flatnonzero(np.isfinite(observation))
    shift=np.zeros(len(poses));plane=None
    if len(reliable)>=max(4,round(fps*.2)):
        first=reliable[reliable<=reliable[0]+round(fps*.5)]
        plane=float(np.median(observation[first]))
        targets=np.clip(plane-observation[reliable],-.35*leg,.35*leg)
        # A slow correction field can continue through flight; it does not clamp
        # the airborne pose to the floor. The jump itself is preserved in poses.
        shift=np.interp(np.arange(len(poses)),reliable,targets)
        shift=gaussian_filter1d(shift,max(.5,fps*.15),mode='nearest')
        limit=.08*leg/fps
        for f in range(1,len(shift)):shift[f]=np.clip(shift[f],shift[f-1]-limit,shift[f-1]+limit)
    roots=poses[:,0].copy();roots[:,1]+=shift
    shifted=fk_positions(world,parents,rest,roots)
    before=[];after=[];unreachable=0;grounded_anchors=0
    for seg in segments:
        side,a,b=seg['side'],seg['start'],seg['end']
        hip,knee,ankle=[(1,4,7),(2,5,8)][side]
        core=np.flatnonzero(support[a:b,side])+a
        anchor=np.median(shifted[core,ankle],axis=0) if len(core) else np.median(shifted[a:b,ankle],axis=0)
        # A stationary but floating ankle must not become a new floating anchor.
        # Keep the existing slow root correction; adjust only a bounded stance
        # endpoint using the same toe-plane estimate and the segment's toe offset.
        if plane is not None and len(core):
            residual=plane-float(np.median(shifted[core,[10,11][side],1]))
            if abs(residual)<=.08*leg:
                anchor[1]+=residual
                grounded_anchors+=1
        la,lb=np.linalg.norm(rest[knee]-rest[hip]),np.linalg.norm(rest[ankle]-rest[knee])
        for f in range(a,b):
            delta=anchor-shifted[f,ankle]
            # Do not pull a foot across a large inferred travel error.
            if np.linalg.norm(delta)>.12*leg:
                weights[f,side]=0;continue
            goal=shifted[f,ankle]+weights[f,side]*delta
            middle,end=fit_two_bone(shifted[f,hip],goal,shifted[f,knee],la,lb,shifted[f,knee])
            if np.linalg.norm(end-goal)>.005:unreachable+=1
            world[f,hip]=_swing(world[f,hip]@(rest[knee]-rest[hip]),middle-shifted[f,hip])@world[f,hip]
            world[f,knee]=_swing(world[f,knee]@(rest[ankle]-rest[knee]),end-middle)@world[f,knee]
    final=fk_positions(world,parents,rest,roots)
    for seg in segments:
        s,a,b=seg['side'],seg['start'],seg['end'];core=np.flatnonzero(weights[a:b,s]>=.999)+a
        if len(core)>1:
            # Use the same retained pairs on both sides of the comparison.
            adjacent=np.diff(core)==1
            before.extend(np.linalg.norm(np.diff(shifted[core,[7,8][s]][:,[0,2]],axis=0),axis=-1)[adjacent])
            after.extend(np.linalg.norm(np.diff(final[core,[7,8][s]][:,[0,2]],axis=0),axis=-1)[adjacent])
    for frame,p in zip(result['frames'],final):frame['positions']=p.tolist()
    result['world_rotations']=world.tolist()
    if 'global_motion' in result:
        result['global_motion']['raw_root_translation_range_m']=result['global_motion']['root_translation_range_m']
        result['global_motion']['root_translation_range_m']=np.ptp(roots,axis=0).tolist()
    result['foot_contacts']={'enabled':True,'method':'world_image_stance_v1','weights':weights.tolist(),'segments':segments,
                             'ground_weights':ground_evidence(keypoints,fps).tolist(),
                             'note':'Contact hypotheses only; not measured confidence or true ground contact.'}
    result['world_contact_refinement']={'enabled':True,'method':'slow_stance_height_and_grounded_anchor_ik_v2',
        'grounded_anchor_segments':grounded_anchors,
        'floor_toe_proxy_m':plane,'support_frame_counts':(weights>0).sum(axis=0).tolist(),
        'height_correction_m':shift.tolist(),'max_height_correction_m':float(np.max(np.abs(shift))),
        'max_height_correction_speed_m_s':float(np.max(np.abs(np.diff(shift)))*fps) if len(shift)>1 else 0.,
        'mean_support_step_before_m':float(np.mean(before)) if before else None,
        'mean_support_step_after_m':float(np.mean(after)) if after else None,
        'unreachable_samples':unreachable,'horizontal_root_unchanged':True,
        'note':'Image-supported stance and ground are estimates. Slow vertical correction can alter weak jumps; no contact is forced during flight.'}
    result.setdefault('warnings',[]).append('世界接地按画面及脚部速度估算，只缓慢修正高度漂移并有限锁脚；小跳、移动镜头或错误观测仍可能误判。')
    solve_rotations(result)
    return result
