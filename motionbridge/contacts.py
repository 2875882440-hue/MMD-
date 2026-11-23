"""Conservative image-supported foot contact, with fixed-length endpoint IK.

This is a contact prior for mostly stationary cameras, not metric world motion.
Airborne, fast, off-screen and uncertain feet must remain unlocked.
"""
import copy
import numpy as np
from .refinement import fit_two_bone


def contact_tracks(image, confidence, fps, width, height):
    image, confidence = np.asarray(image,float), np.asarray(confidence,float)
    count = len(image)
    weights = np.zeros((count,2))
    segments = []
    aspect = np.array([width/height,1.])
    feet = image[:,[27,28],:2]*aspect
    reliable = (confidence[:,[27,28]] >= .65) & (confidence[:,[31,32]] >= .5)
    reliable &= np.isfinite(feet).all(axis=-1) & (image[:,[27,28],:2]>=0).all(axis=-1) & (image[:,[27,28],:2]<=1).all(axis=-1)
    if not reliable.any() or count < max(3,round(fps*.18)):
        return weights, segments
    # Estimate the lower image band only from observed feet. Both feet above it
    # is a possible jump, so do not force a contact to exist in every frame.
    floor_y = np.quantile(feet[:,:,1][reliable], .85)
    speed = np.linalg.norm(np.gradient(feet,axis=0),axis=-1)*fps
    near_floor = (feet[:,:,1] >= floor_y-.045) & (feet[:,:,1] >= feet[:,:,1].max(axis=1,keepdims=True)-.035)
    eligible = reliable & near_floor & (speed < .16)
    minimum = max(3,round(fps*.18))
    ramp = max(2,round(fps*.12))
    for side in range(2):
        start = None
        for f in range(count+1):
            valid = f<count and eligible[f,side]
            if valid and start is not None and np.linalg.norm(feet[f,side]-feet[start,side]) > .028:
                valid = False
            if start is not None and not valid:
                end = f
                if end-start >= minimum:
                    k=np.arange(end-start)
                    weights[start:end,side] = np.minimum(1.,np.minimum((k+1)/ramp,(end-start-k)/ramp))
                    segments.append({'side':side,'start':start,'end':end})
                start=None
            if valid and start is None:
                start=f
    return weights,segments


def lock_support_feet(motion,image,confidence,width,height):
    from .exporters import solve_rotations, _swing, validate_motion
    from .stability import fk_positions
    from scipy.ndimage import gaussian_filter1d
    names,parents,rest,poses,fps=validate_motion(motion)
    _,world,solved=solve_rotations(motion)
    world=world.copy()
    weights,segments=contact_tracks(image,confidence,fps,width,height)
    result=copy.deepcopy(motion)
    # Keep the root frame in place horizontally. Small vertical pelvis motion
    # can absorb leg-length differences instead of stretching the support leg.
    root_shift=np.zeros(len(poses))
    feet=[names.index(n) for n in ['LeftFoot','RightFoot']]
    for f in range(len(poses)):
        if weights[f].max()>.0:
            offsets=[rest[j,1]-solved[f,j,1] for s,j in enumerate(feet) if weights[f,s]>.0]
            root_shift[f]=np.clip(np.median(offsets),-.12,.12)*weights[f].max()
    root_shift=gaussian_filter1d(root_shift,max(.5,fps*.08),mode='nearest')
    roots=solved[:,0].copy()
    roots[:,1]+=root_shift
    shifted=fk_positions(world,parents,rest,roots)
    targets=shifted[:,feet].copy()
    before_slip=[]
    for seg in segments:
        s,a,b=seg['side'],seg['start'],seg['end']
        # The segment median is robust to a single poor contact observation.
        anchor=np.median(shifted[a:b,feet[s]],axis=0)
        anchor[1]=rest[feet[s],1]
        targets[a:b,s]=shifted[a:b,feet[s]]*(1-weights[a:b,s,None])+anchor*weights[a:b,s,None]
        core=np.flatnonzero(weights[a:b,s]>=.999)+a
        if len(core)>1:
            before_slip.extend(np.linalg.norm(np.diff(shifted[core,feet[s]][:,[0,2]],axis=0),axis=-1).tolist())
    for side in range(2):
        a,b,c=[names.index(('Left' if side==0 else 'Right')+name) for name in ['UpLeg','Leg','Foot']]
        la,lb=np.linalg.norm(rest[b]-rest[a]),np.linalg.norm(rest[c]-rest[b])
        for f in range(len(poses)):
            if weights[f,side]<=0:
                continue
            start=shifted[f,a]
            middle,end=fit_two_bone(start,targets[f,side],shifted[f,b],la,lb,shifted[f,b])
            world[f,a]=_swing(world[f,a]@(rest[b]-rest[a]),middle-start)@world[f,a]
            world[f,b]=_swing(world[f,b]@(rest[c]-rest[b]),end-middle)@world[f,b]
    rate_limited=0
    stability=motion.get('rotation_stabilization',{})
    if stability.get('enabled'):
        from scipy.spatial.transform import Rotation
        limit=np.deg2rad(stability['limb_speed_limit_deg_s'])/fps
        for joint in [names.index(n) for n in ['LeftUpLeg','LeftLeg','RightUpLeg','RightLeg']]:
            for f in range(1,len(poses)):
                delta=Rotation.from_matrix(world[f-1,joint].T@world[f,joint]).as_rotvec()
                angle=np.linalg.norm(delta)
                if angle>limit:
                    world[f,joint]=world[f-1,joint]@Rotation.from_rotvec(delta*(limit/angle)).as_matrix()
                    rate_limited+=1
    final=fk_positions(world,parents,rest,roots)
    after_slip=[]
    for seg in segments:
        s,a,b=seg['side'],seg['start'],seg['end']
        core=np.flatnonzero(weights[a:b,s]>=.999)+a
        if len(core)>1:
            after_slip.extend(np.linalg.norm(np.diff(final[core,feet[s]][:,[0,2]],axis=0),axis=-1).tolist())
    result['world_rotations']=world.round(9).tolist()
    result['positions_are_solved']=True
    result['schema_version']='1.3' if motion.get('projection_refinement',{}).get('enabled') else '1.2'
    result['root_mode']='in_place_contact_vertical'
    result['warnings']=[w.replace('根骨固定，未恢复真实场景位移','水平根位移固定，骨盆高度可能随接地调整，未恢复真实场景位移') for w in result.get('warnings',[])]
    for frame,points in zip(result['frames'],final):
        frame['positions']=points.round(7).tolist()
    result['foot_contacts']={'enabled':True,'method':'image_support_hysteresis_ik_v1',
        'weights':weights.round(4).tolist(),'segments':segments,
        'locked_frame_counts':(weights>0).sum(axis=0).tolist(),
        'mean_support_horizontal_step_before_m':float(np.mean(before_slip)) if before_slip else None,
        'mean_support_horizontal_step_after_m':float(np.mean(after_slip)) if after_slip else None,
        'maximum_pelvis_adjustment_m':float(np.max(np.abs(root_shift))),
        'rate_limited_samples':rate_limited,
        'note':'Contact hypothesis, not measured foot contact or world travel. Releases raised/fast/unreliable feet. IK may not reach every anchor.'}
    result.setdefault('warnings',[]).append('接地为固定镜头下的支撑脚估计；低可信、抬脚和快速移动时不锁定，不能保证所有脚滑或跳跃准确。')
    return result
