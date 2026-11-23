"""Conservative arm fitting to reliable image observations; not accuracy truth."""
import copy
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from .exporters import solve_rotations,_swing
from .refinement import fit_two_bone
from .stability import fk_positions


def reconcile_world_arms(motion,wc,camera_root,intrinsics,keypoints):
    result=copy.deepcopy(motion)
    if result.get('image_arm_refinement',{}).get('enabled'):return result
    _,world,poses=solve_rotations(motion);rest=np.asarray(motion['rest_positions']);n=len(poses)
    wc=np.asarray(wc,float);camera_root=np.asarray(camera_root,float);K=np.broadcast_to(intrinsics,(n,3,3)).copy();kp=np.asarray(keypoints,float)
    if motion.get('root_mode')!='estimated_world_translation' or len(motion['joint_names'])!=22:raise ValueError('Image arm fitting requires world SMPL22')
    if wc.shape!=(n,3,3) or camera_root.shape!=(n,3) or kp.shape!=(n,17,3) or not all(np.isfinite(x).all() for x in [wc,camera_root,K,kp]):raise ValueError('Invalid camera/observation tracks')
    if np.min(abs(np.linalg.det(K)))<1e-8:raise ValueError('Invalid camera intrinsics')
    camera=np.einsum('fij,fkj->fki',wc,poses-poses[:,:1])+camera_root[:,None]
    def project(p):
        h=np.einsum('fij,fkj->fki',K,p);return h[...,:2]/np.maximum(h[...,2:],1e-6)
    pixels=project(camera);final=world.copy();reports={}
    for label,(a,b,c),observed in [('left',(16,18,20),(5,7,9)),('right',(17,19,21),(6,8,10))]:
        confidence=np.min(kp[:,observed,2],axis=1)
        trust=np.clip((confidence-.70)/.20,0,1)
        # Only coherent image tracks; reject implausibly large one-frame jumps.
        size=np.maximum(np.linalg.norm(kp[:,5,:2]-kp[:,6,:2],axis=1),30.)
        steps=np.linalg.norm(np.diff(kp[:,observed,:2],axis=0),axis=-1).max(1)
        if n>1:
            bad=steps>size[1:]*.65;trust[1:][bad]=0;trust[:-1][bad]=0
        trust=np.minimum(trust,gaussian_filter1d(trust,max(.5,motion['fps']*.06),mode='nearest'))
        # A confidence threshold must not switch several degrees of correction
        # on/off in one frame. This minorant never increases measured trust.
        step=3./motion['fps']
        for f in range(1,n):trust[f]=min(trust[f],trust[f-1]+step)
        for f in range(n-2,-1,-1):trust[f]=min(trust[f],trust[f+1]+step)
        desired=pixels[:,a,None,:]+kp[:,observed[1:],:2]-kp[:,observed[0],None,:2]
        rays=np.concatenate([desired,np.ones((n,2,1))],axis=-1)
        rays=np.einsum('fij,fkj->fki',np.linalg.inv(K),rays)
        goal_cam=rays*camera[:,[b,c],2:]
        goal_world=np.einsum('fij,fkj->fki',wc.swapaxes(1,2),goal_cam-camera_root[:,None])+poses[:,:1]
        delta=np.zeros((n,2,3));lengths=[np.linalg.norm(rest[b]-rest[a]),np.linalg.norm(rest[c]-rest[b])]
        for f in np.flatnonzero(trust>0):
            elbow,wrist=fit_two_bone(poses[f,a],goal_world[f,1],goal_world[f,0],*lengths,poses[f,b])
            for k,old,new in [(0,poses[f,b]-poses[f,a],elbow-poses[f,a]),(1,poses[f,c]-poses[f,b],wrist-elbow)]:
                delta[f,k]=Rotation.from_matrix(_swing(old,new)).as_rotvec()*trust[f]*.65
        # Filter the correction, not the original gesture; limit to 12 degrees.
        delta=gaussian_filter1d(delta,max(.5,motion['fps']*.075),axis=0,mode='nearest')
        delta*=trust[:,None,None]
        delta*=np.minimum(1,np.deg2rad(12)/np.maximum(np.linalg.norm(delta,axis=-1,keepdims=True),1e-12))
        before=np.linalg.norm(pixels[:,[b,c]]-desired,axis=-1).mean(1)
        strength=np.ones(n)
        for _ in range(12):
            adjusted=Rotation.from_rotvec((delta*strength[:,None,None]).reshape(-1,3)).as_matrix().reshape(n,2,3,3)
            candidate=world.copy();candidate[:,a]=adjusted[:,0]@world[:,a];candidate[:,b]=adjusted[:,1]@world[:,b];candidate[:,c]=adjusted[:,1]@world[:,c]
            solved=fk_positions(candidate,motion['parents'],rest,poses[:,0])
            cp=np.einsum('fij,fkj->fki',wc,solved-solved[:,:1])+camera_root[:,None]
            after=np.linalg.norm(project(cp)[:,[b,c]]-desired,axis=-1).mean(1)
            bad=(after>before+.05)|(np.linalg.norm(solved[:,c]-poses[:,c],axis=1)>.06)
            if not bad.any():break
            strength[bad]*=.5
            strength=np.minimum(strength,gaussian_filter1d(strength,max(.5,motion['fps']*.04),mode='nearest'))
        # A final rejected frame keeps its original pose, never a worse fit.
        candidate[bad]=world[bad]
        after[bad]=before[bad]
        for j in [a,b,c]:final[:,j]=candidate[:,j]
        valid=trust>0
        reports[label]={'trusted_frames':int(valid.sum()),'mean_shoulder_relative_pixel_error_before':float(before[valid].mean()) if valid.any() else None,
            'mean_shoulder_relative_pixel_error_after':float(after[valid].mean()) if valid.any() else None,
            'maximum_rotation_adjustment_deg':float(np.rad2deg(np.linalg.norm(delta*strength[:,None,None],axis=-1)).max()),
            'rejected_frames':int(bad.sum()),'note':'Agreement with the same 2D detector, not ground-truth accuracy. Depth retained as a prior.'}
    solved=fk_positions(final,motion['parents'],rest,poses[:,0]);result['world_rotations']=final.tolist()
    for f,p in zip(result['frames'],solved):f['positions']=p.tolist()
    result['image_arm_refinement']={'enabled':True,'method':'confidence_capped_camera_arm_ik_v1','sides':reports,'rotation_limit_deg':12.,'wrist_displacement_limit_m':.06,'root_torso_legs_unchanged':True}
    result.setdefault('warnings',[]).append('二维肘腕观测仅在可靠且连续时校正三维手臂，深度仍为模型估计；观测贴合不等于真实动作准确率。')
    solve_rotations(result)
    return result


def refine_from_archive(motion,path,keypoints):
    with np.load(path,allow_pickle=False) as data:
        wc=Rotation.from_rotvec(data['camera_global_orient']).as_matrix()@Rotation.from_rotvec(data['world_global_orient']).as_matrix().swapaxes(-1,-2)
        return reconcile_world_arms(motion,wc,data['joints_camera'][:,0],data['K'],keypoints)
