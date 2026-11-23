"""Bounded arm/torso proxy avoidance, shared by source and target rigs.

NumPy only so the same solver runs inside Blender. This is a geometric prior,
not a measured body surface, cloth simulation, or recovered human intent.
"""
import copy
import numpy as np


def _unit(value):
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-12)


def torso_space(hips, left, right, rest_hips, rest_left, rest_right):
    """Animated orthonormal frame and fixed radii from this rig's proportions."""
    hips, left, right = [np.asarray(v,dtype=float) for v in (hips,left,right)]
    top = (left + right) * .5
    y = _unit(top - hips)
    x = left - right
    x = _unit(x - y * np.sum(x * y, axis=-1, keepdims=True))
    z = _unit(np.cross(x, y))
    basis = np.stack([x, y, z], axis=-1)
    width = float(np.linalg.norm(np.asarray(rest_left,dtype=float) - rest_right))
    height = float(np.linalg.norm((np.asarray(rest_left,dtype=float) + rest_right) * .5 - rest_hips))
    if min(width, height) < .01 or np.min(np.linalg.det(basis)) < .99:
        raise ValueError('Degenerate torso geometry')
    return hips + (top - hips) * .52, basis, np.array([width*.43, height*.60, width*.28]), width*.085


def _smooth(values, fps):
    radius = max(1, round(fps * .07))
    offsets = np.arange(-radius, radius+1)
    weights = np.exp(-.5*(offsets/max(1., fps*.035))**2)
    weights /= weights.sum()
    padded = np.pad(values, [(radius,radius)]+[(0,0)]*(values.ndim-1), mode='edge')
    return sum(weight*padded[i:i+len(values)] for i,weight in enumerate(weights))


def _cap_direction(value, original, angle):
    value = _unit(value)
    cosine = np.sum(value*original, axis=-1, keepdims=True)
    tangent = _unit(value - cosine*original)
    return np.where(cosine < np.cos(angle), original*np.cos(angle)+tangent*np.sin(angle), value)


def arm_clearance(points, center, basis, radii, arm_radius):
    """Signed radial proxy clearance in metres; NOT mesh penetration depth.

    Exclude the proximal shoulder attachment where body and arm must connect.
    Sample entire forearm including wrist, and distal 75% of upper arm.
    """
    local = np.einsum('nij,nkj->nki', basis.swapaxes(1,2), np.asarray(points)-center[:,None])
    a,b,c = local[:,0],local[:,1],local[:,2]
    t = np.linspace(0,1,13)
    samples = np.concatenate([a[:,None]+(b-a)[:,None]*np.linspace(.25,1,10)[None,:,None],
                              b[:,None]+(c-b)[:,None]*t[None,:,None]],axis=1)
    expanded = np.asarray(radii)+arm_radius
    return (np.linalg.norm(samples/expanded,axis=-1)-1)*np.min(expanded)


def avoid_arm_torso(points, center, basis, radii, arm_radius, fps, max_angle_deg=25.):
    """Minimise proxy intersections with fixed bone lengths and bounded edits.

    Solve in moving torso coordinates. Smooth only correction directions, not
    the input motion, and keep a pose spring to avoid changing clear gestures.
    A rotation cap takes priority over fully resolving impossible poses.
    """
    points = np.asarray(points,float);center=np.asarray(center,float);basis=np.asarray(basis,float)
    if points.ndim!=3 or points.shape[1:]!=(3,3) or len(points)==0 or center.shape!=(len(points),3) or basis.shape!=(len(points),3,3):
        raise ValueError('Invalid arm track shapes')
    if not all(np.isfinite(x).all() for x in [points,center,basis,radii,arm_radius,fps,max_angle_deg]) or fps<=0 or np.min(radii)<=0 or arm_radius<0 or not 0<max_angle_deg<=45:
        raise ValueError('Invalid arm solver inputs')
    before=arm_clearance(points,center,basis,radii,arm_radius)
    local=np.einsum('nij,nkj->nki',basis.swapaxes(1,2),points-center[:,None])
    a=local[:,0];lengths=np.linalg.norm(np.diff(local,axis=1),axis=-1)
    if np.min(lengths)<1e-5:raise ValueError('Degenerate arm lengths')
    u0=_unit(local[:,1]-a);v0=_unit(local[:,2]-local[:,1]);u=u0.copy();v=v0.copy()
    # Broad-phase only correction neighbourhoods, leaving unrelated motion exact.
    active=(before.min(axis=1)<0).astype(float)
    for _ in range(3):active=np.maximum(active,_smooth(active,fps))
    active=active>1e-8
    if active.any():
        r=np.asarray(radii)+arm_radius;lu=lengths[:,0,None];lv=lengths[:,1,None]
        # Sample locations p = a + alpha*Lu*u + beta*Lv*v.
        alpha=np.r_[np.linspace(.25,1,10),np.ones(13)]
        beta=np.r_[np.zeros(10),np.linspace(0,1,13)]
        angle=np.deg2rad(max_angle_deg)
        for _ in range(160):
            sample=a[:,None]+alpha[None,:,None]*lu[:,None]*u[:,None]+beta[None,:,None]*lv[:,None]*v[:,None]
            norm=np.linalg.norm(sample/r,axis=-1,keepdims=True)
            normal=_unit(sample/(r*r))
            # Exact centre has no radial direction. Use this arm's original
            # shoulder direction, never an arbitrary alternating front/back.
            fallback=_unit(a)[:,None]
            normal=np.where(norm<1e-9,fallback,normal)
            force=normal*np.maximum(1.005-norm,0)*np.min(r)
            gu=np.sum(force*alpha[None,:,None],axis=1)/4
            gv=np.sum(force*beta[None,:,None],axis=1)/4
            du=(gu-u*np.sum(gu*u,axis=-1,keepdims=True))/lu
            dv=(gv-v*np.sum(gv*v,axis=-1,keepdims=True))/lv
            u=_cap_direction(u+.65*du-.006*(u-u0),u0,angle)
            v=_cap_direction(v+.65*dv-.006*(v-v0),v0,angle)
            u=_cap_direction(u0+_smooth(u-u0,fps),u0,angle)
            v=_cap_direction(v0+_smooth(v-v0,fps),v0,angle)
            u[~active]=u0[~active];v[~active]=v0[~active]
        local[:,1]=a+lu*u;local[:,2]=local[:,1]+lv*v
    result=np.einsum('nij,nkj->nki',basis,local)+center[:,None]
    result[~active]=points[~active]
    after=arm_clearance(result,center,basis,radii,arm_radius)
    displacement=np.linalg.norm(result-points,axis=-1)
    old_directions=_unit(np.diff(points,axis=1));new_directions=_unit(np.diff(result,axis=1))
    angles=np.rad2deg(np.arccos(np.clip(np.sum(old_directions*new_directions,axis=-1),-1,1)))
    return result,{'before_penetrating_frames':int((before.min(axis=1)<-.001).sum()),
        'after_penetrating_frames':int((after.min(axis=1)<-.001).sum()),
        'before_min_proxy_clearance_m':float(before.min()),'after_min_proxy_clearance_m':float(after.min()),
        'before_total_proxy_deficit_m':float(np.maximum(-before,0).sum()),
        'after_total_proxy_deficit_m':float(np.maximum(-after,0).sum()),
        'max_joint_adjustment_m':float(displacement.max()),'segment_adjustment_limit_deg':max_angle_deg,
        'max_segment_adjustment_deg':float(angles.max()),
        'corrected_frames':int((displacement.max(axis=1)>1e-6).sum())}


def constrain_body_collision(motion):
    from .exporters import solve_rotations,_swing
    from .stability import fk_positions
    from .global_motion import SMPL22_NAMES,SMPL22_PARENTS
    result=copy.deepcopy(motion)
    if result.get('body_collision',{}).get('enabled'):return result
    if motion.get('root_mode')!='estimated_world_translation' or motion['joint_names']!=SMPL22_NAMES or motion['parents']!=SMPL22_PARENTS:
        raise ValueError('Body collision requires SMPL22 world motion')
    _,world,poses=solve_rotations(motion);world=world.copy();rest=np.asarray(motion['rest_positions'])
    center,basis,radii,radius=torso_space(poses[:,0],poses[:,16],poses[:,17],rest[0],rest[16],rest[17])
    reports={}
    for label,indices in [('left',[16,18,20]),('right',[17,19,21])]:
        solved,reports[label]=avoid_arm_torso(poses[:,indices],center,basis,radii,radius,motion['fps'])
        for f in range(len(poses)):
            for a,b,k in [(indices[0],indices[1],0),(indices[1],indices[2],1)]:
                delta=_swing(poses[f,b]-poses[f,a],solved[f,k+1]-solved[f,k])
                world[f,a]=delta@world[f,a]
                if k==1:world[f,indices[2]]=delta@world[f,indices[2]]
    final=fk_positions(world,motion['parents'],rest,poses[:,0])
    result['world_rotations']=world.tolist()
    for frame,points in zip(result['frames'],final):frame['positions']=points.tolist()
    result['body_collision']={'enabled':True,'method':'bounded_temporal_arm_torso_proxy_v1',
        'radii_m':radii.tolist(),'arm_radius_m':radius,'sides':reports,
        'root_and_legs_unchanged':True,'note':'Proportion-based torso ellipsoid and sampled arm capsules. Bounded corrections can leave collisions. No mesh/cloth/head/inter-arm collision or capture accuracy claim.'}
    result.setdefault('warnings',[]).append('手臂避碰使用躯干几何代理，最多调整每段25度；可能改变贴身手势，复杂穿插与衣裙仍可能存在。')
    solve_rotations(result)
    return result
