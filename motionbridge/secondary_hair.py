"""Experimental one-way spring-bone hair dynamics, in metres and seconds.

Only hair rotations are returned. Driving body transforms are immutable inputs.
This is NOT a PMX/Bullet six-DOF emulator: parent joints constrain local XYZ
angular displacement, segment length is exact, cross-strand joints and hair
self-collision are not solved. Body collision uses explicit model proxies.
"""
from __future__ import annotations
import math
import numpy as np


def orthonormal(matrix):
    u,_,v=np.linalg.svd(matrix)
    result=u@v
    if np.linalg.det(result)<0:
        u[:,-1]*=-1;result=u@v
    return result


def unit(v, fallback=None):
    length = np.linalg.norm(v)
    if length < 1e-12:
        if fallback is None:
            raise ValueError('零长度方向')
        return np.asarray(fallback, dtype=float).copy()
    return v / length


def swing(a, b):
    a, b = unit(a), unit(b)
    cross = np.cross(a, b)
    cosine = float(np.clip(np.dot(a,b), -1, 1))
    if cosine < -1 + 1e-8:
        axis = unit(np.cross(a, [1,0,0] if abs(a[0]) < .8 else [0,1,0]))
        return 2*np.outer(axis,axis)-np.eye(3)
    matrix = np.array([[0,-cross[2],cross[1]],[cross[2],0,-cross[0]],[-cross[1],cross[0],0]])
    return np.eye(3)+matrix+matrix@matrix/(1+cosine)


def euler_matrix(xyz):
    x,y,z = xyz
    cx,cy,cz = np.cos(xyz); sx,sy,sz = np.sin(xyz)
    return np.array([[cz*cy,cz*sy*sx-sz*cx,cz*sy*cx+sz*sx],
                     [sz*cy,sz*sy*sx+cz*cx,sz*sy*cx-cz*sx],[-sy,cy*sx,cy*cx]])


def matrix_euler(matrix):
    y = math.asin(float(np.clip(-matrix[2,0],-1,1)))
    if abs(math.cos(y)) < 1e-7:
        return np.array([math.atan2(-matrix[1,2],matrix[1,1]), y, 0.])
    return np.array([math.atan2(matrix[2,1],matrix[2,2]),y,math.atan2(matrix[1,0],matrix[0,0])])


def limit_rotation(candidate, reference, joint_frame, low, high):
    delta = joint_frame.T @ candidate @ reference.T @ joint_frame
    angles = np.clip(matrix_euler(delta), low, high)
    return orthonormal(joint_frame @ euler_matrix(angles) @ joint_frame.T @ reference), angles


def proxy_contacts(points, centers, rotations, kinds, sizes):
    """Signed distances and outward normals for sphere/box/Z-capsule proxies.

Shapes encode [radius, cylinder_height,0] or box half extents. Results are
analytic proxy contacts, not a claim about the actual skinned hair surface.
"""
    local = np.einsum('mji,kmj->kmi', rotations, points[:,None,:]-centers[None,:,:])
    normals = local.copy()
    capsule = kinds == 2
    normals[:,capsule,2] -= np.clip(local[:,capsule,2],-sizes[capsule,1]/2,sizes[capsule,1]/2)
    norm = np.linalg.norm(normals,axis=2)
    distance = norm-sizes[None,:,0]
    normals /= np.maximum(norm[:,:,None],1e-12)
    zero = norm < 1e-12
    normals[zero] = [1,0,0]
    box = np.where(kinds == 1)[0]
    if len(box):
        coords = local[:,box,:]
        q = np.abs(coords)-sizes[None,box,:]
        outside = np.maximum(q,0)
        outside_length = np.linalg.norm(outside,axis=2)
        distance[:,box] = outside_length+np.minimum(np.max(q,axis=2),0)
        n = np.sign(coords)*outside/np.maximum(outside_length[:,:,None],1e-12)
        inside = outside_length < 1e-12
        for row,col in zip(*np.where(inside)):
            axis = int(np.argmax(q[row,col]))
            n[row,col] = 0
            n[row,col,axis] = 1 if coords[row,col,axis] >= 0 else -1
        normals[:,box,:] = n
    return distance, np.einsum('mij,kmj->kmi',rotations,normals)


def validate_settings(settings):
    if settings.get('method') != 'spring_bones_v1':
        raise ValueError('不支持的发束求解器')
    for key,low,high in [('substeps',2,16),('collision_iterations',1,8)]:
        value=settings.get(key)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f'非法头发参数: {key}')
    for key,low,high in [('warmup_seconds',.5,5),('drag_per_second',.1,30),
                         ('rest_acceleration',0,1000),('collision_margin_m',0,.02),('strand_radius_scale',.05,1),
                         ('driver_force_filter_hz',1,30),('driver_acceleration_limit_m_s2',1,100),
                         ('driver_angular_acceleration_limit_rad_s2',1,200),
                         ('max_joint_speed_rad_s',.1,10),('contact_joint_speed_rad_s',.1,10),('inertia_strength',0,1)]:
        value=settings.get(key)
        if type(value) not in (float,int) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'非法头发参数: {key}')
    gravity=np.asarray(settings.get('gravity_m_s2'),dtype=float)
    if gravity.shape != (3,) or not np.isfinite(gravity).all() or np.linalg.norm(gravity)>30:
        raise ValueError('非法重力')
    if settings.get('body_collision_policy') != 'explicit_hair_body_override':
        raise ValueError('必须明确原生掩码之外的头发/身体碰撞策略')


def select_hair(plan, settings):
    validate_settings(settings)
    tokens=settings.get('include_bone_tokens',[]); excluded=settings.get('exclude_bone_tokens',[])
    if not tokens or any(not isinstance(t,str) or not t for t in tokens+excluded):
        raise ValueError('发束选择必须使用非空的模型配置标记')
    allowed=set(plan['selected_secondary_bone_ids'])-set(plan['protected_bone_ids'])
    ids=[b['id'] for b in plan['bones'] if b['id'] in allowed and any(t in b['name'] for t in tokens)
         and not any(t in b['name'] for t in excluded)]
    if not ids:
        raise ValueError('未选择任何安全发束')
    # Selection must belong to a hair surface or its native ancestor chain.
    hair_vertices=[v for region in plan['regions'] if region['role']=='hair' for v in region['vertex_ids']]
    weighted={int(b) for v in hair_vertices for b in plan['geometry']['skin_weights'][v]}
    family=set(weighted)
    for index in list(weighted):
        seen=set()
        while index not in (None,-1):
            if index in seen: raise ValueError('发束父级环')
            seen.add(index); family.add(index); index=plan['bones'][index]['parent']
    # Keep unweighted terminal bones of an explicitly selected hair chain.
    pending=set(ids)-family
    while pending:
        attached={i for i in pending if plan['bones'][i]['parent'] in (family & set(ids))}
        if not attached: break
        family.update(attached); pending-=attached
    if pending:
        raise ValueError('名称候选不属于头发蒙皮及其骨链')
    if len({plan['bones'][i]['blender_name'] for i in ids}) != len(ids):
        raise ValueError('发束绑定重复')
    return ids


def rotation_vector(matrix):
    angle=math.acos(float(np.clip((np.trace(matrix)-1)/2,-1,1)))
    skew=np.array([matrix[2,1]-matrix[1,2],matrix[0,2]-matrix[2,0],matrix[1,0]-matrix[0,1]])
    return skew*.5 if angle<1e-6 else skew*(angle/(2*math.sin(angle)))


def capped(vector,maximum):
    return vector*min(1.,maximum/max(np.linalg.norm(vector),1e-12))


class HairSolver:
    """Implicit damped angular springs; immutable body drivers, fixed lengths.

Angular state avoids injecting parent endpoint corrections into descendant
linear velocities. Bounded driver acceleration and contact angular speed are
explicit experimental model parameters, not changes to the driving action.
"""
    def __init__(self,nodes,settings):
        validate_settings(settings)
        self.nodes=[dict(n) for n in nodes]; self.settings=dict(settings)
        for node in self.nodes:
            for key in ('local_rotation','joint_frame_parent','angular_local_rotation'):
                if key in node:node[key]=orthonormal(np.asarray(node[key],float))
        self.positions=None; self.velocities=np.zeros((len(nodes),3))
        self.angles=np.zeros((len(nodes),3));self.angular_velocities=np.zeros_like(self.angles)
        self.driver_history={}
        for i,n in enumerate(nodes):
            if not -1 <= n['parent'] < i or not 1e-5 < n['length'] < 1:
                raise ValueError('发束必须拓扑排序且具有有效长度')
            if any(a>b for a,b in zip(n['angular_min'],n['angular_max'])):
                raise ValueError('发束角限位上下界非法')
        self.contacts_applied=0;self.contact_rollbacks=0
        self.maximum_length_error=0.; self.maximum_limit_error=0.

    def driver_forces(self,drivers,dt):
        result={}
        for name,matrix in drivers.items():
            position=matrix[:3,3];rotation=matrix[:3,:3]
            previous=self.driver_history.get(name)
            if previous is None:
                velocity=omega=acceleration=alpha=np.zeros(3)
            else:
                old_position,old_rotation,old_velocity,old_omega,old_acceleration,old_alpha=previous
                velocity=(position-old_position)/dt;omega=rotation_vector(rotation@old_rotation.T)/dt
                blend=1-math.exp(-2*math.pi*self.settings['driver_force_filter_hz']*dt)
                acceleration=old_acceleration+blend*((velocity-old_velocity)/dt-old_acceleration)
                alpha=old_alpha+blend*((omega-old_omega)/dt-old_alpha)
                acceleration=capped(acceleration,self.settings['driver_acceleration_limit_m_s2'])
                alpha=capped(alpha,self.settings['driver_angular_acceleration_limit_rad_s2'])
            self.driver_history[name]=(position.copy(),rotation.copy(),velocity.copy(),omega.copy(),acceleration.copy(),alpha.copy())
            result[name]=(position,omega,acceleration,alpha)
        return result

    def step(self, external_parents, colliders, dt, force_enabled=True):
        if not math.isfinite(dt) or not 0 < dt <= 1/30:
            raise ValueError('积分步长过大或无效')
        heads=np.zeros((len(self.nodes),3)); rotations=np.zeros((len(self.nodes),3,3))
        tails=np.zeros_like(heads); angles=np.zeros_like(heads)
        initialized=self.positions is not None
        old=self.positions.copy() if initialized else None
        gravity=np.asarray(self.settings['gravity_m_s2']) if force_enabled else np.zeros(3)
        forces=self.driver_forces(external_parents,dt)
        fractions=np.array([.35,.65,1.])
        roots=[];depths=[]
        for i,node in enumerate(self.nodes):
            if node['parent'] >= 0:
                parent_rotation=rotations[node['parent']]; parent_head=heads[node['parent']]
                roots.append(roots[node['parent']]);depths.append(depths[node['parent']]+1)
            else:
                matrix=external_parents[node['external_parent']]
                parent_rotation=matrix[:3,:3]; parent_head=matrix[:3,3]
                roots.append(node['external_parent']);depths.append(0)
            head=parent_head+parent_rotation@np.asarray(node['local_head'])
            angular_parent=node.get('angular_parent',node['parent'])
            angular_rotation=(rotations[angular_parent] if angular_parent>=0 else
                              external_parents[node.get('angular_external_parent',node['external_parent'])][:3,:3])
            reference=angular_rotation@np.asarray(node.get('angular_local_rotation',node['local_rotation']))
            joint_frame=angular_rotation@np.asarray(node['joint_frame_parent'])
            previous_angles=self.angles[i].copy();contacts_before=self.contacts_applied
            angular=previous_angles.copy()
            orientation=joint_frame@euler_matrix(angular)@joint_frame.T@reference
            origin,omega,acceleration,alpha=forces[roots[i]]
            lever=head-origin
            inertial=(-acceleration-np.cross(alpha,lever)-np.cross(omega,np.cross(omega,lever))) if force_enabled else np.zeros(3)
            inertial=capped(inertial,self.settings['driver_acceleration_limit_m_s2'])*self.settings['inertia_strength']/(1+.5*depths[i])
            torque=joint_frame.T@np.cross(orientation[:,1],gravity+inertial)/max(node['length'],.025)
            if force_enabled:torque-=joint_frame.T@alpha*self.settings['inertia_strength']/(1+.5*depths[i])
            stiffness=self.settings['rest_acceleration'];drag=self.settings['drag_per_second']
            velocity=(self.angular_velocities[i]+dt*(torque-stiffness*angular))/(1+drag*dt+stiffness*dt*dt)
            velocity=capped(velocity,self.settings['max_joint_speed_rad_s'])
            proposed=angular+velocity*dt
            angular=np.clip(proposed,node['angular_min'],node['angular_max'])
            velocity[np.abs(proposed-angular)>1e-12]=0
            for iteration in range(self.settings['collision_iterations']+1):
                orientation=orthonormal(joint_frame@euler_matrix(angular)@joint_frame.T@reference)
                candidate=head+orientation[:,1]*node['length']
                if colliders is None or iteration == self.settings['collision_iterations']:
                    break
                points=head+fractions[:,None]*(candidate-head)
                distance,normal=proxy_contacts(points,**colliders)
                penetration=node['radius']+self.settings['collision_margin_m']-distance
                row,col=np.unravel_index(int(np.argmax(penetration)),penetration.shape)
                if penetration[row,col] <= 0: break
                corrected=candidate+normal[row,col]*min(float(penetration[row,col])/fractions[row],.08)
                desired=swing(orientation[:,1],unit(corrected-head,orientation[:,1]))@orientation
                _,desired_angles=limit_rotation(desired,reference,joint_frame,node['angular_min'],node['angular_max'])
                angular+=capped(desired_angles-angular,self.settings['contact_joint_speed_rad_s']*dt/self.settings['collision_iterations'])
                angular=np.clip(angular,node['angular_min'],node['angular_max'])
                velocity*=.35
                self.contacts_applied+=1
            if colliders is not None and self.contacts_applied>contacts_before:
                previous_orientation=orthonormal(joint_frame@euler_matrix(previous_angles)@joint_frame.T@reference)
                previous_tail=head+previous_orientation[:,1]*node['length']
                points=np.concatenate((head+fractions[:,None]*(previous_tail-head),
                                       head+fractions[:,None]*(candidate-head)))
                distance,_=proxy_contacts(points,**colliders)
                old_depth=node['radius']+self.settings['collision_margin_m']-distance[:3].min()
                new_depth=node['radius']+self.settings['collision_margin_m']-distance[3:].min()
                if new_depth>max(0.,old_depth)+1e-7:
                    # Unilateral contact reaction: reject a force update that
                    # worsens penetration after limited angular projection.
                    # Keep the old angle in the CURRENT moving parent frame;
                    # never freeze a world-space bone or edit a body driver.
                    angular=previous_angles;orientation=previous_orientation;candidate=previous_tail
                    velocity[:]=0;self.contact_rollbacks+=1
            self.angles[i]=angular;self.angular_velocities[i]=velocity
            heads[i]=head; rotations[i]=orientation; tails[i]=candidate; angles[i]=angular
            self.maximum_length_error=max(self.maximum_length_error,abs(np.linalg.norm(candidate-head)-node['length']))
            self.maximum_limit_error=max(self.maximum_limit_error,float(np.max(np.maximum(
                np.asarray(node['angular_min'])-angular,angular-np.asarray(node['angular_max'])))))
        if not np.isfinite(tails).all():
            raise ValueError('发束求解产生非有限值')
        self.positions=tails
        if initialized:
            self.velocities=(tails-old)/dt
        return heads,rotations,tails,angles
