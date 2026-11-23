"""Hair capsule self-contact, authored cross joints and floor in angular solve.

Bodies are immutable drivers. Constraints are bounded, iterative/discrete, not
a full Bullet/CCD implementation. Authored initial overlaps and adjacent or
joint-connected capsules are explicitly reported exclusions, not zero contacts.
"""
import numpy as np
from .secondary_hair import HairSolver,orthonormal,euler_matrix,matrix_euler,capped
from .secondary_surface import closest_segments


def native_joint_residual(heads,rotations,joint):
    a,b=joint['a'],joint['b']
    pa=heads[a]+rotations[a]@np.asarray(joint['anchor_a'])
    pb=heads[b]+rotations[b]@np.asarray(joint['anchor_b'])
    fa=rotations[a]@np.asarray(joint['frame_a']);fb=rotations[b]@np.asarray(joint['frame_b'])
    translation=fa.T@(pb-pa);angles=matrix_euler(fa.T@fb)
    linear_error=translation-np.clip(translation,joint['linear_min'],joint['linear_max'])
    angular_error=angles-np.clip(angles,joint['angular_min'],joint['angular_max'])
    return linear_error,angular_error,pa,pb,fa


class CoupledHairSolver:
    def __init__(self,nodes,settings,cross_joints,rest_heads,rest_tails,floor_z=0.):
        self.floor_z=float(floor_z)
        if not np.isfinite(self.floor_z):raise ValueError("地面高度必须有限")
        self.base=HairSolver(nodes,settings);self.nodes=self.base.nodes;self.settings=settings
        self.cross_joints=cross_joints;self.count=len(nodes)
        self.weld_locked_bones=[]
        for joint in cross_joints:
            a,b=joint['a'],joint['b']
            fixed=all(np.max(np.abs(joint[key]))<1e-9 for key in ('linear_min','linear_max','angular_min','angular_max'))
            # Two welded bodies anchored at separate points of the same parent
            # form a rigid child assembly. Use its neutral relative pose as the
            # conservative constraint manifold; the parent remains dynamic.
            if fixed and self.nodes[a]['parent']==self.nodes[b]['parent'] and self.nodes[a]['parent']>=0:
                for i in (a,b):
                    self.nodes[i]['angular_min']=[0.,0.,0.];self.nodes[i]['angular_max']=[0.,0.,0.]
                    self.weld_locked_bones.append(i)
        self.available=np.array([np.asarray(n['angular_max'])-np.asarray(n['angular_min'])>1e-8 for n in self.nodes])
        self.parents=np.array([n['parent'] for n in nodes]);self.radii=np.array([n['radius'] for n in nodes])
        self.ancestors=np.eye(self.count,dtype=bool)
        for i,n in enumerate(nodes):
            for parent in {n['parent'],n.get('angular_parent',n['parent'])}:
                if parent>=0:self.ancestors[i]|=self.ancestors[parent]
        adjacency=np.eye(self.count,dtype=bool)
        for i,n in enumerate(nodes):
            for parent in {n['parent'],n.get('angular_parent',n['parent'])}:
                if parent>=0:adjacency[i,parent]=adjacency[parent,i]=True
        adjacent_two=(adjacency.astype(int)@adjacency.astype(int))>0
        pairs=np.array([(a,b) for a in range(self.count) for b in range(a+1,self.count) if not adjacent_two[a,b]],int).reshape(-1,2)
        joined={tuple(sorted((j['a'],j['b']))) for j in cross_joints}
        pairs=np.asarray([p for p in pairs if tuple(p) not in joined],int).reshape(-1,2)
        if len(pairs):
            a,b,_,_=closest_segments(np.asarray(rest_heads)[pairs[:,0]],np.asarray(rest_tails)[pairs[:,0]],
                                    np.asarray(rest_heads)[pairs[:,1]],np.asarray(rest_tails)[pairs[:,1]])
            initial=np.linalg.norm(a-b,axis=1)<self.radii[pairs].sum(1)-1e-5
            self.authored_overlap_exclusions=pairs[initial].tolist();self.pairs=pairs[~initial]
        else:self.authored_overlap_exclusions=[];self.pairs=pairs
        self.self_contacts=0;self.floor_contacts=0;self.triangle_contacts=0;self.cross_projections=0
        self.merit_backoffs=0

    def energy(self,drivers,body_contact,external_torque):
        h,r,t,_,_=self.fk(drivers)
        value=.5*self.settings['rest_acceleration']*float(np.sum(self.base.angles**2))-float(np.sum(external_torque*self.base.angles))
        for joint in self.cross_joints:
            linear,angular,*_=native_joint_residual(h,r,joint)
            value+=5e5*float(np.dot(linear,linear))+2500*float(np.dot(angular,angular))
        if len(self.pairs):
            pairs=self.pairs;a,b,_,_=closest_segments(h[pairs[:,0]],t[pairs[:,0]],h[pairs[:,1]],t[pairs[:,1]])
            depth=np.maximum(self.radii[pairs].sum(1)-np.linalg.norm(a-b,axis=1),0)
            value+=5e5*float(np.sum(depth**2))
        for i,n in enumerate(self.nodes):
            points=h[i]+np.array([.35,.65,1.])[:,None]*(t[i]-h[i])
            floor=np.maximum(self.floor_z+n['radius']+.001-points[:,2],0)
            value+=5e5*float(np.sum(floor**2))
            if body_contact is not None:value+=5e5*sum(row[2]**2 for row in body_contact(points,n['radius']))
        return value

    def fk(self,drivers):
        heads=np.zeros((self.count,3));rotations=np.zeros((self.count,3,3));frames=np.zeros_like(rotations);axes=np.zeros_like(rotations)
        references=np.zeros_like(rotations)
        external_rotations={name:orthonormal(matrix[:3,:3]) for name,matrix in drivers.items()}
        for i,n in enumerate(self.nodes):
            if n['parent']>=0:pr=rotations[n['parent']];ph=heads[n['parent']]
            else:matrix=drivers[n['external_parent']];pr=external_rotations[n['external_parent']];ph=matrix[:3,3]
            ap=n.get('angular_parent',n['parent'])
            ar=(rotations[ap] if ap>=0 else external_rotations[n.get('angular_external_parent',n['external_parent'])])
            heads[i]=ph+pr@np.asarray(n['local_head']);ref=ar@np.asarray(n.get('angular_local_rotation',n['local_rotation']))
            frame=ar@np.asarray(n['joint_frame_parent']);angles=self.base.angles[i]
            local_frame=np.asarray(n['joint_frame_parent']);local_ref=np.asarray(n.get('angular_local_rotation',n['local_rotation']))
            rotations[i]=ar@local_frame@euler_matrix(angles)@local_frame.T@local_ref;frames[i]=frame;references[i]=ref
            x,y,z=angles;rz=euler_matrix([0,0,z]);ry=euler_matrix([0,y,0])
            axes[i]=np.column_stack((frame@rz@ry[:,0],frame@rz[:,1],frame[:,2]))
        tails=heads+rotations[:,:,1]*np.array([n['length'] for n in self.nodes])[:,None]
        return heads,rotations,tails,frames,axes

    def point_jacobian(self,bone,point,variable,heads,axes):
        if not self.ancestors[bone,variable]:return np.zeros((3,3))
        return np.cross(axes[variable].T,point-heads[variable]).T*self.available[variable][None,:]

    def apply_delta(self,index,delta,start_angles,dt):
        n=self.nodes[index];speed=self.settings['max_joint_speed_rad_s']+self.settings['contact_joint_speed_rad_s']
        candidate=self.base.angles[index]+capped(delta,self.settings['contact_joint_speed_rad_s']*dt)
        candidate=start_angles[index]+capped(candidate-start_angles[index],speed*dt)
        self.base.angles[index]=np.clip(candidate,n['angular_min'],n['angular_max'])
        self.base.angular_velocities[index]*=.25

    def step(self,drivers,body_contact,dt,force_enabled=True):
        old=self.base.positions.copy() if self.base.positions is not None else None
        start=self.base.angles.copy();old_heads,old_rots,_,old_frames,_=self.fk(drivers)
        self.base.step(drivers,None,dt,force_enabled)
        external_torque=np.zeros_like(start);roots=[];depths=[]
        gravity=np.asarray(self.settings['gravity_m_s2']) if force_enabled else np.zeros(3)
        for i,n in enumerate(self.nodes):
            if n['parent']>=0:roots.append(roots[n['parent']]);depths.append(depths[n['parent']]+1)
            else:roots.append(n['external_parent']);depths.append(0)
            position,_,_,omega,acceleration,alpha=self.base.driver_history[roots[i]]
            lever=old_heads[i]-position
            inertia=(-acceleration-np.cross(alpha,lever)-np.cross(omega,np.cross(omega,lever))) if force_enabled else np.zeros(3)
            inertia=capped(inertia,self.settings['driver_acceleration_limit_m_s2'])*self.settings['inertia_strength']/(1+.5*depths[i])
            external_torque[i]=old_frames[i].T@np.cross(old_rots[i,:,1],gravity+inertia)/max(n['length'],.025)
            if force_enabled:external_torque[i]-=old_frames[i].T@alpha*self.settings['inertia_strength']/(1+.5*depths[i])
        proposed=self.base.angles.copy();self.base.angles[:]=start
        initial_energy=self.energy(drivers,body_contact,external_torque);self.base.angles[:]=proposed
        for iteration in range(2):
            h,r,t,frames,axes=self.fk(drivers)
            accumulated=np.zeros_like(self.base.angles);weights=np.zeros(self.count)
            for joint in self.cross_joints:
                linear,angular,pa,pb,fa=native_joint_residual(h,r,joint)
                if max(np.linalg.norm(linear),np.linalg.norm(angular))<1e-7:continue
                a,b=joint['a'],joint['b'];blocks=[]
                for variable in (a,b):
                    point=self.point_jacobian(b,pb,variable,h,axes)-self.point_jacobian(a,pa,variable,h,axes)
                    # Small-angle rotational Jacobian in the joint-A frame.
                    rotational=axes[variable]*(int(self.ancestors[b,variable])-int(self.ancestors[a,variable]))*self.available[variable][None,:]
                    blocks.append(np.vstack((fa.T@point,.08*fa.T@rotational)))
                jacobian=np.column_stack(blocks);error=np.r_[linear,.08*angular]
                delta=-jacobian.T@np.linalg.solve(jacobian@jacobian.T+np.eye(6)*1e-6,error)
                accumulated[a]+=delta[:3]*.5;accumulated[b]+=delta[3:]*.5;weights[a]+=1;weights[b]+=1
                self.cross_projections+=1
            for i in np.flatnonzero(weights):self.apply_delta(i,accumulated[i]/weights[i],start,dt)
            h,r,t,frames,axes=self.fk(drivers)
            if len(self.pairs):
                pairs=self.pairs;a,b,s,u=closest_segments(h[pairs[:,0]],t[pairs[:,0]],h[pairs[:,1]],t[pairs[:,1]])
                distances=np.linalg.norm(a-b,axis=1);depth=self.radii[pairs].sum(1)-distances
                for row in np.flatnonzero(depth>1e-5):
                    ia,ib=map(int,pairs[row]);normal=a[row]-b[row]
                    if distances[row]<1e-10:
                        normal=np.cross(t[ia]-h[ia],t[ib]-h[ib])
                        if np.linalg.norm(normal)<1e-10:normal=np.array([0,0,1.])
                    normal=normal/max(np.linalg.norm(normal),1e-12)
                    jac=[]
                    for variable in (ia,ib):
                        j=self.point_jacobian(ia,a[row],variable,h,axes)-self.point_jacobian(ib,b[row],variable,h,axes)
                        jac.append(normal@j)
                    gradient=np.r_[*jac];delta=gradient*(depth[row]/max(np.dot(gradient,gradient),1e-8))*.35
                    self.apply_delta(ia,delta[:3],start,dt);self.apply_delta(ib,delta[3:],start,dt);self.self_contacts+=1
            h,r,t,frames,axes=self.fk(drivers)
            for i,n in enumerate(self.nodes):
                points=h[i]+np.array([.35,.65,1.])[:,None]*(t[i]-h[i])
                corrections=[]
                for point in points:
                    penetration=self.floor_z+n['radius']+.001-point[2]
                    if penetration>0:
                        corrections.append((point,np.array([0,0,1.]),penetration));self.floor_contacts+=1
                if body_contact is not None:
                    rows=body_contact(points,n['radius'])
                    corrections.extend(rows);self.triangle_contacts+=len(rows)
                if not corrections:continue
                point,normal,depth=max(corrections,key=lambda row:row[2])
                jac=normal@self.point_jacobian(i,point,i,h,axes)
                if np.dot(jac,jac)<1e-10:
                    # A vertical strand entering a floor has zero first-order
                    # normal Jacobian. Break symmetry with a bounded bend,
                    # never by translating its protected root/body.
                    allowed=np.asarray(n['angular_max'])-np.asarray(n['angular_min'])
                    effectiveness=np.linalg.norm(np.cross(axes[i].T,point-h[i]),axis=1)*allowed
                    axis=int(np.argmax(effectiveness));delta=np.zeros(3)
                    if effectiveness[axis]>1e-9:
                        delta[axis]=.1 if n['angular_max'][axis]-self.base.angles[i,axis]>=self.base.angles[i,axis]-n['angular_min'][axis] else -.1
                else:delta=jac*(min(depth,.025)/np.dot(jac,jac))*.5
                self.apply_delta(i,delta,start,dt)
        candidate=self.base.angles.copy();candidate_energy=self.energy(drivers,body_contact,external_torque)
        if candidate_energy>initial_energy+1e-8:
            accepted=False
            for fraction in (.5,.25,.125,.0625):
                self.base.angles[:]=start+fraction*(candidate-start)
                if self.energy(drivers,body_contact,external_torque)<=initial_energy+1e-8:
                    accepted=True;break
            if not accepted:self.base.angles[:]=start
            self.base.angular_velocities[:]=0;self.merit_backoffs+=1
        h,r,t,_,_=self.fk(drivers);self.base.positions=t
        self.base.velocities=np.zeros_like(t) if old is None else (t-old)/dt
        if not np.isfinite(t).all():raise ValueError('耦合发束产生非有限值')
        return h,r,t,self.base.angles.copy()

    def residuals(self,drivers):
        h,r,t,_,_=self.fk(drivers);linear=[];angular=[]
        for joint in self.cross_joints:
            a,b,*_=native_joint_residual(h,r,joint);linear.append(np.linalg.norm(a));angular.append(np.linalg.norm(b))
        pairs=self.pairs
        if len(pairs):
            a,b,_,_=closest_segments(h[pairs[:,0]],t[pairs[:,0]],h[pairs[:,1]],t[pairs[:,1]])
            depth=np.maximum(self.radii[pairs].sum(1)-np.linalg.norm(a-b,axis=1),0)
        else:depth=np.zeros(1)
        return {'cross_linear_violation_max_m':float(max(linear,default=0)),
                'cross_angular_violation_max_rad':float(max(angular,default=0)),
                'self_capsule_contact_count':int(np.sum(depth>1e-4)),'self_capsule_depth_max_m':float(depth.max()),
                'floor_depth_max_m':float(np.maximum(self.floor_z+self.radii-t[:,2],0).max())}
