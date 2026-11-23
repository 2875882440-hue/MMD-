"""Explicit secondary attachment DOFs and local two-sided constraint projection.

No collision/integration/export solver is implemented here. Source arrays and
bone matrices are copied; only secondary frame state is modified. Numerical
Jacobians include the rotating surface offset, including the target-side force.
"""
import copy
import math
import numpy as np
from .secondary_contact_policy import validate_bindings
from .secondary_segmentation import _sha
from .secondary_surface import rigid_fit


def _array(value, shape, name):
    a=np.array(value,dtype=float,copy=True)
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError(name+'尺寸错误或非有限')
    return a


def _rotation(r):
    r=_array(r,(3,3),'转动矩阵')
    u,scales,vt=np.linalg.svd(r)
    # A volume/determinant threshold sums independent float32 axis errors.
    # Bound each singular scale instead, then remove only this roundoff.
    if np.linalg.det(r)<=0 or np.max(np.abs(scales-1.))>1e-6:
        raise ValueError('刚体转动不能含缩放或镜像')
    return u@vt  # Remove float32 roundoff only; never normalize genuine scale.


def _matrix(m):
    m=_array(m,(4,4),'骨骼矩阵')
    if not np.allclose(m[3],[0,0,0,1],atol=1e-9,rtol=0):raise ValueError('骨骼矩阵不是仿射矩阵')
    if abs(np.linalg.det(m[:3,:3]))<1e-12:raise ValueError('骨座静态/当前矩阵不可逆')
    return m


def exp_rotation(v):
    v=_array(v,(3,),'角度增量');theta=np.linalg.norm(v)
    if theta<1e-15:return np.eye(3)
    x,y,z=v/theta;k=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+math.sin(theta)*k+(1-math.cos(theta))*(k@k)


def rotation_vector(r):
    r=_rotation(r);theta=math.acos(float(np.clip((np.trace(r)-1)/2,-1,1)))
    skew=np.array([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])
    if theta<1e-8:return skew/2
    if math.pi-theta<1e-5:
        _,vectors=np.linalg.eigh((r+r.T+2*np.eye(3))/4);axis=vectors[:,-1]
        if np.linalg.norm(skew)>1e-10:
            if np.dot(axis,skew)<0:axis=-axis
        elif axis[np.argmax(np.abs(axis))]<0:axis=-axis
        return theta*axis
    return theta/(2*math.sin(theta))*skew


def triangle_frame(points):
    p=_array(points,(3,3),'目标三角面');a=p[1]-p[0];b=p[2]-p[0]
    normal=np.cross(a,b);length=np.linalg.norm(a);area=np.linalg.norm(normal)
    if length<1e-9 or np.linalg.norm(b)<1e-9 or area/(length*np.linalg.norm(b))<1e-6:
        raise ValueError('目标三角面退化，停止而非猜测附着朝向')
    x=a/length;z=normal/area;y=np.cross(z,x)
    return np.column_stack((x,y,z))


def driver_status(plan, review=None, bindings=None):
    prepared=review is not None and bindings is not None
    if prepared:validate_bindings(plan,review,bindings)
    return {'implementation':'explicit_attachment_projection_v1','core_implemented':True,
        'binding_structure_checked':prepared,'reference':'immutable_source_frame',
        'modes':['source_root','bone_socket','surface_rigid','surface_seam'],
        'two_sided_attachment_projection':True,'collision_solver_implemented':False,
        'production_ready':False,'native_model_binding_accepted':False}


class AttachmentSystem:
    def __init__(self,plan,review,bindings):
        validate_bindings(plan,review,bindings)
        geometry=plan['geometry'];inventory=plan['surface_segmentation']['inventory']
        if _sha({k:geometry[k] for k in ('positions_m','triangles','face_material_ids','material_names','skin_weights')})!=inventory['geometry_sha256']:
            raise ValueError('运行时几何与已审核指纹不同')
        self.rest=_array(geometry['positions_m'],(len(geometry['positions_m']),3),'源几何');self.rest.setflags(write=False)
        self.faces=np.array(geometry['triangles'],int);self.faces.setflags(write=False)
        self.rows=copy.deepcopy({r['part_id']:r for r in bindings['attachments']})
        self.vertices={p['id']:np.array(sorted({v for f in p['face_ids'] for v in self.faces[f]}),int) for p in review['parts']}
        self.categories={p['id']:p['category'] for p in review['parts']}
        self.rigid={pid for pid,row in self.rows.items() if row['mode'] in ('bone_socket','surface_rigid')}
        self.rigid_owner={};self.rest_centers={};self.rigid_local={}
        for pid in sorted(self.rigid):
            ids=self.vertices[pid]
            for other,other_ids in self.vertices.items():
                if other!=pid and np.intersect1d(ids,other_ids).size:
                    raise ValueError('硬饰共享其他部件顶点，需明确拆分归属后再驱动')
            self.rest_centers[pid]=self.rest[ids].mean(0)
            self.rigid_local[pid]={int(v):self.rest[v]-self.rest_centers[pid] for v in ids}
            self.rigid_owner.update({int(v):pid for v in ids})
        self.fixed=np.ones(len(self.rest),bool)
        for pid,category in self.categories.items():
            if category not in ('protected_surface','display_overlay','rigid_ornament'):self.fixed[self.vertices[pid]]=False
        for region in plan['regions']:
            if region['role'] in ('body','boots'):self.fixed[region['vertex_ids']]=True
        for row in self.rows.values():
            if row['mode']=='source_root':self.fixed[[a['vertex_id'] for a in row['anchors']]]=True
        self.constraints=[]
        for pid,row in self.rows.items():
            if not row['mode'].startswith('surface_'):continue
            for anchor in row['anchors']:
                ids=self.faces[anchor['target_face_id']]
                local=triangle_frame(self.rest[ids]).T@np.array(anchor['rest_offset_m'])
                self.constraints.append({'child':anchor['vertex_id'],'target':ids.copy(),
                    'barycentric':np.array(anchor['barycentric'],float),'offset_local':local,'part':pid})

    def begin_frame(self,source_positions,vertex_inverse_mass,rigid_inverse_mass=None,rigid_inverse_inertia=None,
                    bone_rest=None,bone_current=None):
        return AttachmentFrame(self,source_positions,vertex_inverse_mass,rigid_inverse_mass or {},
                               rigid_inverse_inertia or {},bone_rest or {},bone_current or {})


class AttachmentFrame:
    def __init__(self,system,source,weights,masses,inertias,bone_rest,bone_current):
        self.system=system;self.source=_array(source,system.rest.shape,'当前源帧');self.source.setflags(write=False)
        self.x=self.source.copy();self.weights=_array(weights,(len(source),),'顶点逆质量')
        if np.any(self.weights<0):raise ValueError('逆质量不能为负')
        self.weights[system.fixed]=0.;self.rigid={};self.reference={};self.mass={};self.inertia={}
        dynamic={pid for pid in system.rigid if system.rows[pid]['mode']=='surface_rigid'}
        if set(masses)!=dynamic or set(inertias)!=dynamic:raise ValueError('必须显式提供各活动硬饰的质量和惯量，不能施力到固定骨座')
        for pid in sorted(system.rigid):
            row=system.rows[pid];ids=system.vertices[pid]
            if row['mode']=='bone_socket':
                bone=row['bone_id']
                if bone not in bone_rest or bone not in bone_current:raise ValueError('缺少原始骨座静态/当前矩阵')
                rest=_matrix(bone_rest[bone]);current=_matrix(bone_current[bone]);relative=current@np.linalg.inv(rest)
                r=_rotation(relative[:3,:3]);t=relative[:3,3];mass=0.;inertia=np.zeros((3,3))
            else:
                anchors=[a['vertex_id'] for a in row['anchors']]
                if np.linalg.matrix_rank(self.source[anchors]-self.source[anchors].mean(0),tol=1e-9)<2:raise ValueError('源帧刚性参考点退化')
                _,r,t=rigid_fit(system.rest[anchors],self.source[anchors]);mass=masses[pid]
                if type(mass) not in (int,float) or not math.isfinite(mass) or mass<0:raise ValueError('硬饰逆质量非法')
                inertia=_array(inertias[pid],(3,3),'硬饰逆惯量')
                if not np.allclose(inertia,inertia.T,atol=1e-12,rtol=0) or np.linalg.eigvalsh(inertia).min()<-1e-12:raise ValueError('逆惯量必须对称半正定')
                if row['limits']['translation_m']==0:mass=0.
                if row['limits']['rotation_rad']==0:inertia=np.zeros((3,3))
            state={'rotation':r.copy(),'center':r@system.rest_centers[pid]+t}
            self.rigid[pid]=state;self.reference[pid]=copy.deepcopy(state);self.mass[pid]=float(mass);self.inertia[pid]=inertia
        self.enforce_limits()

    def point(self,v):
        pid=self.system.rigid_owner.get(int(v))
        if pid is None:return self.x[v]
        p=self.rigid[pid];return p['rotation']@self.system.rigid_local[pid][int(v)]+p['center']

    def positions(self):
        out=self.x.copy()
        for pid in self.rigid:
            for v in self.system.vertices[pid]:out[v]=self.point(v)
        return out

    def residual(self,c):
        target=np.array([self.point(v) for v in c['target']])
        return self.point(c['child'])-(c['barycentric']@target+triangle_frame(target)@c['offset_local'])

    def enforce_limits(self):
        for pid,state in self.rigid.items():
            _rotation(state['rotation']);_array(state['center'],(3,),'硬饰中心')
            row=self.system.rows[pid];ref=self.reference[pid]
            if row['mode']=='bone_socket':
                state['rotation']=ref['rotation'].copy();state['center']=ref['center'].copy();continue
            delta=state['center']-ref['center'];distance=np.linalg.norm(delta);limit=row['limits']['translation_m']
            if distance>limit:state['center']=ref['center']+delta*(limit/distance)
            turn=rotation_vector(state['rotation']@ref['rotation'].T);angle=np.linalg.norm(turn);limit=row['limits']['rotation_rad']
            if angle>limit:state['rotation']=exp_rotation(turn*(limit/angle))@ref['rotation']

    def _dofs(self,c):
        keys=set()
        for v in [c['child'],*c['target']]:
            pid=self.system.rigid_owner.get(int(v))
            if pid is not None:
                if self.mass[pid]>0 or np.any(self.inertia[pid]):keys.add(('rigid',pid))
            elif self.weights[v]>0:keys.add(('vertex',int(v)))
        return sorted(keys,key=str)

    def _weight(self,key):
        kind,idx=key
        if kind=='vertex':return self.weights[idx]*np.eye(3)
        result=np.zeros((6,6));result[:3,:3]=self.mass[idx]*np.eye(3);r=self.rigid[idx]['rotation']
        result[3:,3:]=r@self.inertia[idx]@r.T
        return result

    def _jacobian(self,c,key,eps):
        kind,idx=key;count=3 if kind=='vertex' else 6;jac=np.zeros((3,count))
        saved=self.x[idx].copy() if kind=='vertex' else copy.deepcopy(self.rigid[idx])
        try:
            for axis in range(count):
                values=[]
                for sign in (1,-1):
                    if kind=='vertex':
                        self.x[idx]=saved.copy();self.x[idx,axis]+=sign*eps
                    else:
                        self.rigid[idx]=copy.deepcopy(saved)
                        if axis<3:self.rigid[idx]['center'][axis]+=sign*eps
                        else:
                            vector=np.zeros(3);vector[axis-3]=sign*eps
                            self.rigid[idx]['rotation']=exp_rotation(vector)@saved['rotation']
                    values.append(self.residual(c))
                jac[:,axis]=(values[0]-values[1])/(2*eps)
        finally:
            if kind=='vertex':self.x[idx]=saved
            else:self.rigid[idx]=saved
        return jac

    def project_attachments(self,*,iterations=32,tolerance_m=1e-5,dt=1/30,compliance=0.,jacobian_step=1e-6):
        if type(iterations) is not int or not 1<=iterations<=256:raise ValueError('附着迭代预算必须1至256')
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in (tolerance_m,dt,jacobian_step)) or not math.isfinite(compliance) or compliance<0:
            raise ValueError('附着容差/步长/顺应性非法')
        if not np.isfinite(self.x).all() or not np.array_equal(self.x[self.system.fixed],self.source[self.system.fixed]):raise ValueError('预测器修改了固定源顶点')
        self.enforce_limits();constraints=self.system.constraints;lambdas=[np.zeros(3) for _ in constraints];alpha=compliance/dt**2
        initial=max((float(np.linalg.norm(self.residual(c))) for c in constraints),default=0.)
        used=0
        for step in range(iterations):
            if max((float(np.linalg.norm(self.residual(c))) for c in constraints),default=0.)<=tolerance_m:break
            used=step+1
            for i,c in enumerate(constraints):
                residual=self.residual(c);keys=self._dofs(c)
                blocks=[(key,self._jacobian(c,key,jacobian_step),self._weight(key)) for key in keys]
                effective=alpha*np.eye(3)
                for _,j,w in blocks:effective+=j@w@j.T
                delta_lambda=np.linalg.pinv(effective,rcond=1e-10)@(-residual-alpha*lambdas[i])
                updates=[(key,w@j.T@delta_lambda) for key,j,w in blocks]
                # Bound a local nonlinear step, applying the same factor to
                # both reaction sides; this is not a physical friction model.
                maximum=max((np.linalg.norm(d[3:]) for key,d in updates if key[0]=='rigid'),default=0.)
                scale=min(1.,.25/maximum) if maximum else 1.
                lambdas[i]+=scale*delta_lambda
                for (kind,idx),delta in updates:
                    delta=delta*scale
                    if kind=='vertex':self.x[idx]+=delta
                    else:
                        self.rigid[idx]['center']+=delta[:3]
                        self.rigid[idx]['rotation']=exp_rotation(delta[3:])@self.rigid[idx]['rotation']
                self.enforce_limits()
        residual=max((float(np.linalg.norm(self.residual(c))) for c in constraints),default=0.)
        protected=np.array_equal(self.x[self.system.fixed],self.source[self.system.fixed])
        if not protected or not np.isfinite(self.positions()).all():raise ValueError('附着结果破坏固定点或产生非有限数据')
        return {'attachment_converged':residual<=tolerance_m,'initial_residual_m':initial,
                'maximum_residual_m':residual,'iterations_used':used,'source_vertices_unchanged':protected,
                'collision_checked':False,'physics_accepted':False,
                'reason':None if residual<=tolerance_m else 'attachment_limits_or_iteration_budget_unsatisfied'}


def reject_legacy_assembly():
    raise RuntimeError('旧最终衣料单向饰品装配已停用；新附着驱动须结合已审核模型绑定和联合碰撞求解，不能用旧缓存绕过。')
