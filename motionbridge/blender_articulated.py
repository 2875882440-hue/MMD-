"""Two-sided native joint-island projection; no scene/bone mutation or contacts."""
import numpy as np
from mathutils import Matrix,Euler
from .secondary_attachment_driver import exp_rotation

def project_island(compiled, predicted, inverse_metrics, max_iterations=96, tolerance=2e-6):
    ids=[b['id'] for b in compiled['bodies']];dynamic=compiled['dynamic_rigid_ids']
    if set(predicted)!=set(ids) or set(inverse_metrics)!=set(dynamic):raise ValueError('必须明确全岛状态和动态度量')
    if type(max_iterations)!=int or not 1<=max_iterations<=256 or not 0<tolerance<=2e-6:raise ValueError('预算或门限非法')
    states={i:np.array(predicted[i],float,copy=True) for i in ids}
    for a in states.values():
        if a.shape!=(4,4) or not np.isfinite(a).all() or not np.allclose(a[3],[0,0,0,1],atol=1e-8) or not np.allclose(a[:3,:3].T@a[:3,:3],np.eye(3),atol=2e-6) or np.linalg.det(a[:3,:3])<0:raise ValueError('刚体状态非法')
    centers={b['id']:np.array(b['blender']['position_m']) for b in compiled['bodies']}
    weights=[]
    for i in dynamic:
        a=np.array(inverse_metrics[i],float)
        if a.shape!=(6,) or not np.isfinite(a).all() or np.any(a<0):raise ValueError('投影逆度量须为非负6维向量')
        weights.extend(a)
    weights=np.array(weights);frames=[]
    for j in compiled['joints']:
        p=j['blender'];m=np.eye(4);m[:3,:3]=np.array(Euler(p['rotation_rad'],p['rotation_order']).to_matrix());m[:3,3]=p['position_m'];frames.append(m)
    def residual(s):
        rows=[];raw=[]
        for j,frame in zip(compiled['joints'],frames):
            a,b=j['rigid_ids'];p=j['blender'];relative=np.linalg.inv(s[a]@frame)@s[b]@frame
            angles=np.array(Matrix(relative[:3,:3].tolist()).to_euler(p['rotation_order']))
            linear=relative[:3,3]-np.clip(relative[:3,3],p['linear_min_m'],p['linear_max_m'])
            angular=angles-np.clip(angles,p['angular_min_rad'],p['angular_max_rad'])
            rows.extend(linear);rows.extend(angular*.05);raw.append((float(np.max(np.abs(linear))),float(np.max(np.abs(angular)))))
        return np.array(rows),raw
    def perturb(s,i,delta):
        out={k:v.copy() for k,v in s.items()};a=out[i];center=a[:3,:3]@centers[i]+a[:3,3]
        a[:3,:3]=exp_rotation(delta[3:])@a[:3,:3];a[:3,3]=center+delta[:3]-a[:3,:3]@centers[i]
        return out
    for iteration in range(max_iterations+1):
        values,raw=residual(states)
        if max(max(v) for v in raw)<=tolerance:
            return states,{'passed':True,'iterations':iteration,'max_linear_error_m':max(v[0] for v in raw),'max_angular_error_rad':max(v[1] for v in raw),'contacts_checked':False}
        if iteration==max_iterations:break
        jac=np.zeros((len(values),len(weights)));h=1e-4
        for k,i in enumerate(dynamic):
            for axis in range(6):
                if weights[k*6+axis]==0:continue
                delta=np.zeros(6);delta[axis]=h
                jac[:,k*6+axis]=(residual(perturb(states,i,delta))[0]-residual(perturb(states,i,-delta))[0])/(2*h)
        step=-weights*(jac.T@np.linalg.lstsq((jac*weights)@jac.T+np.eye(len(values))*1e-12,values,rcond=1e-12)[0])
        norm=max((np.linalg.norm(step[k*6:k*6+3])/.02 for k in range(len(dynamic))),default=0)
        norm=max(norm,max((np.linalg.norm(step[k*6+3:k*6+6])/.25 for k in range(len(dynamic))),default=0),1)
        step/=norm;accepted=False
        for scale in (1,.5,.25,.125):
            candidate=states
            for k,i in enumerate(dynamic):candidate=perturb(candidate,i,step[k*6:k*6+6]*scale)
            if np.linalg.norm(residual(candidate)[0])<np.linalg.norm(values):states=candidate;accepted=True;break
        if not accepted:break
    raise ValueError('原生联合关节未收敛；输入保持，禁止删除闭环或固定动态父级')
