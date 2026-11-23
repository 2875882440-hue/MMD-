"""Explicit common-boundary hardwear/cloth connections. No nearest-surface follow."""
import numpy as np
from .secondary_source_map import SourceSurfaceMap
from .secondary_segmentation import _sha
from .secondary_attachment_driver import exp_rotation
from .secondary_mount_reference import mount_mode, mount_reference

def compile_mounts(plan,review,manifest,config):
    mapping=SourceSurfaceMap(plan,review,manifest);g=plan['geometry'];parts={p['id']:p for p in review['parts']}
    if config.get('schema_version')!=1 or config.get('model_sha256')!=plan['model_sha256'] or config.get('review_sha256')!=_sha(review):raise ValueError('衣面附着配置指纹无效')
    mounts=[];seen=set();owner={};boundary_lookup={}
    keys=[tuple(round(float(x),6) for x in point) for point in mapping.rest]
    for part in parts.values():
        if part['category'] in ('inner_cloth','outer_cloth','flexible_trim'):
            for f in part['face_ids']:
                for v in g['triangles'][f]:boundary_lookup.setdefault(keys[v],set()).add((part['id'],v))
    for row in config['mounts']:
        pid=row['part_id'];mode=mount_mode(row)
        if pid not in parts or pid in seen or parts[pid]['category']!='rigid_ornament':raise ValueError('硬饰部件无效或重复')
        seen.add(pid);vertices=sorted({v for f in parts[pid]['face_ids'] for v in g['triangles'][f]})
        canonical=sorted(set(mapping.canonical_vertex[vertices].tolist()))
        if set(canonical)&set(mapping.fixed_canonical):raise ValueError('硬饰涉及保护表面')
        if any(v in owner for v in canonical):raise ValueError('硬饰之间共享主点，需单独审核')
        owner.update({v:pid for v in canonical});anchors=[];pairs=set();targets=set()
        if not row.get('evidence'):raise ValueError('缺少实际连接边界依据')
        expected={(v,parent,target) for v in vertices for parent,target in boundary_lookup.get(keys[v],set())}
        supplied={(a['rigid_vertex_id'],a['cloth_part'],a['cloth_vertex_id']) for a in row['anchors']}
        if supplied!=expected:raise ValueError('必须保留完整共点边界及内外衣层，不能选择单层跟随')
        for a in row['anchors']:
            child=a['rigid_vertex_id'];target=a['cloth_vertex_id'];parent=a['cloth_part']
            if parent not in parts or parts[parent]['category'] not in ('inner_cloth','outer_cloth','flexible_trim'):raise ValueError('目标不是声明衣料')
            vs={v for f in parts[parent]['face_ids'] for v in g['triangles'][f]}
            if type(child)!=int or type(target)!=int or child not in vertices or target not in vs:raise ValueError('连接点不属于声明部件')
            if np.linalg.norm(mapping.rest[child]-mapping.rest[target])>2e-6:raise ValueError('只接受真实共点边界，不接受最近邻猜测')
            pair=(int(mapping.canonical_vertex[child]),int(mapping.canonical_vertex[target]));targets.add(parent)
            if pair not in pairs:pairs.add(pair);anchors.append(list(pair))
        points=mapping.rest[[a for a,b in anchors]]
        if len(points)<3 or np.linalg.matrix_rank(points-points.mean(0),tol=1e-8)<2:raise ValueError('连接位置共线或不足，不能确定硬饰朝向')
        mounts.append({'part_id':pid,'vertex_ids':vertices,'canonical_vertex_ids':canonical,'anchors':anchors,'cloth_parts':sorted(targets),
                       'motion_mode':mode,'motion_mode_evidence':row.get('motion_mode_evidence','legacy rest-rigid reference')})
    if not mounts:raise ValueError('缺少明确衣面附着')
    return {'mounts':mounts,'config_sha256':_sha(config),'simulation_ready':False,'collision_checked':False,'full_physics_accepted':False}

def project_mounts(source,mapping,compiled,fixed_vertices,cloth_inverse_mass,rigid_inverse_metrics,predicted_offsets=None,iterations=128,tolerance=2e-6):
    """Whole-part rigid DOFs and independent cloth nodes, jointly projected.

    Returns a separate render array. Unit diagnostic metrics are not SI inertia.
    No mesh, source curve, skin, or shape-key mutation; failure is atomic.
    """
    source=np.array(source,float,copy=True);mapping.check_positions(source);x=source.copy()
    weights=np.array(cloth_inverse_mass,float,copy=True)
    if weights.shape!=(len(x),) or not np.isfinite(weights).all() or np.any(weights<0):raise ValueError('衣料逆质量非法')
    if type(iterations)!=int or not 1<=iterations<=256 or not 0<tolerance<=2e-6:raise ValueError('预算或门限非法')
    roots=np.unique(np.r_[mapping.fixed_vertices,np.asarray(fixed_vertices,int)])
    if np.any(roots<0) or np.any(roots>=len(x)):raise ValueError('固定点越界')
    weights[roots]=0;states={};local={};metrics={};reference={};fits={};reference_audits={}
    pids={m['part_id'] for m in compiled['mounts']}
    if set(rigid_inverse_metrics)!=pids or set(predicted_offsets or {})-pids:raise ValueError('硬饰度量或预测部件不完整')
    for m in compiled['mounts']:
        pid=m['part_id'];ids=m['canonical_vertex_ids']
        rot,center,points,world,reference_audit=mount_reference(source,mapping,m)
        states[pid]=[rot,center];local[pid]=points;reference[pid]=[s.copy() for s in states[pid]]
        fits[pid]=reference_audit['source_rigid_fit_error_m'];reference_audits[pid]=reference_audit
        metric=np.asarray(rigid_inverse_metrics[pid],float)
        if metric.shape!=(2,) or not np.isfinite(metric).all() or np.any(metric<0):raise ValueError('需要非负平移/旋转逆度量')
        metrics[pid]=metric
        offset=np.asarray((predicted_offsets or {}).get(pid,[0,0,0]),float)
        if offset.shape!=(3,) or not np.isfinite(offset).all():raise ValueError('预测位移非法')
        if metric[0]==0 and np.any(offset):raise ValueError('固定硬饰不能施加预测位移')
        states[pid][1]+=offset
    def point(pid,v):return states[pid][0]@local[pid][v]+states[pid][1]
    def error():return max(float(np.linalg.norm(point(m['part_id'],a)-x[b])) for m in compiled['mounts'] for a,b in m['anchors'])
    used=0
    for step in range(iterations):
        if error()<=tolerance:break
        used=step+1
        for m in compiled['mounts']:
            pid=m['part_id'];mass,inertia=metrics[pid]
            for a,b in m['anchors']:
                arm=states[pid][0]@local[pid][a];dx,dy,dz=arm
                j=np.array([[0,dz,-dy],[-dz,0,dx],[dy,-dx,0]])
                residual=arm+states[pid][1]-x[b]
                effective=(mass+weights[b])*np.eye(3)+inertia*j@j.T
                lam=-np.linalg.pinv(effective,rcond=1e-12)@residual
                turn=inertia*j.T@lam;factor=min(1.,.25/max(np.linalg.norm(turn),1e-30))
                states[pid][1]+=factor*mass*lam;states[pid][0]=exp_rotation(factor*turn)@states[pid][0]
                x[b]-=factor*weights[b]*lam
    residual=error()
    if residual>tolerance:raise ValueError('衣面双向附着未收敛，拒绝候选且不修改输入')
    # Reconcile virtual cloth/rigid nodes to the original single render index.
    delta=np.zeros_like(x);canonical=np.unique(mapping.canonical_vertex);delta[canonical]=x[canonical]-source[canonical]
    for m in compiled['mounts']:
        for v in m['canonical_vertex_ids']:delta[v]=point(m['part_id'],v)-source[v]
    if np.any(delta[mapping.canonical_vertex[roots]]!=0):raise ValueError('附着结果触及固定源根')
    out=mapping.expand_delta(source,delta)
    rendered=max(float(np.linalg.norm(out[a]-out[b])) for m in compiled['mounts'] for a,b in m['anchors'])
    if rendered>tolerance:raise ValueError('源面回传后连接不一致，拒绝输出')
    return out,{'passed':True,'iterations':used,'maximum_residual_m':residual,'rendered_residual_m':rendered,
        'cloth_response_m':float(np.max(np.linalg.norm(x-source,axis=1))),
        'rigid_response_m':{pid:float(np.linalg.norm(states[pid][1]-reference[pid][1])) for pid in states},
        'source_rigid_fit_error_m':fits,'mount_references':reference_audits,'collision_checked':False,'full_physics_accepted':False}
