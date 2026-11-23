"""Real source-root, explicit stitch and authored kinematic joint compilation.

Partial connection stage only. No contact exclusions or physics enablement.
"""
import copy
import numpy as np
from .secondary_segmentation import _sha
from .secondary_source_map import SourceSurfaceMap


def compile_connections(plan, review, manifest, config):
    mapping=SourceSurfaceMap(plan,review,manifest)
    if config.get('schema_version')!=2 or config.get('model_sha256')!=plan['model_sha256'] or config.get('review_sha256')!=_sha(review):
        raise ValueError('连接配置模型或分类指纹过期')
    parts={p['id']:p for p in review['parts']};g=plan['geometry'];protected=set(plan['protected_bone_ids'])
    roots=set(mapping.fixed_vertices.tolist())
    cloth_roots=set();root_parts=set()
    from collections import Counter
    keys=[tuple(round(float(c),6) for c in point) for point in mapping.rest]
    for row in config.get('roots',[]):
        pid=row.get('part_id');ids=row.get('vertex_ids')
        if pid not in parts or pid in root_parts or parts[pid]['category'] not in ('inner_cloth','outer_cloth','flexible_trim','hair'):
            raise ValueError('固定连接部件无效或重复')
        root_parts.add(pid);part=parts[pid];edges=Counter();vertices=set()
        for f in part['face_ids']:
            tri=g['triangles'][f];vertices.update(tri)
            for a,b in zip(tri,tri[1:]+tri[:1]):
                if keys[a]!=keys[b]:edges[tuple(sorted((keys[a],keys[b])))]+=1
        boundary={v for edge,n in edges.items() if n==1 for v in edge}
        if not isinstance(ids,list) or len(ids)<2 or len(set(ids))!=len(ids) or not row.get('evidence'):
            raise ValueError('固定段必须明确顶点和复核依据')
        if any(type(v)!=int or v not in vertices or keys[v] not in boundary for v in ids):
            raise ValueError('固定点不属于指定部件真实边界')
        cloth_roots.update(ids)
    if not root_parts:raise ValueError('缺少显式衣根固定段')
    roots.update(cloth_roots)
    root_canonical=set(mapping.canonical_vertex[sorted(roots)].tolist())
    roots=sorted(i for i,c in enumerate(mapping.canonical_vertex) if c in root_canonical)
    groups={x['id']:x for x in plan['surface_segmentation']['inventory']['coincident_face_groups']}
    joints=[];seen=set()
    for row in config['rigid_joints']:
        pid=row['part_id'];jid=row['joint_id']
        if pid in seen or pid not in parts or parts[pid]['category']!='rigid_ornament':raise ValueError('硬饰连接归属错误')
        seen.add(pid);joint=plan['joints'][jid];a,b=joint['rigid_ids'];parent,child=plan['rigid_bodies'][a],plan['rigid_bodies'][b]
        if parent['native']['mode']!=0 or parent['bone_id'] not in protected:raise ValueError('动态父级不能降级为固定连接')
        vs=sorted({v for f in parts[pid]['face_ids'] for v in g['triangles'][f]})
        if child['native']['mode'] not in (1,2) or any(g['skin_weights'][v]!={str(child['bone_id']):1.0} for v in vs):raise ValueError('硬饰并非原生单骨绑定')
        if set(vs)&set(roots):raise ValueError('硬饰与保护根冲突')
        params=joint['blender']
        if any(params['linear_spring_import_value']+params['angular_spring_import_value']):raise ValueError('原生弹簧尚未适配，禁止丢弃')
        joints.append({'part_id':pid,'joint_id':jid,'parent_bone_id':parent['bone_id'],'child_bone_id':child['bone_id'],
                       'vertex_ids':vs,'parameters':copy.deepcopy(params)})
    stitches=[];used=set()
    for row in config['stitches']:
        group=groups.get(row['group_id'])
        if not group or group['matching_skin_weights'] or row['group_id'] in used:raise ValueError('局部缝合须指定未合并的不同蒙皮组')
        used.add(row['group_id']);a,b=group['face_ids'];pairs=row['vertex_pairs']
        if len(pairs)!=3 or sorted(x[0] for x in pairs)!=sorted(g['triangles'][a]) or sorted(x[1] for x in pairs)!=sorted(g['triangles'][b]):raise ValueError('缝合点必须对应指定两源面')
        for u,v in pairs:
            if np.linalg.norm(mapping.rest[u]-mapping.rest[v])>2e-6:raise ValueError('缝合点不在原连接位置')
            ca,cb=int(mapping.canonical_vertex[u]),int(mapping.canonical_vertex[v])
            if ca!=cb:stitches.append([ca,cb])
    stitches=sorted({tuple(sorted(pair)) for pair in stitches})
    return {'schema_version':2,'explicit_root_parts':sorted(root_parts),'source_root_vertex_ids':roots,'cloth_root_vertex_count':len(cloth_roots),
        'rigid_joints':joints,'stitch_vertex_pairs':[list(p) for p in stitches],
        'stitch_group_count':len(used),'config_sha256':_sha(config),
        'complete':False,'contact_exclusions_added':False,'simulation_ready':False,
        'remaining':['dynamic_parent_and_chain_joints','cloth_mounted_hardware','joint_contact_integration']}


def project_source_stitches(source, predicted, inverse_mass, connections, mapping, iterations=32):
    """Two-sided equal-position stitches, fixed source roots; failure is atomic."""
    source=np.array(source,float,copy=True);x=np.array(predicted,float,copy=True);w=np.array(inverse_mass,float,copy=True)
    if source.shape!=mapping.rest.shape or x.shape!=source.shape or w.shape!=(len(x),) or not all(np.isfinite(a).all() for a in (source,x,w)) or np.any(w<0):raise ValueError('连接输入非法')
    if type(iterations)!=int or not 1<=iterations<=256:raise ValueError('连接迭代预算非法')
    mapping.check_positions(source)
    roots=np.array(connections['source_root_vertex_ids'],int);w[roots]=0;x[roots]=source[roots]
    for _ in range(iterations):
        for a,b in connections['stitch_vertex_pairs']:
            d=x[b]-x[a];total=w[a]+w[b]
            if total<=0:
                if np.linalg.norm(d)>2e-6:raise ValueError('固定源根与局部缝合冲突')
                continue
            x[a]+=d*w[a]/total;x[b]-=d*w[b]/total
        x[roots]=source[roots]
    error=max((float(np.linalg.norm(x[a]-x[b])) for a,b in connections['stitch_vertex_pairs']),default=0.)
    if error>2e-6:raise ValueError('缝合未收敛，拒绝候选')
    delta=np.zeros_like(x);canonical=np.unique(mapping.canonical_vertex)
    delta[canonical]=x[canonical]-source[canonical]
    return mapping.expand_delta(source,delta)
