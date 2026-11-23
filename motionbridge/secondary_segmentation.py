"""Non-destructive, model-owned surface inventory and explicit review gate.

Connectivity and coincident coordinates are evidence, never permission to weld,
discard faces, infer attachment, or change a physical solver.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math

CATEGORIES = {'protected_surface', 'inner_cloth', 'outer_cloth', 'hair',
              'rigid_ornament', 'flexible_trim', 'display_overlay'}
RELATIONS = {'display_duplicate', 'separate_layers', 'authored_overlap'}


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def build_inventory(plan):
    g = plan['geometry']; triangles = g['triangles']; positions = g['positions_m']
    mids = g['face_material_ids']; names = g['material_names']; skins = g['skin_weights']
    if not triangles or len(mids) != len(triangles) or len(skins) != len(positions):
        raise ValueError('分割输入长度不匹配')
    if any(len(p) != 3 or not all(math.isfinite(x) for x in p) for p in positions):
        raise ValueError('分割顶点坐标必须有限')
    if any(len(t) != 3 or len(set(t)) != 3 or any(type(v) is not int or not 0 <= v < len(positions) for v in t) for t in triangles):
        raise ValueError('分割需要有效的源三角面')
    if any(type(m) is not int or not 0 <= m < len(names) for m in mids):
        raise ValueError('分割面材质索引非法')
    roles = {}; protected = set()
    for r in plan['regions']:
        if r['role'] in ('body', 'boots'): protected.update(r['vertex_ids'])
        for f in r['face_ids']:
            if f in roles or type(f) is not int or not 0 <= f < len(triangles):
                raise ValueError('分割源面重复或越界')
            roles[f] = r['role']
    if set(roles) != set(range(len(triangles))): raise ValueError('分割源面存在遗漏')
    geometry_sha = _sha({k:g[k] for k in ('positions_m','triangles','face_material_ids','material_names','skin_weights')})
    parent = list(range(len(triangles))); edges = {}; vertex_faces = defaultdict(list)
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for f, t in enumerate(triangles):
        for v in t: vertex_faces[v].append(f)
        for a, b in zip(t, t[1:]+t[:1]):
            key = (mids[f], min(a,b), max(a,b))
            if key in edges: parent[find(f)] = find(edges[key])
            else: edges[key] = f
    groups = defaultdict(list)
    for f in range(len(triangles)): groups[find(f)].append(f)
    components = []; owner = {}
    for faces in sorted(groups.values(), key=lambda fs:fs[0]):
        vertices = sorted({v for f in faces for v in triangles[f]}); bone_mass = Counter()
        for v in vertices:
            for b, w in skins[v].items(): bone_mass[b] += w/len(vertices)
        cid = 'part-'+_sha([plan['model_sha256'],faces])[:20]
        for f in faces: owner[f] = cid
        components.append({'id':cid, 'material_id':mids[faces[0]], 'material':names[mids[faces[0]]],
            'source_roles':sorted({roles[f] for f in faces}), 'face_ids':faces, 'vertex_ids':vertices,
            'bounds_m':[[min(positions[v][ax] for v in vertices) for ax in range(3)],
                        [max(positions[v][ax] for v in vertices) for ax in range(3)]],
            'protected_vertex_count':len(protected.intersection(vertices)),
            'dominant_bones':[{'id':int(b),'name':plan['bones'][int(b)]['name'],'mean_weight':w}
                              for b,w in bone_mass.most_common(4)]})
    shared = [{'vertex_id':v,'component_ids':sorted({owner[f] for f in fs}),
               'material_ids':sorted({mids[f] for f in fs}), 'protected':v in protected}
              for v,fs in sorted(vertex_faces.items()) if len({owner[f] for f in fs})>1]
    # Quantization is for review candidates only; no mesh vertices are merged.
    position_keys = [tuple(round(c,6) for c in p) for p in positions]
    coordinate_vertices = defaultdict(list); coordinate_faces = defaultdict(list)
    for v,key in enumerate(position_keys): coordinate_vertices[key].append(v)
    for f,t in enumerate(triangles): coordinate_faces[tuple(sorted(position_keys[v] for v in t))].append(f)
    duplicates = []
    for faces in sorted((fs for fs in coordinate_faces.values() if len(fs)>1), key=lambda fs:fs[0]):
        signatures = [sorted((position_keys[v],tuple(sorted(skins[v].items()))) for v in triangles[f]) for f in faces]
        duplicates.append({'id':'overlap-'+_sha([plan['model_sha256'],faces])[:20],
            'face_ids':faces,'material_ids':sorted({mids[f] for f in faces}),
            'same_vertex_set':len({tuple(sorted(triangles[f])) for f in faces})==1,
            'matching_skin_weights':all(s==signatures[0] for s in signatures),
            'morph_equivalence_verified':False})
    return {'schema_version':1, 'model_sha256':plan['model_sha256'], 'geometry_sha256':geometry_sha,
        'face_count':len(triangles), 'vertex_count':len(positions), 'components':components,
        'shared_vertices':shared, 'protected_boundary_face_ids':[f for f,t in enumerate(triangles) if protected.intersection(t)],
        'coincident_vertex_groups':[vs for vs in coordinate_vertices.values() if len(vs)>1],
        'coincident_face_groups':duplicates, 'coordinate_rounding_decimals':6,
        'topology_changed':False,'simulation_performed':False,
        'limitations':['同坐标/同蒙皮只是复核线索，不证明焊接、刚性、叠层用途或全形态键一致。',
                       '同材质共享边连通片不是语义部件；一个片允许继续按源面细分。']}


def review_gate(inventory, review=None):
    """Validate explicit face ownership; no automatic semantic classification."""
    if review is None:
        return {'accepted':False, 'reasons':['missing_semantic_review'],
                'unclassified_faces':inventory['face_count'],
                'unreviewed_overlap_groups':len(inventory['coincident_face_groups'])}
    if not isinstance(review,dict) or review.get('schema_version') != 1 or any(review.get(k) != inventory[k] for k in ('model_sha256','geometry_sha256')):
        raise ValueError('分类审核指纹或版本不匹配，需重新审核模型')
    owners = {}; ids = set(); faces = set(range(inventory['face_count']))
    protected_faces = {f for c in inventory['components'] if set(c['source_roles']) <= {'body','boots'} for f in c['face_ids']}
    boundary_faces = set(inventory['protected_boundary_face_ids'])
    for part in review.get('parts',[]):
        pid=part.get('id'); category=part.get('category'); selected=part.get('face_ids')
        if not isinstance(pid,str) or not pid.strip() or pid in ids: raise ValueError('分类部件ID缺失或重复')
        ids.add(pid)
        if category not in CATEGORIES or not isinstance(part.get('evidence'),str) or not part['evidence'].strip():
            raise ValueError('分类必须有合法类别和可追溯审核依据')
        if not isinstance(selected,list) or not selected or any(type(f) is not int or f not in faces or f in owners for f in selected) or len(set(selected)) != len(selected):
            raise ValueError('分类源面重复、缺失或越界')
        if category != 'protected_surface' and protected_faces.intersection(selected):
            raise ValueError('受保护表面不能改成动力学部件')
        if category in ('inner_cloth','outer_cloth') and (type(part.get('layer')) is not int or part['layer']<0):
            raise ValueError('衣料分类必须明确层级')
        if category=='rigid_ornament' and boundary_faces.intersection(selected):
            raise ValueError('刚性部件涉及受保护连接边界，必须先显式细分审核')
        for f in selected: owners[f]=pid
    group_ids = {g['id'] for g in inventory['coincident_face_groups']}; reviewed=set()
    for relation in review.get('overlaps',[]):
        gid=relation.get('group_id')
        if gid not in group_ids or gid in reviewed: raise ValueError('重叠关系ID过期或重复')
        if relation.get('kind') not in RELATIONS or not isinstance(relation.get('evidence'),str) or not relation['evidence'].strip():
            raise ValueError('重叠关系必须有类型及审核依据')
        reviewed.add(gid)
    missing=len(faces-set(owners)); overlaps=len(group_ids-reviewed); reasons=[]
    if missing: reasons.append('unclassified_faces')
    if overlaps: reasons.append('unreviewed_overlap_groups')
    return {'accepted':not reasons,'reasons':reasons,'unclassified_faces':missing,
            'unreviewed_overlap_groups':overlaps,'review_sha256':_sha(review),
            'collision_rules_verified':False,'attachment_verified':False,'simulation_accepted':False}


def require_review(plan):
    segmentation=plan.get('surface_segmentation',{})
    if segmentation.get('required_for_surface_dynamics') and not segmentation.get('review_gate',{}).get('accepted'):
        gate=segmentation.get('review_gate',{})
        raise ValueError('模型表面分类未通过：待分类面 %s，待确认重叠组 %s；已在仿真前停止。' %
                         (gate.get('unclassified_faces','未知'),gate.get('unreviewed_overlap_groups','未知')))
    from .secondary_contact_policy import require_contact_policy
    require_contact_policy(plan)
