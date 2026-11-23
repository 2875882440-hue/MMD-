"""Declarative contact/attachment contract. This module is NOT a solver.

Asset-independent rules may be defined before semantic review. Concrete source
face/anchor bindings remain separate and fail closed. No mesh/action mutation.
"""
from itertools import combinations, combinations_with_replacement
import math

from .secondary_segmentation import CATEGORIES, _sha

POLICY = 'model_contacts_v1'
MODES = {
    'protected_surface': set(), 'display_overlay': set(),
    'inner_cloth': {'source_root', 'surface_seam'},
    'outer_cloth': {'source_root', 'surface_seam'},
    'hair': {'source_root'},
    'flexible_trim': {'source_root', 'surface_seam'},
    'rigid_ornament': {'bone_socket', 'surface_rigid'},
}


def contact_rule(first, second):
    """Symmetric response specification; motion is restricted again by anchors."""
    if first not in CATEGORIES | {'ground'} or second not in CATEGORIES | {'ground'}:
        raise ValueError('未知碰撞类别，禁止默认忽略')
    pair = sorted((first, second))
    if 'display_overlay' in pair:
        response = 'canonical_surface_only_after_verified_mapping'
        movable = []
    else:
        movable = sorted(set(pair) - {'ground', 'protected_surface'})
        response = ('report_fixed_conflict' if not movable else
                    'dynamic_side_only' if any(c in pair for c in ('ground', 'protected_surface')) else
                    'joint_response_in_free_dofs')
    return {'categories':pair, 'response':response, 'potentially_movable_categories':movable,
            'respect_attachment_dofs':True, 'rigid_vertices_independently_movable':False,
            'self_contact':first == second and first not in ('protected_surface','ground','display_overlay'),
            'raw_audit_excluded':False}


def policy_definition():
    categories = sorted(CATEGORIES | {'ground'})
    return {
        'id':POLICY, 'schema_version':1,
        'pair_rules':[contact_rule(a,b) for a,b in combinations_with_replacement(categories,2)],
        'primary_motion':'immutable_bones_actions_skin_morphs_and_root',
        'cloth_layers':'explicit_local_order_not_material_index_or_global_height',
        'attachment_reference':'immutable_input_geometry_and_explicit_anchors',
        'surface_attachment_response':'same_joint_solve_as_contacts_never_postsolve_one_way_copy',
        'rigid_response':'whole_part_translation_rotation_only_with_declared_limits',
        'ground':'fixed_Z0_existing_clearance_no_root_lift',
        'seams':'explicit_face_pairs_bounded_depth_and_slip_never_whole_category_ignore',
        'initial_overlap':'report_before_and_after_never_silently_erase',
        'unresolvable_fixed_constraints':'reject_candidate_do_not_move_primary_motion',
        'continuous_contact':'swept_motion_required_including_rotation_not_just_endpoints',
        'solve_order':['source_pose_and_pins','predict_secondary_motion',
                       'joint_attachment_shape_and_all_contacts','recheck_after_last_correction',
                       'unchanged_prewarm_and_before_relative_acceptance'],
        'acceptance':{'baseline':'captured_input_before', 'accessory_channels':'each_nonworsening_body_cloth_hair',
                      'body_cloth':'strictly_decrease','preserve_existing_other_gates':True,
                      'automatic_parameter_retry':False},
        'solver_implemented':False,
    }


def _vector(values, count, label):
    if not isinstance(values,list) or len(values)!=count or any(type(x) not in (int,float) or not math.isfinite(x) for x in values):
        raise ValueError(label+'必须是有限数值向量')
    return values


def _noncollinear(points):
    origin=points[0]
    vectors=[[p[k]-origin[k] for k in range(3)] for p in points[1:]]
    for a,b in combinations(vectors,2):
        cross=[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
        if sum(x*x for x in cross)>1e-20:return True
    return False


def validate_bindings(plan, review, bindings):
    """Checks structure/references only; cannot certify authored semantics or physics."""
    inv=plan['surface_segmentation']['inventory'];gate=plan['surface_segmentation']['review_gate']
    if gate.get('accepted') is not True:raise ValueError('分类审核未通过，不能确认模型碰撞附着绑定')
    if gate.get('review_sha256')!=_sha(review):raise ValueError('分类审核内容与已编译指纹不一致')
    if not isinstance(bindings,dict) or set(bindings)!={'schema_version','model_sha256','geometry_sha256','review_sha256','attachments','aliases','seams'}:
        raise ValueError('附着配置字段缺失或存在未知项')
    if bindings['schema_version']!=1 or bindings['model_sha256']!=plan['model_sha256'] or bindings['geometry_sha256']!=inv['geometry_sha256'] or bindings['review_sha256']!=_sha(review):
        raise ValueError('附着配置指纹过期，必须按当前模型和分类重新绑定')
    if any(not isinstance(bindings[k],list) for k in ('attachments','aliases','seams')):raise ValueError('附着/映射/接缝必须是列表')
    parts={p['id']:p for p in review['parts']};geo=plan['geometry'];tri=geo['triangles'];pos=geo['positions_m']
    face_owner={f:p['id'] for p in parts.values() for f in p['face_ids']}
    vertices={pid:{v for f in p['face_ids'] for v in tri[f]} for pid,p in parts.items()}
    physical={pid for pid,p in parts.items() if p['category']!='display_overlay'}
    dynamic={pid for pid in physical if parts[pid]['category']!='protected_surface'}
    protected_vertices={v for region in plan['regions'] if region['role'] in ('body','boots') for v in region['vertex_ids']}
    bound={};dependencies={};anchored_faces={}
    for row in bindings['attachments']:
        if not isinstance(row,dict) or set(row)-{'part_id','mode','anchors','bone_id','target_part','limits','evidence'}:
            raise ValueError('未知附着字段；不接受隐式最近邻或最终衣料跟随')
        pid=row.get('part_id');mode=row.get('mode')
        if pid not in dynamic or pid in bound or mode not in MODES[parts[pid]['category']]:raise ValueError('附着部件/模式无效或重复，受保护表面不能被驱动')
        if not isinstance(row.get('evidence'),str) or not row['evidence'].strip():raise ValueError('附着缺少审核依据')
        limits=row.get('limits',{})
        if not isinstance(limits,dict) or set(limits)!={'translation_m','rotation_rad'}:raise ValueError('附着必须显式声明位移/角度限位')
        _vector([limits['translation_m'],limits['rotation_rad']],2,'限位')
        if limits['translation_m']<0 or not 0<=limits['rotation_rad']<=math.pi:raise ValueError('限位不能为负或超过pi')
        if mode in ('source_root','bone_socket') and any(limits.values()):raise ValueError('固定源根/骨座不能借碰撞移动')
        anchors=row.get('anchors');child=[];target=[];target_faces=set()
        minimum=3 if mode in ('bone_socket','surface_rigid') else 2 if mode=='surface_seam' else 1
        if not isinstance(anchors,list) or len(anchors)<minimum:raise ValueError('显式附着点不足')
        if mode=='bone_socket':
            if type(row.get('bone_id')) is not int or row['bone_id'] not in plan['protected_bone_ids']:raise ValueError('骨座必须引用已保护的源骨骼')
        elif 'bone_id' in row:raise ValueError('非骨座模式不能声明骨骼')
        if mode.startswith('surface_'):
            parent=row.get('target_part')
            if parent not in physical or parent==pid:raise ValueError('附着目标必须是另一真实物理表面')
            dependencies[pid]=parent
        elif 'target_part' in row:raise ValueError('源根/骨座不能声明衣料目标')
        seen=set()
        for anchor in anchors:
            expected={'vertex_id','target_face_id','barycentric','rest_offset_m'} if mode.startswith('surface_') else {'vertex_id'}
            if not isinstance(anchor,dict) or set(anchor)!=expected:raise ValueError('附着点字段必须明确')
            v=anchor['vertex_id']
            if type(v) is not int or v not in vertices[pid] or v in seen:raise ValueError('附着点越界、不属于部件或重复')
            if v in protected_vertices and mode!='source_root':raise ValueError('共享受保护顶点只能保持源位置，不能附着到运动衣料')
            seen.add(v);child.append(pos[v])
            if mode.startswith('surface_'):
                f=anchor['target_face_id']
                if type(f) is not int or face_owner.get(f)!=parent:raise ValueError('附着目标面不属于声明部件')
                bary=_vector(anchor['barycentric'],3,'重心坐标');offset=_vector(anchor['rest_offset_m'],3,'静态偏移')
                if any(b<0 or b>1 for b in bary) or abs(sum(bary)-1)>1e-8:raise ValueError('非法重心坐标')
                point=[sum(bary[j]*pos[tri[f][j]][k] for j in range(3)) for k in range(3)]
                if max(abs(pos[v][k]-point[k]-offset[k]) for k in range(3))>1e-6:raise ValueError('附着静态偏移与源模型不符，禁止吸附跳位')
                target.append(point);target_faces.add(f)
        if mode in ('bone_socket','surface_rigid') and not _noncollinear(child):raise ValueError('刚性附着点共线，无法确定稳定朝向')
        if mode=='surface_rigid' and not _noncollinear(target):raise ValueError('目标附着点共线，无法确定稳定朝向')
        bound[pid]=row;anchored_faces[pid]=target_faces
    if set(bound)!=dynamic:raise ValueError('动态部件存在未声明的根部/附着')
    for pid in dependencies:
        visited=set();current=pid
        while current in dependencies:
            if current in visited:raise ValueError('附着归属成环且没有源锚点；接触闭环需联合求解，不能充当跟随链')
            visited.add(current);current=dependencies[current]
    groups={g['id']:g for g in inv['coincident_face_groups']}
    relations={r['group_id']:r['kind'] for r in review.get('overlaps',[])}
    alias_faces=set();mapped_groups={}
    for row in bindings['aliases']:
        if not isinstance(row,dict) or set(row)!={'group_id','display_face','canonical_face','evidence'}:raise ValueError('显示映射字段错误')
        group=groups.get(row['group_id']);display=row['display_face'];canonical=row['canonical_face']
        if not group or relations.get(row['group_id'])!='display_duplicate':raise ValueError('显示映射必须来自已声明的重复面关系')
        if type(display) is not int or type(canonical) is not int or display==canonical or display not in group['face_ids'] or canonical not in group['face_ids'] or display in alias_faces:
            raise ValueError('显示映射面编号错误/重复')
        if not group['matching_skin_weights']:raise ValueError('不同蒙皮重合面禁止合并为同一物理表面')
        if parts[face_owner[canonical]]['category']=='display_overlay':raise ValueError('物理主面不能又是显示副本')
        if not isinstance(row['evidence'],str) or not row['evidence'].strip():raise ValueError('显示映射缺少依据')
        alias_faces.add(display);mapped_groups.setdefault(row['group_id'],[]).append((display,canonical))
    for gid,kind in relations.items():
        if kind=='display_duplicate':
            rows=mapped_groups.get(gid,[])
            if len(rows)!=len(groups[gid]['face_ids'])-1 or len({c for _,c in rows})!=1:raise ValueError('重复面必须有唯一主面及完整副本映射')
    display_faces={f for p in parts.values() if p['category']=='display_overlay' for f in p['face_ids']}
    if not display_faces<=alias_faces:raise ValueError('显示副本存在未映射面')
    seam_keys=set()
    for row in bindings['seams']:
        if not isinstance(row,dict) or set(row)!={'child_part','target_part','face_pairs','max_depth_m','max_slip_m','evidence'}:raise ValueError('接缝例外字段错误')
        child=row['child_part'];parent=row['target_part']
        if dependencies.get(child)!=parent:raise ValueError('接缝例外必须对应显式表面附着')
        for k in ('max_depth_m','max_slip_m'):
            x=row[k]
            if type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=.002:raise ValueError('接缝例外必须限定在0至2毫米，不允许整类忽略')
        if not isinstance(row['evidence'],str) or not row['evidence'].strip() or not isinstance(row['face_pairs'],list) or not row['face_pairs']:raise ValueError('接缝例外缺少局部面和依据')
        anchor_vertices={a['vertex_id'] for a in bound[child]['anchors']}
        for pair in row['face_pairs']:
            if not isinstance(pair,list) or len(pair)!=2 or any(type(f) is not int for f in pair):raise ValueError('接缝必须逐对指定源面')
            a,b=pair
            if face_owner.get(a)!=child or face_owner.get(b)!=parent or not anchor_vertices.intersection(tri[a]) or b not in anchored_faces[child]:raise ValueError('接缝例外超出实际附着邻域')
            key=tuple(sorted(pair))
            if key in seam_keys:raise ValueError('接缝面例外重复')
            seam_keys.add(key)
    return {'structurally_valid':True,'attachments':len(bound),'aliases':len(alias_faces),
            'seam_face_pairs':len(seam_keys),'bindings_sha256':_sha(bindings),
            'asset_mapping_verified':False,'requires_native_geometry_and_motion_proof':True}


def compile_contact_policy(plan, profile):
    config=profile['secondary_contact_rules']
    if not isinstance(config,dict) or set(config)!={'schema_version','policy','bindings'} or config['schema_version']!=1 or config['policy']!=POLICY:
        raise ValueError('不支持的碰撞附着规则配置，禁止静默降级')
    result={'definition':policy_definition(),'definition_sha256':_sha(policy_definition()),
            'rules_defined':True,'asset_bindings':None,'blockers':[],
            'solver_implemented':False,'simulation_ready':False}
    segmentation=plan.get('surface_segmentation',{})
    if segmentation.get('review_gate',{}).get('accepted') is not True:
        result['blockers'].append('semantic_segmentation_not_accepted')
    if config['bindings'] is None:result['blockers'].append('model_attachments_and_surface_mapping_missing')
    elif not result['blockers']:
        review=profile['surface_segmentation']['review']
        result['asset_bindings']=validate_bindings(plan,review,config['bindings'])
        result['blockers'].append('native_geometry_and_motion_proof_pending')
    else:result['blockers'].append('bindings_not_checked_without_segmentation')
    from .secondary_attachment_driver import driver_status
    review = profile.get('surface_segmentation', {}).get('review')
    result['attachment_driver'] = driver_status(plan, review, config['bindings']) if result['asset_bindings'] is not None else driver_status(plan)
    from .secondary_joint_contacts import joint_solver_status
    result['joint_solver'] = joint_solver_status()
    from .secondary_prewarm import prewarm_status
    result['prewarm'] = prewarm_status()
    result['blockers'].append('production_joint_solver_and_native_mapping_not_accepted')
    return result


def require_contact_policy(plan):
    if 'secondary_contact_rules' in plan:
        # Declarative metadata must never unlock the legacy one-way follower.
        raise ValueError('碰撞和附着规则已定义，但模型绑定/联合求解尚未验收；禁止回退到旧饰品跟随算法。')
