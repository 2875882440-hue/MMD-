"""Read-only audit of saved target animation; no motion-accuracy claims."""
import bpy
import sys
import json
import math
import hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

folder = Path(sys.argv[sys.argv.index('--') + 1]).resolve()
report = json.loads((folder / 'retarget_report.json').read_text(encoding='utf-8'))
bpy.ops.wm.open_mainfile(filepath=report['output'], load_ui=False)
scene = bpy.context.scene
rig = bpy.data.objects['Nahida_arm']
names = list(report['mapped_bones'])
maximum_delta = {'degrees': 0., 'bone': None, 'frame': None}
previous = {}
bound_records = []
finite = True
world_mode = report.get('root_mode') == 'estimated_world_translation'
root_error = 0.
floor_check=report.get('target_mesh_floor_collision',{}).get('enabled',False)
foot_heights=[]
body_check=report.get('target_body_collision',{}).get('enabled',False)
head_check=report.get('target_head_collision',{}).get('enabled',False)
accessory_check=report.get('target_accessory_collision',{}).get('enabled',False)
body_check |= accessory_check
head_check |= accessory_check
head_matrices=[]
body_heads=[]
if floor_check:
    from motionbridge.blender_feet import foot_vertex_groups,evaluated_foot_minimum
    foot_samples=foot_vertex_groups(scene,rig)
if world_mode and len(report.get('expected_root_positions',[])) != scene.frame_end-scene.frame_start+1:
    raise ValueError('World target audit requires expected root positions for every frame')
for frame in range(scene.frame_start, scene.frame_end + 1):
    scene.frame_set(frame)
    if body_check or head_check:body_heads.append({n:list(rig.pose.bones[n].head) for n in ['腰','左腕','右腕','左ひじ','右ひじ','左手首','右手首']})
    if head_check:head_matrices.append([list(row) for row in rig.pose.bones['頭'].matrix])
    if floor_check:foot_heights.append(evaluated_foot_minimum(foot_samples,bpy.context.evaluated_depsgraph_get()))
    if world_mode:
        expected = report['expected_root_positions'][frame-scene.frame_start]
        actual = rig.pose.bones['全ての親'].matrix.translation
        root_error = max(root_error, math.sqrt(sum((actual[i]-expected[i])**2 for i in range(3))))
    for name in names:
        pb = rig.pose.bones[name]
        q = pb.matrix.to_quaternion()
        finite &= all(math.isfinite(c) for row in pb.matrix for c in row)
        if name in previous:
            angle = math.degrees(2 * math.acos(min(1., abs(q.dot(previous[name])))))
            if angle > maximum_delta['degrees']:
                maximum_delta = {'degrees': angle, 'bone': name, 'frame': frame}
        previous[name] = q
    if frame % 30 == 1:
        points = []
        for obj in scene.objects:
            if obj.type != 'MESH' or not obj.name.startswith('Nahida') or 'contour' in obj.name or obj.hide_render:
                continue
            ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
            mesh = ev.to_mesh()
            coords = [obj.matrix_world @ v.co for v in mesh.vertices]
            finite &= all(math.isfinite(c) for p in coords for c in p)
            points.extend(coords)
            ev.to_mesh_clear()
        bound_records.append({'frame': frame, 'min': [min(p[i] for p in points) for i in range(3)],
                              'max': [max(p[i] for p in points) for i in range(3)]})
images = [{'name': i.name, 'packed': bool(i.packed_file), 'path': i.filepath}
          for i in bpy.data.images if i.users and i.source == 'FILE']
original_hash = hashlib.sha256(Path(report['original_project']).read_bytes()).hexdigest()
audit = {'frames': scene.frame_end - scene.frame_start + 1, 'fps': scene.render.fps / scene.render.fps_base,
         'all_frame_finite_bone_transforms': finite, 'maximum_adjacent_world_rotation_change': maximum_delta,
         'mesh_bounds_samples': bound_records, 'images': images, 'all_used_file_images_packed': all(i['packed'] for i in images),
         'active_mesh_cache_modifiers': [[o.name, m.name] for o in scene.objects for m in o.modifiers if m.type == 'MESH_CACHE' and m.show_render],
         'original_unchanged': original_hash == report['original_sha256_before'],
         'active_action': rig.animation_data.action.name, 'mapped_bone_count': len(names),
         'note': 'Structural and continuity diagnostics only. Not capture accuracy or collision-free certification.'}
cleanup = report.get('temporal_cleanup')
audit['continuity_limit_degrees_per_frame'] = max(cleanup['torso_head_speed_limit_deg_s'], cleanup['limb_speed_limit_deg_s']) / audit['fps'] if cleanup else None
audit['continuity_passed'] = maximum_delta['degrees'] <= audit['continuity_limit_degrees_per_frame'] + .05 if cleanup else None
audit['root_mode'] = report.get('root_mode','in_place')
audit['world_root_max_error_m'] = root_error if world_mode else None
audit['continuity_limit_applicable'] = bool(cleanup)
audit['passed'] = finite and audit['frames'] == report['frames'] and audit['fps'] == report['fps'] and audit['original_unchanged'] and audit['all_used_file_images_packed'] and not audit['active_mesh_cache_modifiers'] and (root_error < 1e-5 if world_mode else audit['continuity_passed'] is True)
if floor_check:
    half_heights=[]
    for frame in range(scene.frame_start,scene.frame_end):
        scene.frame_set(frame,subframe=.5)
        half_heights.append(evaluated_foot_minimum(foot_samples,bpy.context.evaluated_depsgraph_get()))
    all_heights=foot_heights+half_heights
    audit['foot_floor_collision']={'enabled':True,'evaluated_frames':len(foot_heights),'min_z_m':min(foot_heights),
        'half_frame_samples':len(half_heights),'half_frame_min_z_m':min(half_heights) if half_heights else None,
        'below_floor_frames':sum(z<-.00001 for z in foot_heights),'passed':min(all_heights)>=-.00001,
        'note':'Evaluated weighted shoe vertices at all frames and midpoints; not arbitrary subframes or full-body collision certification.'}
    audit['passed'] &= audit['foot_floor_collision']['passed']
if body_check or head_check:
    import numpy as np
    from motionbridge.body_collision import torso_space,arm_clearance
    def track(n):return np.array([f[n] for f in body_heads])
    def rest(n):return np.array(rig.data.bones[n].head_local)
    center,basis,radii,radius=torso_space(track('腰'),track('左腕'),track('右腕'),rest('腰'),rest('左腕'),rest('右腕'))
    diagnostics={}
    for side,label in [('左','left'),('右','right')]:
        points=np.stack([track(side+n) for n in ['腕','ひじ','手首']],axis=1)
        clearance=arm_clearance(points,center,basis,radii,radius)
        total=float(np.maximum(-clearance,0).sum())
        expected=(report['target_accessory_collision']['sides'][label]['guard']['torso_after']['total_proxy_deficit_m'] if accessory_check else
                  report['target_head_collision']['sides'][label]['torso_after']['total_proxy_deficit_m'] if head_check
                  else report['target_body_collision']['sides'][label]['after_total_proxy_deficit_m'])
        diagnostics[label]={'penetrating_frames':int((clearance.min(axis=1)<-.001).sum()),
            'min_proxy_clearance_m':float(clearance.min()),'total_proxy_deficit_m':total,
            'saved_solver_metric_error_m':abs(total-expected)}
    audit['body_collision']={'enabled':True,'sides':diagnostics,
        'saved_solver_consistent':all(v['saved_solver_metric_error_m']<.001 for v in diagnostics.values()),
        'note':'Saved bone proxy samples only. Remaining overlaps are allowed and reported, not a mesh collision-free certificate.'}
    audit['passed'] &= audit['body_collision']['saved_solver_consistent']
    if head_check:
        from motionbridge.blender_head import animated_head_proxy
        from motionbridge.head_collision import clearance_report
        head_report=report['target_accessory_collision'] if accessory_check else report['target_head_collision']
        hc,hb,hr=animated_head_proxy(head_matrices,np.array(rig.data.bones['頭'].matrix_local),head_report['profile'])
        diagnostics={}
        for side,label in [('左','left'),('右','right')]:
            points=np.stack([track(side+n) for n in ['腕','ひじ','手首']],axis=1)
            measured=clearance_report(arm_clearance(points,hc,hb,hr,head_report['arm_radius_m']))
            expected_head=(head_report['sides'][label]['guard']['after'] if accessory_check else head_report['sides'][label]['after'])
            measured['saved_solver_metric_error_m']=abs(measured['total_proxy_deficit_m']-expected_head['total_proxy_deficit_m'])
            diagnostics[label]=measured
        audit['head_collision']={'enabled':True,'sides':diagnostics,
            'saved_solver_consistent':all(v['saved_solver_metric_error_m']<.001 for v in diagnostics.values()),
            'note':'Integer-frame arm/head proxy check; not mesh collision-free or accuracy certification.'}
        audit['passed'] &= audit['head_collision']['saved_solver_consistent']
    if accessory_check:
        from motionbridge.blender_accessories import accessory_surface,surface_vectors
        tree,surface=accessory_surface(bpy.data.objects['Nahida_mesh'],rig)
        delta=np.array(head_matrices)@np.linalg.inv(np.array(rig.data.bones['頭'].matrix_local))
        alpha=np.r_[np.linspace(.25,1,7),np.ones(9)];beta=np.r_[np.zeros(7),np.linspace(0,1,9)]
        diagnostics={}
        for side,label in [('左','left'),('右','right')]:
            points=np.stack([track(side+n) for n in ['腕','ひじ','手首']],axis=1)
            p=np.einsum('fij,fkj->fki',delta[:,:3,:3].swapaxes(1,2),points-delta[:,None,:3,3])
            samples=p[:,0,None]+alpha[None,:,None]*(p[:,1]-p[:,0])[:,None]+beta[None,:,None]*(p[:,2]-p[:,1])[:,None]
            _,depth=surface_vectors(tree,samples,report['target_accessory_collision']['arm_radius_m'])
            expected=report['target_accessory_collision']['sides'][label]['after_sum_surface_overlap_m']
            diagnostics[label]={'frames_over_1mm':int((depth.max(1)>.001).sum()),'max_surface_overlap_m':float(depth.max()),'saved_metric_error_m':abs(float(depth.sum())-expected)}
        audit['accessory_collision']={'sides':diagnostics,'rigid_geometry_unchanged':surface['rest_geometry_sha256']==report['target_accessory_collision']['surface']['rest_geometry_sha256']}
        audit['passed'] &= audit['accessory_collision']['rigid_geometry_unchanged'] and all(v['saved_metric_error_m']<.001 for v in diagnostics.values())
if report.get('surface_clearance',{}).get('enabled'):
    from motionbridge.blender_surfaces import audit_surface_clearance
    audit['surface_clearance']=audit_surface_clearance(scene,rig,report['surface_clearance'],reference_path=folder/'surface_solver_reference.npz')
    audit['passed'] &= audit['surface_clearance']['passed']
(folder / 'saved_project_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
print('AUDIT', audit['passed'], maximum_delta, flush=True)
if not audit['passed']:
    raise RuntimeError('Saved project audit failed')
preview=folder/'character_preview.png'
if not preview.is_file() or preview.stat().st_size<1000:
    scene.frame_set(report.get('preview_frame',scene.frame_start))
    scene.render.filepath=str(preview);scene.cycles.device='CPU';scene.render.threads_mode='FIXED';scene.render.threads=4
    bpy.ops.render.render(write_still=True)
