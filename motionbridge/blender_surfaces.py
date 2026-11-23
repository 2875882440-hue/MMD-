"""Bake per-video garment/hair corrective shapes, fully embedded in .blend.

Explicit Nahida profile; no old dance caches, external sidecars or cloth physics.
"""
import numpy as np
from .surface_clearance import capsule_distance,project_surface,deficits


def body_capsules(rig):
    pairs=[('腰','首',.045),('左足','左ひざ',.035),('右足','右ひざ',.035),
           ('左ひざ','左足首',.024),('右ひざ','右足首',.024),
           ('左腕','左ひじ',.024),('右腕','右ひじ',.024),
           ('左ひじ','左手首',.019),('右ひじ','右手首',.019)]
    return pairs


def bake_surface_clearance(scene,rig,objects,reference_path=None):
    """Correct selected flexible meshes after skinning via inverse LBS shapes."""
    import bpy
    pairs=body_capsules(rig);reference_tracks={}
    rest_caps=np.array([[rig.data.bones[a].head_local,rig.data.bones[b].head_local] for a,b,_ in pairs],float)
    all_bones=list(rig.data.bones);ids={b.name:i for i,b in enumerate(all_bones)}
    inverse=np.linalg.inv(np.array([b.matrix_local for b in all_bones],float))
    count=scene.frame_end-scene.frame_start+1
    report={'enabled':True,'method':'bounded_skin_inverse_triangle_contact_shapes_v4','max_vertex_shift_m':.12,
        'rest_overlap_policy':'Preserve original attachment/layer overlap; constrain additional penetration.',
        'capsules':[{'start':a,'end':b,'radius_m':r} for a,b,r in pairs],'objects':[],
        'note':'Embedded per-video corrective shapes. Capsule baseline plus bounded actual-triangle and inter-component layer constraints. No cloth dynamics, same-component self-collision, watertight-volume or continuous collision guarantee. Rigid head accessories are not deformed.'}
    # Bone tracks are small; evaluate the scene once per frame, then process
    # each mesh with exact linear skinning. Verify against Blender on samples.
    transforms=[];caps=[]
    for frame in range(scene.frame_start,scene.frame_end+1):
        scene.frame_set(frame)
        transforms.append(np.array([rig.pose.bones[b.name].matrix for b in all_bones],float)@inverse)
        # Blender vectors can wrap mutable pose data. Snapshot each frame now,
        # not after the animation loop, or all capsules follow the last frame.
        caps.append(np.array([[rig.pose.bones[a].head,rig.pose.bones[b].head] for a,b,_ in pairs],float))
    transforms=np.asarray(transforms);caps=np.asarray(caps,float)
    # Toon outlines share their source mesh. Bake the mesh datablock once;
    # the outline must inherit the very same shapes and skinning weights.
    groups={}
    for obj in objects:groups.setdefault(obj.data.as_pointer(),[]).append(obj)
    for consumers in groups.values():
        obj=next((o for o in consumers if len([m for m in o.modifiers if m.show_viewport])==1),None)
        if obj is None:raise ValueError('No plain skinned source for shared garment mesh')
        signature=lambda o: [(g.index,g.name) for g in o.vertex_groups]
        for linked in consumers:
            mods=[m for m in linked.modifiers if m.show_viewport]
            if (signature(linked)!=signature(obj) or not np.allclose(linked.matrix_world,obj.matrix_world)
                or not mods or mods[0].type!='ARMATURE' or mods[0].object!=rig
                or mods[0].use_deform_preserve_volume or any(m.type!='NODES' for m in mods[1:])):
                raise ValueError('Shared garment outline has incompatible skinning')
        if obj.data.shape_keys:raise ValueError('Surface profile requires a flexible mesh without existing shape keys')
        mods=[m for m in obj.modifiers if m.show_viewport]
        if len(mods)!=1 or mods[0].type!='ARMATURE' or mods[0].object!=rig or mods[0].use_deform_preserve_volume:
            raise ValueError('Surface inverse requires the inspected linear armature modifier')
        if not np.allclose(obj.matrix_world,np.eye(4)):raise ValueError('Surface profile requires identity mesh transform')
        n=len(obj.data.vertices);base=np.empty(n*3);obj.data.vertices.foreach_get('co',base);base=base.reshape(-1,3)
        names={g.index:g.name for g in obj.vertex_groups}
        influences=[[(ids[names[g.group]],g.weight) for g in v.groups if names.get(g.group) in ids and g.weight>0] for v in obj.data.vertices]
        width=max(map(len,influences));indices=np.zeros((n,width),int);weights=np.zeros((n,width))
        for v,row in enumerate(influences):
            if not row:raise ValueError('Unweighted flexible vertex')
            for k,(index,weight) in enumerate(row):indices[v,k]=index;weights[v,k]=weight
        weights/=weights.sum(1,keepdims=True)
        # Per-vertex thresholds preserve intentional rest attachments.
        limits=np.stack([np.minimum(radius+.001,capsule_distance(base,*cap)[0]) for cap,(_,_,radius) in zip(rest_caps,pairs)])
        raw=[];offsets=[];before_max=0.;before_hits=0;skin_error=0.
        for f in range(count):
            blended=(transforms[f,indices]*weights[...,None,None]).sum(1)
            positions=np.einsum('vij,vj->vi',blended[:,:3,:3],base)+blended[:,:3,3]
            if f in {0,count//2,count-1}:
                scene.frame_set(scene.frame_start+f);ev=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh()
                try:
                    actual=np.empty(n*3);mesh.vertices.foreach_get('co',actual)
                    skin_error=max(skin_error,float(np.max(abs(actual.reshape(-1,3)-positions))))
                finally:ev.to_mesh_clear()
            d=deficits(positions,caps[f],limits);before_max=max(before_max,float(d.max()));before_hits+=int((d.max(0)>.001).sum())
            solved=project_surface(positions,caps[f],limits)
            raw.append(positions.astype(np.float32));offsets.append((solved-positions).astype(np.float32))
        if skin_error>1e-5:raise ValueError('Inverse skin calibration does not match Blender: '+str(skin_error))
        raw=np.asarray(raw);offsets=np.asarray(offsets)
        padded=np.pad(offsets,[(2,2),(0,0),(0,0)],mode='edge')
        offsets=sum(w*padded[k:k+count] for k,w in enumerate([1/9,2/9,3/9,2/9,1/9]))
        baseline=[]
        for f in range(count):
            baseline.append(project_surface(raw[f],caps[f],limits,initial=raw[f]+offsets[f]))
        from .surface_topology import regularize_surface_tracks
        edges=np.array([edge.vertices[:] for edge in obj.data.edges],int)
        print('SURFACE_TOPOLOGY_START',obj.name,count,flush=True)
        solved_tracks,topology=regularize_surface_tracks(raw,np.asarray(baseline),caps,limits,edges,scene.render.fps/scene.render.fps_base)
        print('SURFACE_TOPOLOGY_DONE',topology['edge_frames_changed_over_5mm_before'],topology['edge_frames_changed_over_5mm_after'],flush=True)
        del baseline,offsets,padded
        from .surface_directed import coherent_surface_tracks
        obj.data.calc_loop_triangles()
        faces=np.array([triangle.vertices[:] for triangle in obj.data.loop_triangles],int)
        baseline_topology=topology.copy()
        solved_tracks,coherent=coherent_surface_tracks(raw,solved_tracks,caps,limits,edges,base,faces,rest_caps,scene.render.fps/scene.render.fps_base)
        final_metrics=coherent['after']
        topology.update(method='topology_with_coherent_contact_v2',
            maximum_edge_change_after_m=final_metrics['max_edge_change_m'],
            edge_frames_changed_over_5mm_after=final_metrics['edge_frames_over_5mm'],
            edge_change_squared_sum_after=final_metrics['edge_energy'],
            note='Final edge metrics include coherent contact. Proposal/backoff/extra-shift fields describe the v067 baseline pass only; see baseline_topology and coherent_contact for their separate budgets and decisions.')
        print('SURFACE_COHERENT_DONE',coherent['accepted'],coherent['accepted_components'],coherent['retained_components'],flush=True)
        from .blender_triangle_contacts import configured_body_surface,evaluated_points,TriangleSurface,mesh_contact_tracks
        body,body_rest,body_faces=configured_body_surface(scene)
        def body_at(f):
            scene.frame_set(scene.frame_start+f)
            return TriangleSurface(evaluated_points(body),body_faces)
        solved_tracks,triangle_contact=mesh_contact_tracks(raw,solved_tracks,caps,limits,edges,base,faces,
            body_rest,body_faces,body_at,scene.render.fps/scene.render.fps_base)
        final_metrics=triangle_contact['after_motion']
        topology.update(method='topology_with_triangle_contact_v3',
            maximum_edge_change_after_m=final_metrics['max_edge_change_m'],
            edge_frames_changed_over_5mm_after=final_metrics['edge_frames_over_5mm'],
            edge_change_squared_sum_after=final_metrics['edge_energy'],
            note='Final edge metrics include actual-triangle contact. Earlier proposal/backoff fields refer to the v067 baseline; coherent_contact is the v068 stage. See triangle_contact for the final incremental edge budget, decisions and temporal metrics.')
        print('SURFACE_TRIANGLE_CONTACT_DONE',triangle_contact['before']['body_samples_over_1mm'],triangle_contact['after']['body_samples_over_1mm'],flush=True)
        corrections=[];after_max=0.;after_hits=0;singular=0;max_shift=0.
        for f in range(count):
            solved=solved_tracks[f]
            blended=(transforms[f,indices]*weights[...,None,None]).sum(1)[:,:3,:3]
            valid=abs(np.linalg.det(blended))>.02;singular+=int((~valid).sum())
            delta=np.zeros_like(base);delta[valid]=np.linalg.solve(blended[valid],(solved-raw[f])[valid,:,None])[...,0]
            solved[~valid]=raw[f][~valid]
            d=deficits(solved,caps[f],limits);after_max=max(after_max,float(d.max()));after_hits+=int((d.max(0)>.001).sum())
            max_shift=max(max_shift,float(np.linalg.norm(solved-raw[f],axis=1).max()))
            corrections.append(delta.astype(np.float32))
        if reference_path is not None:reference_tracks[obj.name]=solved_tracks.astype(np.float32)
        obj.shape_key_add(name='Basis',from_mix=False);keys=obj.data.shape_keys;keys.use_relative=False
        for f,delta in enumerate(corrections):
            key=obj.shape_key_add(name=f'MotionBridge clearance {f+1:04d}',from_mix=False)
            key.interpolation='KEY_LINEAR';key.data.foreach_set('co',(base+delta).astype(np.float32).ravel())
            # Absolute shape-key frame values accumulate float32 rounding.
            # Key exact stored times rather than infer them from two endpoints.
            keys.eval_time=key.frame;keys.keyframe_insert('eval_time',frame=scene.frame_start+f)
        for layer in keys.animation_data.action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    for fc in bag.fcurves:
                        for key in fc.keyframe_points:key.interpolation='LINEAR'
        obj['motionbridge_surface_clearance']=True
        report['objects'].append({'name':obj.name,'linked_objects':[o.name for o in consumers if o!=obj],'vertices':n,'frames':count,'shapes':count,'topology':topology,'baseline_topology':baseline_topology,'coherent_contact':coherent,'triangle_contact':triangle_contact,
            'skin_math_max_error_m':skin_error,'before_vertex_frames_over_1mm':before_hits,
            'after_vertex_frames_over_1mm':after_hits,'before_max_additional_penetration_m':before_max,
            'after_max_additional_penetration_m':after_max,'maximum_shift_m':max_shift,'singular_vertex_frames_skipped':singular})
        print('SURFACE_CLEARANCE',obj.name,before_hits,after_hits,flush=True)
    if reference_path is not None:
        import hashlib
        from pathlib import Path
        reference_path=Path(reference_path)
        np.savez_compressed(reference_path,**reference_tracks)
        report['solver_reference']={'filename':reference_path.name,'sha256':hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            'encoding':'pre_embedding_world_vertices_float32_v1','playback_dependency':False}
    return report


def audit_surface_clearance(scene,rig,report,reference_path=None):
    import bpy
    pairs=body_capsules(rig);rest_caps=np.array([[rig.data.bones[a].head_local,rig.data.bones[b].head_local] for a,b,_ in pairs],float)
    records=[];reference_archive=None
    if report.get('solver_reference'):
        import hashlib
        from pathlib import Path
        if reference_path is None or not Path(reference_path).is_file():raise ValueError('Missing solver geometry validation reference')
        reference_path=Path(reference_path)
        if hashlib.sha256(reference_path.read_bytes()).hexdigest()!=report['solver_reference']['sha256']:
            raise ValueError('Solver geometry validation reference hash mismatch')
        reference_archive=np.load(reference_path,allow_pickle=False)
    for expected in report['objects']:
        obj=bpy.data.objects[expected['name']];n=len(obj.data.vertices)
        shared=all(bpy.data.objects[name].data==obj.data for name in expected.get('linked_objects',[]))
        base=np.empty(n*3);obj.data.vertices.foreach_get('co',base);base=base.reshape(-1,3)
        limits=np.stack([np.minimum(radius+.001,capsule_distance(base,*cap)[0]) for cap,(_,_,radius) in zip(rest_caps,pairs)])
        maximum=0.;hits=0;finite=True;edge_hits=0;edge_max=0.;edge_energy=0.;maximum_shift=0.
        recent=[];accelerations=[];actual_track=[];fps=scene.render.fps/scene.render.fps_base
        triangle_expected=expected.get('triangle_contact',{})
        geometry_reference={'mode':'not_required','passed':True,'max_error_m':0.,'tolerance_m':1e-5,'point_frames':0}
        solver_points=None;skin_reference=None
        if triangle_expected.get('enabled'):
            if reference_archive is not None:
                solver_points=reference_archive[obj.name]
                if solver_points.shape!=(scene.frame_end-scene.frame_start+1,n,3):raise ValueError('Solver reference dimensions changed')
                geometry_reference['mode']='pre_embedding_solver_vertices'
            else:
                # Compatibility for already-saved exports that predate the
                # reference snapshot. Verify native decoding independently at
                # every vertex; do not pretend this is the original solver track.
                skin_reference=_saved_key_skin_reference(obj,rig,scene.frame_start)
                geometry_reference['mode']='legacy_saved_absolute_key_lbs'
            geometry_reference['original_solver_track_available']=solver_points is not None
        topology=expected.get('topology',{})
        if topology.get('enabled'):
            from .surface_topology import edge_changes
            edges=np.array([edge.vertices[:] for edge in obj.data.edges],int)
        for f in range(scene.frame_start,scene.frame_end+1):
            scene.frame_set(f);caps=np.array([[rig.pose.bones[a].head,rig.pose.bones[b].head] for a,b,_ in pairs],float)
            ev=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh()
            try:
                if len(mesh.vertices)!=n:raise ValueError('Surface topology changed')
                points=np.empty(n*3);mesh.vertices.foreach_get('co',points);points=points.reshape(-1,3)
                finite &= bool(np.isfinite(points).all());d=deficits(points,caps,limits)
                if triangle_expected.get('enabled'):actual_track.append(points.astype(np.float32))
                if triangle_expected.get('enabled'):
                    from .triangle_contacts import geometry_reference_error
                    reference=solver_points[f-scene.frame_start] if solver_points is not None else skin_reference(f)
                    error,reference_ok=geometry_reference_error(points,reference)
                    geometry_reference['max_error_m']=max(geometry_reference['max_error_m'],error)
                    geometry_reference['passed'] &= reference_ok;geometry_reference['point_frames']+=n
                maximum=max(maximum,float(d.max()));hits+=int((d.max(0)>.001).sum())
                if len(recent)==2:accelerations.append(np.linalg.norm((points-2*recent[-1]+recent[-2])*fps**2,axis=1))
                recent.append(points)
                if len(recent)>2:recent.pop(0)
            finally:ev.to_mesh_clear()
            if topology.get('enabled'):
                # Independently evaluate the saved Basis through the same rig.
                # This only changes this read-only audit process, never the file.
                keys=obj.data.shape_keys;time=keys.eval_time;keys.eval_time=0;bpy.context.view_layer.update()
                ev=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh()
                try:
                    raw=np.empty(n*3);mesh.vertices.foreach_get('co',raw)
                    change=edge_changes(points,raw.reshape(-1,3),edges)
                    edge_hits+=int((change>.005).sum());edge_max=max(edge_max,float(change.max()));edge_energy+=float(np.sum(change**2))
                    maximum_shift=max(maximum_shift,float(np.linalg.norm(points-raw.reshape(-1,3),axis=1).max()))
                finally:ev.to_mesh_clear();keys.eval_time=time
        consistent=abs(maximum-expected['after_max_additional_penetration_m'])<1e-5 and abs(hits-expected['after_vertex_frames_over_1mm'])<30
        topology_consistent=(not topology.get('enabled') or
            (abs(edge_max-topology['maximum_edge_change_after_m'])<1e-5
             and abs(edge_energy-topology['edge_change_squared_sum_after'])<.001
             and abs(edge_hits-topology['edge_frames_changed_over_5mm_after'])<10))
        displacement_consistent=(not topology.get('enabled') or
            (abs(maximum_shift-expected['maximum_shift_m'])<1e-5 and maximum_shift<=report['max_vertex_shift_m']+1e-5))
        accelerations=np.asarray(accelerations)
        acceleration_p99=float(np.percentile(accelerations,99)) if accelerations.size else 0.
        acceleration_max=float(accelerations.max()) if accelerations.size else 0.
        expected_motion=triangle_expected.get('after_motion',expected.get('coherent_contact',{}).get('after'))
        temporal_consistent=(not expected_motion or
            (abs(acceleration_p99-expected_motion['vertex_acceleration_p99'])<.01
             and abs(acceleration_max-expected_motion['vertex_acceleration_max'])<.01))
        triangle_audit=None
        if triangle_expected.get('enabled'):
            from .blender_triangle_contacts import audit_mesh_contacts
            triangle_audit=audit_mesh_contacts(scene,obj,np.asarray(actual_track),triangle_expected,geometry_reference)
        records.append({'name':obj.name,'evaluated_frames':scene.frame_end-scene.frame_start+1,
            'vertex_frames_over_1mm':hits,'max_additional_penetration_m':maximum,'finite':finite,
            'saved_solver_consistent':consistent,'linked_shapes_consistent':shared,'embedded_shapes':len(obj.data.shape_keys.key_blocks)-1,
            'topology_consistent':topology_consistent,'displacement_consistent':displacement_consistent,'temporal_consistent':temporal_consistent,
            'vertex_acceleration_p99':acceleration_p99,'vertex_acceleration_max':acceleration_max,
            'maximum_shift_m':maximum_shift if topology else None,'edge_frames_changed_over_5mm':edge_hits if topology else None,
            'maximum_edge_change_m':edge_max if topology else None,'edge_change_squared_sum':edge_energy if topology else None,
            'triangle_contacts':triangle_audit,'triangle_consistent':not triangle_audit or triangle_audit['saved_geometry_consistent']})
    if reference_archive is not None:reference_archive.close()
    return {'enabled':True,'objects':records,'passed':all(r['finite'] and r['saved_solver_consistent'] and r['linked_shapes_consistent'] and r['topology_consistent'] and r['displacement_consistent'] and r['temporal_consistent'] and r['triangle_consistent'] for r in records),
        'note':'Actual saved vertices checked against body capsules and, for v069+, actual saved collider triangles/layers; remaining contacts reported, not collision-free certification.'}


def _saved_key_skin_reference(obj,rig,frame_start):
    """Independent forward LBS for legacy absolute-key export verification."""
    bones=list(rig.data.bones);bone_ids={b.name:i for i,b in enumerate(bones)}
    inverse=np.linalg.inv(np.array([b.matrix_local for b in bones],float))
    names={g.index:g.name for g in obj.vertex_groups}
    rows=[[(bone_ids[names[g.group]],g.weight) for g in v.groups if names.get(g.group) in bone_ids and g.weight>0] for v in obj.data.vertices]
    if any(not row for row in rows):raise ValueError('Unweighted legacy verification vertex')
    width=max(map(len,rows));indices=np.zeros((len(rows),width),int);weights=np.zeros_like(indices,float)
    for v,row in enumerate(rows):
        for k,(i,w) in enumerate(row):indices[v,k]=i;weights[v,k]=w
    weights/=weights.sum(1,keepdims=True)
    keys=obj.data.shape_keys
    if keys is None or keys.use_relative:raise ValueError('Legacy surface requires absolute corrective keys')
    def at_frame(frame):
        key=keys.key_blocks[frame-frame_start+1]
        if abs(keys.eval_time-key.frame)>1e-4:raise ValueError('Corrective key timing mismatch')
        coords=np.empty(len(rows)*3);key.data.foreach_get('co',coords);coords=coords.reshape(-1,3)
        transforms=np.array([rig.pose.bones[b.name].matrix for b in bones],float)@inverse
        blended=(transforms[indices]*weights[...,None,None]).sum(1)
        return np.einsum('vij,vj->vi',blended[:,:3,:3],coords)+blended[:,:3,3]
    return at_frame
