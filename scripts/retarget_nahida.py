"""Explicit Nahida/MMD profile: reuse MotionBridge output in an existing scene.

Blender --background --factory-startup --threads 4 --python this.py --
    source.blend motion.json full_character.blend output_directory
This is an offline, named-rig integration tool, not automatic arbitrary-rig support.
"""
from pathlib import Path
import sys
import json
import hashlib
import math
import bpy
import numpy as np
from mathutils import Matrix, Vector, Quaternion
from mathutils.kdtree import KDTree

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from motionbridge.exporters import solve_rotations, validate_motion

C = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    arguments=sys.argv[sys.argv.index('--') + 1:]
    source, motion_path, complete, output = map(Path,arguments[:4])
    if len(arguments)>5 or (len(arguments)==5 and arguments[4] not in {'true','false'}):raise ValueError('Invalid surface clearance option')
    surface_enabled=len(arguments)<5 or arguments[4]=='true'
    source, motion_path, complete, output = [p.resolve() for p in (source, motion_path, complete, output)]
    output.mkdir(parents=True, exist_ok=True)
    report = {'profile': 'Nahida MMD 488-bone / v1', 'original_project': str(source),
        'original_sha256_before': digest(source), 'motion': str(motion_path),
        'motion_sha256': digest(motion_path), 'complete_character_source': str(complete),
        'limitations': ['Monocular pose estimation errors remain; this is not capture accuracy validation.',
            'No captured finger or facial animation; neutral hands and face.',
            'Lowest-foot support height is a grounding heuristic; jumps and global travel are not recovered.',
            'Old garment cache is disabled; skeletal garment deformation has no secondary dynamics.']}
    motion = json.loads(motion_path.read_text(encoding='utf-8'))
    names, parents, rest, poses, fps = validate_motion(motion)
    world_mode = motion.get('root_mode') == 'estimated_world_translation'
    _, world, _ = solve_rotations(motion)
    rest = (C @ rest.T).T
    rotations = C @ world @ C.T
    index = {n: i for i, n in enumerate(names)}
    bpy.ops.wm.open_mainfile(filepath=str(source), load_ui=False)
    bpy.ops.file.make_paths_absolute()
    scene = bpy.context.scene
    rig = bpy.data.objects['Nahida_arm']
    mapping = {'腰': 'Hips', '下半身': 'Hips', '上半身': 'Spine',
               '上半身2': 'Chest', '首': 'Neck', '頭': 'Head'}
    calibrated = world_mode or bool(motion.get('projection_refinement', {}).get('enabled'))
    segments = {}
    for side, english in [('左', 'Left'), ('右', 'Right')]:
        if calibrated:
            mapping[side + '肩'] = english + 'Shoulder'
            segments[side + '肩'] = (english + 'Shoulder', english + 'Arm')
        for target, source_name, end in [('腕', 'Arm', 'ForeArm'), ('ひじ', 'ForeArm', 'Hand'),
                                       ('足', 'UpLeg', 'Leg'), ('ひざ', 'Leg', 'Foot'), ('足首', 'Foot', 'ToeBase')]:
            mapping[side + target] = english + source_name
            segments[side + target] = (english + source_name, english + end)
        mapping[side + '手首'] = english + 'ForeArm'
        for main_bone in ['足', 'ひざ', '足首']:
            mapping[side + main_bone + 'D'] = mapping[side + main_bone]
            segments[side + main_bone + 'D'] = segments[side + main_bone]
    if world_mode:
        report['twist_policy'] = 'Use predicted SMPL-X rotations without legacy direction fitting or speed caps'
        report['temporal_cleanup'] = None
        report['limitations'][2] = 'World travel is an uncalibrated monocular estimate, scaled by leg lengths; foot sliding may remain.'
    elif not motion.get('rotation_stabilization', {}).get('enabled'):
        # A single observed segment constrains direction, not axial twist. The source
        # parent-relative shortest arc can jump near a folded-back elbow. Parallel
        # transport the previous frame's orientation to retain the observed direction
        # without inventing discontinuous roll. No timestamp-specific pose edits.
        positions_blender = poses @ C.T
        direction_error = 0.
        for start, end in sorted(set(segments.values()) | {('Spine', 'Chest'), ('Neck', 'Head')}):
            j, k = index[start], index[end]
            rest_direction = Vector(rest[k] - rest[j]).normalized()
            previous_orientation = Matrix(rotations[0, j].tolist())
            for f in range(len(poses)):
                observed = Vector(positions_blender[f, k] - positions_blender[f, j]).normalized()
                delta = (previous_orientation @ rest_direction).rotation_difference(observed)
                orientation = delta.to_matrix() @ previous_orientation
                # Re-normalize after repeated transports to prevent matrix drift.
                orientation = orientation.to_quaternion().normalized().to_matrix()
                rotations[f, j] = np.asarray(orientation)
                direction_error = max(direction_error, (orientation @ rest_direction - observed).length)
                previous_orientation = orientation
        report['twist_policy'] = 'Temporal parallel transport; limb directions retained; axial twist not measured'
        report['transport_direction_max_unit_vector_error'] = direction_error
        rotations[:, index['Head']] = rotations[:, index['Neck']]
        # Source estimates occasionally jump in depth/left-right orientation. Use a
        # centered five-frame quaternion medoid to reject isolated spikes, then bound
        # angular velocity. This intentionally trades rapid-motion fidelity for a
        # usable target animation; source FBX is kept unchanged for comparison.
        filtered_changes = []
        used = sorted(set(mapping.values()))
        for name in used:
            j = index[name]
            raw = [Matrix(r.tolist()).to_quaternion().normalized() for r in rotations[:, j]]
            samples = []
            for f in range(len(raw)):
                window = raw[max(0, f - 2):min(len(raw), f + 3)]
                medoid = min(window, key=lambda q: sum(1 - abs(q.dot(other)) for other in window))
                samples.append(medoid.copy())
            limit = math.radians(360 if name in {'Hips', 'Spine', 'Chest', 'Neck', 'Head'} else 720) / fps
            prev = samples[0]
            for f, q in enumerate(samples):
                if prev.dot(q) < 0:
                    q.negate()
                angle = 2 * math.acos(min(1., max(-1., prev.dot(q))))
                limited = prev.slerp(q, min(1., limit / max(angle, 1e-8))).normalized()
                error = math.degrees(2 * math.acos(min(1., abs(limited.dot(raw[f])))))
                filtered_changes.append(error)
                rotations[f, j] = np.asarray(limited.to_matrix())
                prev = limited
        report['temporal_cleanup'] = {'quaternion_medoid_window_frames': 5,
            'torso_head_speed_limit_deg_s': 360, 'limb_speed_limit_deg_s': 720,
            'maximum_rotation_adjustment_deg': max(filtered_changes),
            'median_rotation_adjustment_deg': float(np.median(filtered_changes)),
            'note': 'Changes estimated poses to suppress spikes; can attenuate very fast movement. Not measured physical accuracy.'}
        report['limitations'].append('Five-frame robust filtering and angular speed limits can attenuate very fast motion.')
    else:
        report['twist_policy'] = 'Use explicit stabilized source rotations without filtering twice'
        report['temporal_cleanup'] = motion['rotation_stabilization']
    missing = set(mapping) - set(rig.pose.bones.keys())
    if missing:
        raise ValueError('The selected scene is not the inspected MMD profile: ' + str(missing))
    if rig.matrix_world != Matrix.Identity(4):
        raise ValueError('This profile requires the inspected identity object transform.')
    old_action = rig.animation_data.action if rig.animation_data else None
    if old_action:
        old_action.use_fake_user = True
        report['preserved_old_action'] = old_action.name
    # Preserve all original datablocks; detach actions only in the NEW project.
    for obj in scene.objects:
        if obj.animation_data:
            if obj.animation_data.action:
                obj.animation_data.action.use_fake_user = True
            obj.animation_data_clear()
        if obj.type == 'MESH' and obj.data.shape_keys:
            obj.data.shape_keys.animation_data_clear()
            for key in obj.data.shape_keys.key_blocks:
                key.value = 0
    muted = []
    for b in rig.pose.bones:
        b.matrix_basis = Matrix.Identity(4)
        b.rotation_mode = 'QUATERNION'
        for c in b.constraints:
            c.mute = True
            muted.append([b.name, c.name])
    report['muted_original_constraints'] = muted
    # The cloth objects have original rest geometry but no vertex groups. Transfer
    # existing skin weights from the same full character, preserving cleaned UVs.
    with bpy.data.libraries.load(str(complete), link=False) as (src, dst):
        dst.objects = ['Nahida_mesh']
    donor = dst.objects[0]
    tree = KDTree(len(donor.data.vertices))
    for v in donor.data.vertices:
        tree.insert(v.co, v.index)
    tree.balance()
    garment_objects = [o for o in scene.objects if o.type == 'MESH' and any(m.type == 'MESH_CACHE' for m in o.modifiers)]
    weight_error = 0.
    for obj in garment_objects:
        for m in list(obj.modifiers):
            if m.type == 'MESH_CACHE':
                obj.modifiers.remove(m)
        groups = {g.index: obj.vertex_groups.get(g.name) or obj.vertex_groups.new(name=g.name) for g in donor.vertex_groups}
        for v in obj.data.vertices:
            _, source_index, distance = tree.find(v.co)
            weight_error = max(weight_error, distance)
            for g in donor.data.vertices[source_index].groups:
                groups[g.group].add([v.index], g.weight, 'REPLACE')
        m = obj.modifiers.new('MotionBridge new motion skinning', 'ARMATURE')
        m.object = rig
        obj.modifiers.move(len(obj.modifiers) - 1, 0)
        obj.parent = rig
    if weight_error > .0001:
        raise ValueError('Full character rest topology does not match garment geometry: ' + str(weight_error))
    report['garment_weight_transfer_max_distance_m'] = weight_error
    report['restored_garments'] = [o.name for o in garment_objects]
    bpy.data.objects.remove(donor, do_unlink=True)
    # Cancel old audio; the previous dance's BGM is not the uploaded motion.
    if scene.sequence_editor:
        scene.sequence_editor_clear()
    scene.render.use_sequencer = False
    scene.frame_start, scene.frame_end = 1, len(poses)
    scene.render.fps = round(fps)
    scene.render.fps_base = round(fps) / fps
    rig.animation_data_create()
    action = bpy.data.actions.new('MotionBridge uploaded dance - Nahida retarget')
    rig.animation_data.action = action
    action.use_fake_user = True
    bone_rest = {b.name: b.matrix_local.copy() for b in rig.data.bones}
    relative = {b.name: (bone_rest[b.parent.name].inverted() @ bone_rest[b.name]) if b.parent else bone_rest[b.name] for b in rig.data.bones}
    corrections = {}
    for target, (start, end) in segments.items():
        main = target[:-1] if target.endswith('D') else target
        a = rig.data.bones[main]
        direction = a.tail_local - a.head_local
        corrections[target] = direction.rotation_difference(Vector(rest[index[end]] - rest[index[start]])).to_matrix()
    for side in ['左', '右']:
        corrections[side + '手首'] = corrections[side + 'ひじ']
    # parent-before-child traversal, independent of Blender datablock order.
    bones = []
    def visit(b):
        bones.append(b)
        for child in b.children:
            visit(child)
    for b in rig.data.bones:
        if b.parent is None:
            visit(b)
    from motionbridge.target_ik import hand_goal, chain_ik
    enhanced = not world_mode and bool(motion.get('settings',{}).get('enhance_motion',False))
    source_blender = poses @ C.T
    contact_weights = np.asarray(motion.get('foot_contacts',{}).get('weights',np.zeros((len(poses),2))))
    world_contact_enabled = world_mode and bool(motion.get('world_contact_refinement',{}).get('enabled'))
    confidence = np.asarray([frame.get('observation_weights',frame.get('confidence',[1.]*len(names))) for frame in motion['frames']])
    # Source Head is an ear midpoint, whereas MMD 頭 starts near the neck.
    # Physical left/right eye bones give an inspected face-height proxy;
    # 両目 is an external gaze control and must never be used as a landmark.
    face_rest = rig.data.bones['頭'].head_local.copy()
    if calibrated and all(n in rig.data.bones for n in ['左目','右目']):
        face_rest = (rig.data.bones['左目'].head_local + rig.data.bones['右目'].head_local) * .5
    face_local = bone_rest['頭'].inverted() @ face_rest
    report['reference_landmarks'] = {'enabled':calibrated,
        'face_rest':list(face_rest), 'face_method':'physical_eye_midpoint_proxy' if calibrated else 'head_bone_origin',
        'head_tip_rest':list(rig.data.bones['頭'].tail_local),
        'shoulder_mapping':calibrated,
        'note':'Eye midpoint approximates source ear height; bone tip is not a measured scalp landmark. Soles remain the original profile heuristic.'}
    def assemble(orientations, lift=0.):
        matrices = {}
        for b in bones:
            base = matrices[b.parent.name] @ relative[b.name] if b.parent else relative[b.name].copy()
            if b.name == '全ての親':
                base.translation += Vector(lift if world_mode else (0,0,lift))
            if b.name in orientations:
                desired = orientations[b.name].to_4x4()
                desired.translation = base.translation
                matrices[b.name] = desired
            else:
                matrices[b.name] = base
        return matrices
    previous = {}
    previous_world = {}
    ik_rate_limited_samples = 0
    expected = []
    lifts = []
    contact_anchors = [None,None]
    endpoint_errors = []
    target_contact_points = []
    solved_orientations = []
    finish_temporal = not world_mode and bool(motion.get('temporal_finish',{}).get('enabled'))
    root_offsets = None
    for f in range(len(poses)):
        orientations = {n:Matrix(rotations[f,index[src]].tolist()) @ corrections.get(n,Matrix.Identity(3)) @ bone_rest[n].to_3x3() for n,src in mapping.items()}
        matrices = assemble(orientations, (0,0,0) if world_mode else 0.)
        # Place the lower of the two rest sole landmarks on the original floor.
        sole_z = []
        for side in ['左', '右']:
            b = rig.data.bones[side + '足首']
            for sole_point in [Vector((b.head_local.x, .032, 0.)), Vector((b.head_local.x, -.065, 0.))]:
                sole_z.append((matrices[b.name] @ b.matrix_local.inverted() @ sole_point).z)
        lift = .003 - min(sole_z)
        if world_mode:
            if root_offsets is None:
                from motionbridge.world_retarget import world_root_offsets
                source_lengths, target_lengths = [], []
                for side, english in [('左','Left'),('右','Right')]:
                    source_lengths.append(sum(float(np.linalg.norm(rest[index[english+b]]-rest[index[english+a]]))
                        for a,b in [('UpLeg','Leg'),('Leg','Foot')]))
                    target_lengths.append(sum((rig.data.bones[side+b].head_local-rig.data.bones[side+a].head_local).length
                        for a,b in [('足','ひざ'),('ひざ','足首')]))
                root_offsets, report['world_root_transfer'] = world_root_offsets(
                    source_blender[:,index['Hips']], float(np.mean(source_lengths)), float(np.mean(target_lengths)), lift)
            lift = root_offsets[f]
        if enhanced:
            # During a probable airborne interval, do not invent a grounded foot.
            # Relative body depth is still monocular; this is not jump recovery.
            if f and not np.any(contact_weights[f] > 0):
                lift = lifts[-1]*.85 + lift*.15
            matrices = assemble(orientations,lift)
            for side, english in [('左','Left'),('右','Right')]:
                a,b,c=[side+n for n in ['腕','ひじ','手首']]
                points=[np.array(matrices[n].translation) for n in [a,b,c]]
                lengths=[float(np.linalg.norm(points[1]-points[0])),float(np.linalg.norm(points[2]-points[1]))]
                goal=hand_goal(source_blender[f],names,{'shoulder':points[0], 'head':np.array(matrices['頭'] @ face_local),
                    'chest':np.array(matrices['上半身2'].translation)},english,lengths)
                strength=float(np.clip((confidence[f,index[english+'Hand']]-.4)/.4,0,1))
                middle,end=chain_ik(*points,goal,lengths,strength)
                for name,old,new in [(a,points[1]-points[0],middle-points[0]),(b,points[2]-points[1],end-middle)]:
                    delta=Vector(old).rotation_difference(Vector(new)).to_matrix()
                    orientations[name]=delta@orientations[name]
                matrices=assemble(orientations,lift)
                endpoint_errors.append(float(np.linalg.norm(np.array(matrices[c].translation)-goal)))
            for s,side in enumerate(['左','右']):
                a,b,c=[side+n for n in ['足','ひざ','足首']]
                points=[np.array(matrices[n].translation) for n in [a,b,c]]
                weight=float(contact_weights[f,s])
                if weight>0:
                    if contact_anchors[s] is None:
                        contact_anchors[s]=points[2].copy()
                    lengths=[float(np.linalg.norm(points[1]-points[0])),float(np.linalg.norm(points[2]-points[1]))]
                    middle,end=chain_ik(*points,contact_anchors[s],lengths,weight)
                    for name,old,new in [(a,points[1]-points[0],middle-points[0]),(b,points[2]-points[1],end-middle)]:
                        delta=Vector(old).rotation_difference(Vector(new)).to_matrix()
                        orientations[name]=delta@orientations[name]
                        orientations[name+'D']=delta@orientations[name+'D']
                    matrices=assemble(orientations,lift)
                else:
                    contact_anchors[s]=None
            target_contact_points.append([list(matrices[side+'足首'].translation) for side in ['左','右']])
        else:
            matrices=assemble(orientations,lift)
        if world_contact_enabled:
            # Preserve global travel; fit only the feet identified as supporting.
            # Unlike the legacy branch, no per-frame ground reset or speed cap.
            for s,side in enumerate(['左','右']):
                a,b,c=[side+n for n in ['足','ひざ','足首']]
                points=[np.array(matrices[n].translation) for n in [a,b,c]]
                weight=float(contact_weights[f,s])
                if weight>0:
                    if contact_anchors[s] is None:
                        contact_anchors[s]=points[2].copy()
                        # Anchor the estimated sole to the stage, not to its
                        # possibly floating height on entry into support.
                        foot=rig.data.bones[c]
                        floor=min((matrices[c] @ foot.matrix_local.inverted() @ q).z
                            for q in [Vector((foot.head_local.x,.032,0.)),Vector((foot.head_local.x,-.065,0.))])
                        correction=.003-floor
                        leg_length=sum(float(np.linalg.norm(points[k+1]-points[k])) for k in [0,1])
                        if abs(correction)<.12*leg_length:
                            contact_anchors[s][2]+=correction
                    lengths=[float(np.linalg.norm(points[1]-points[0])),float(np.linalg.norm(points[2]-points[1]))]
                    if np.linalg.norm(contact_anchors[s]-points[2])<.12*sum(lengths):
                        middle,end=chain_ik(*points,contact_anchors[s],lengths,weight)
                        for name,old,new in [(a,points[1]-points[0],middle-points[0]),(b,points[2]-points[1],end-middle)]:
                            delta=Vector(old).rotation_difference(Vector(new)).to_matrix()
                            orientations[name]=delta@orientations[name]
                            orientations[name+'D']=delta@orientations[name+'D']
                        matrices=assemble(orientations,lift)
                else:contact_anchors[s]=None
            target_contact_points.append([list(matrices[side+'足首'].translation) for side in ['左','右']])
        if enhanced:
            for n,orientation in orientations.items():
                q=orientation.to_quaternion().normalized()
                if n in previous_world:
                    prior=previous_world[n]
                    if prior.dot(q)<0:
                        q.negate()
                    angle=2*math.acos(min(1.,max(-1.,prior.dot(q))))
                    limits=report['temporal_cleanup']
                    cap=limits['torso_head_speed_limit_deg_s'] if mapping[n] in {'Hips','Spine','Chest','Neck','Head'} else limits['limb_speed_limit_deg_s']
                    limit=math.radians(cap)/fps
                    if angle>limit:
                        q=prior.slerp(q,limit/angle).normalized()
                        ik_rate_limited_samples+=1
                orientations[n]=q.to_matrix()
                previous_world[n]=q.copy()
            matrices=assemble(orientations,lift)
            target_contact_points[-1]=[list(matrices[side+'足首'].translation) for side in ['左','右']]
        if finish_temporal:
            for side in ['左','右']:
                wrist, elbow = side+'手首', side+'ひじ'
                orientations[wrist] = orientations[elbow] @ bone_rest[elbow].to_3x3().inverted() @ bone_rest[wrist].to_3x3()
        solved_orientations.append(orientations)
        lifts.append(lift)
        if f % 150 == 0:
            print('RETARGET_FRAME', f + 1, flush=True)
    if finish_temporal:
        from motionbridge.temporal import smooth_rotations, smooth_values
        order=list(mapping)
        tracks=np.array([[np.asarray(o[n]) for n in order] for o in solved_orientations])
        tracks=smooth_rotations(tracks,fps,.045)
        solved_orientations=[{n:Matrix(tracks[f,j].tolist()) for j,n in enumerate(order)} for f in range(len(poses))]
        lifts=smooth_values(np.asarray(lifts),fps,.045).tolist()
        report['target_temporal_finish']={'enabled':True,'sigma_seconds':.045,
            'method':'centered_so3_after_target_ik_v1','note':'Short gestures and exact support positions can soften; unobserved wrists inherit forearms.'}
    if world_contact_enabled and 'ground_weights' in motion.get('foot_contacts',{}):
        from motionbridge.world_retarget import target_ground_offsets
        sole_tracks=[]
        for orientations,lift in zip(solved_orientations,lifts):
            matrices=assemble(orientations,lift);row=[]
            for side in ['左','右']:
                foot=rig.data.bones[side+'足首']
                row.append(min((matrices[foot.name] @ foot.matrix_local.inverted() @ q).z
                    for q in [Vector((foot.head_local.x,.032,0.)),Vector((foot.head_local.x,-.065,0.))]))
            sole_tracks.append(row)
        vertical=target_ground_offsets(sole_tracks,motion['foot_contacts']['ground_weights'],fps,float(np.mean(target_lengths)))
        lifts=np.asarray(lifts).copy();lifts[:,2]+=vertical
        root_offsets=lifts.copy()
        report['target_ground_fit']={'enabled':True,'method':'bounded_sole_baseline_v1',
            'vertical_offsets_m':vertical.tolist(),'max_offset_m':float(np.max(np.abs(vertical))),
            'root_horizontal_unchanged':True,'note':'Estimated sole plane. Short support fades; no correction across long unsupported intervals. Jumps are not clamped.'}
    if world_mode and motion.get('body_collision',{}).get('enabled'):
        # Run the SAME geometric solver in target proportions, before baking.
        # Only arm/forearm and corresponding wrist orientation may change.
        from motionbridge.body_collision import torso_space,avoid_arm_torso
        track=[assemble(o,l) for o,l in zip(solved_orientations,lifts)]
        def heads(name):return np.array([m[name].translation for m in track])
        center,basis,radii,radius=torso_space(heads('腰'),heads('左腕'),heads('右腕'),
            np.array(bone_rest['腰'].translation),np.array(bone_rest['左腕'].translation),np.array(bone_rest['右腕'].translation))
        body_report={'enabled':True,'method':'bounded_temporal_arm_torso_proxy_v1','radii_m':radii.tolist(),
                     'arm_radius_m':radius,'sides':{},'note':'Target-proportioned proxy, not evaluated body/cloth mesh collision. Unresolved overlaps are reported.'}
        for side,label in [('左','left'),('右','right')]:
            order=[side+n for n in ['腕','ひじ','手首']]
            points=np.stack([heads(n) for n in order],axis=1)
            solved,body_report['sides'][label]=avoid_arm_torso(points,center,basis,radii,radius,fps)
            for f in range(len(poses)):
                for k in [0,1]:
                    delta=Vector(points[f,k+1]-points[f,k]).rotation_difference(Vector(solved[f,k+1]-solved[f,k])).to_matrix()
                    solved_orientations[f][order[k]]=delta@solved_orientations[f][order[k]]
                    if k==1:solved_orientations[f][order[2]]=delta@solved_orientations[f][order[2]]
        report['target_body_collision']=body_report
    if world_mode and motion.get('head_collision',{}).get('enabled'):
        from motionbridge.body_collision import torso_space
        from motionbridge.head_collision import avoid_arm_head
        from motionbridge.blender_head import calibrate_face_proxy,animated_head_proxy
        track=[assemble(o,l) for o,l in zip(solved_orientations,lifts)]
        def heads(name):return np.array([m[name].translation for m in track])
        torso=torso_space(heads('腰'),heads('左腕'),heads('右腕'),np.array(bone_rest['腰'].translation),
            np.array(bone_rest['左腕'].translation),np.array(bone_rest['右腕'].translation))
        profile=calibrate_face_proxy(rig,bpy.data.objects['Nahida_mesh'],['颜'],'頭')
        center,basis,radii=animated_head_proxy([np.array(m['頭']) for m in track],np.array(bone_rest['頭']),profile)
        radius=float(np.linalg.norm(np.array(bone_rest['左腕'].translation)-np.array(bone_rest['右腕'].translation))*.12)
        head_report={'enabled':True,'method':'bounded_head_proxy_torso_guard_v1','profile':profile,'arm_radius_m':radius,'sides':{}}
        for side,label in [('左','left'),('右','right')]:
            order=[side+n for n in ['腕','ひじ','手首']]
            points=np.stack([heads(n) for n in order],axis=1)
            solved,head_report['sides'][label]=avoid_arm_head(points,center,basis,radii,radius,fps,torso)
            for f in range(len(poses)):
                for k in [0,1]:
                    delta=Vector(points[f,k+1]-points[f,k]).rotation_difference(Vector(solved[f,k+1]-solved[f,k])).to_matrix()
                    solved_orientations[f][order[k]]=delta@solved_orientations[f][order[k]]
                    if k==1:solved_orientations[f][order[2]]=delta@solved_orientations[f][order[2]]
        report['target_head_collision']=head_report
    if surface_enabled:
        from motionbridge.body_collision import torso_space
        from motionbridge.blender_head import calibrate_face_proxy,animated_head_proxy
        from motionbridge.blender_accessories import accessory_surface,avoid_accessories
        track=[assemble(o,l) for o,l in zip(solved_orientations,lifts)]
        def heads(name):return np.array([m[name].translation for m in track])
        torso=torso_space(heads('腰'),heads('左腕'),heads('右腕'),np.array(bone_rest['腰'].translation),
            np.array(bone_rest['左腕'].translation),np.array(bone_rest['右腕'].translation))
        profile=calibrate_face_proxy(rig,bpy.data.objects['Nahida_mesh'],['颜'],'頭')
        matrices=np.array([m['頭'] for m in track]);head_proxy=animated_head_proxy(matrices,np.array(bone_rest['頭']),profile)
        radius=float(np.linalg.norm(np.array(bone_rest['左腕'].translation)-np.array(bone_rest['右腕'].translation))*.12)
        tree,surface=accessory_surface(bpy.data.objects['Nahida_mesh'],rig)
        accessory_report={'enabled':True,'surface':surface,'profile':profile,'arm_radius_m':radius,'sides':{}}
        for side,label in [('左','left'),('右','right')]:
            order=[side+n for n in ['腕','ひじ','手首']];points=np.stack([heads(n) for n in order],axis=1)
            solved,accessory_report['sides'][label]=avoid_accessories(points,matrices,np.array(bone_rest['頭']),tree,radius,fps,torso,head_proxy)
            for f in range(len(poses)):
                for k in [0,1]:
                    delta=Vector(points[f,k+1]-points[f,k]).rotation_difference(Vector(solved[f,k+1]-solved[f,k])).to_matrix()
                    solved_orientations[f][order[k]]=delta@solved_orientations[f][order[k]]
                    if k==1:solved_orientations[f][order[2]]=delta@solved_orientations[f][order[2]]
        report['target_accessory_collision']=accessory_report
    # Bake final post-IK tracks; nothing re-solves orientations after this point.
    for f,(orientations,lift) in enumerate(zip(solved_orientations,lifts)):
        matrices=assemble(orientations,lift)
        if enhanced:
            target_contact_points[f]=[list(matrices[side+'足首'].translation) for side in ['左','右']]
        for b in bones:
            if b.name not in mapping:
                continue
            base=matrices[b.parent.name]@relative[b.name] if b.parent else relative[b.name].copy()
            q=(base.inverted()@matrices[b.name]).to_quaternion()
            if b.name in previous and q.dot(previous[b.name])<0:
                q.negate()
            previous[b.name]=q.copy()
            pb=rig.pose.bones[b.name]
            pb.rotation_quaternion=q
            pb.keyframe_insert('rotation_quaternion',frame=f+1,group=b.name)
        root = rig.pose.bones['全ての親']
        root.location = bone_rest[root.name].to_3x3().inverted() @ Vector(lift if world_mode else (0, 0, lift))
        root.keyframe_insert('location', frame=f + 1, group=root.name)
        expected.append({n: list(matrices[n].translation) for n in mapping})
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for fc in bag.fcurves:
                    for point in fc.keyframe_points:
                        point.interpolation = 'LINEAR'
    if world_contact_enabled and motion.get('floor_collision',{}).get('enabled'):
        from motionbridge.blender_feet import foot_vertex_groups,evaluated_foot_minimum
        from motionbridge.floor_collision import lift_envelope
        foot_samples=foot_vertex_groups(scene,rig);before_floor=[];between_floor=[]
        for f in range(1,len(poses)+1):
            scene.frame_set(f)
            before_floor.append(evaluated_foot_minimum(foot_samples,bpy.context.evaluated_depsgraph_get()))
            if f<len(poses):
                scene.frame_set(f,subframe=.5)
                between_floor.append(evaluated_foot_minimum(foot_samples,bpy.context.evaluated_depsgraph_get()))
        # Leg IK has already run. Resolve its residual on actual deformed shoes,
        # using only positive root translation and a velocity-bounded envelope.
        needed=np.maximum(.001-np.asarray(before_floor),0)
        for f,z in enumerate(between_floor):needed[f:f+2]=np.maximum(needed[f:f+2],.001-z)
        collision_lift=lift_envelope(needed,fps,.35)
        lifts=np.asarray(lifts).copy();lifts[:,2]+=collision_lift;root_offsets=lifts.copy()
        for f,delta in enumerate(collision_lift):
            if delta<=0:continue
            scene.frame_set(f+1)
            root.location=bone_rest[root.name].to_3x3().inverted()@Vector(lifts[f])
            root.keyframe_insert('location',frame=f+1,group=root.name)
            for point in expected[f].values():point[2]+=float(delta)
        report['target_mesh_floor_collision']={'enabled':True,'method':'evaluated_shoe_unilateral_lift_v1',
            'floor_z_m':0.,'clearance_m':.001,'before_min_z_m':min(before_floor),
            'before_half_frame_min_z_m':min(between_floor) if between_floor else None,
            'before_below_floor_frames':sum(z<0 for z in before_floor),
            'root_lift_m':collision_lift.tolist(),'max_root_lift_m':float(collision_lift.max()),
            'foot_vertex_counts':{obj.name:len(indices) for obj,indices in foot_samples},
            'note':'Actual weighted shoe geometry. Positive residual root lift after IK; flat stage only, no body/cloth collision.'}
        # New keys retain linear interpolation, including fractional-frame audit.
        for layer in action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    for fc in bag.fcurves:
                        for point in fc.keyframe_points:point.interpolation='LINEAR'
    if surface_enabled:
        from motionbridge.blender_surfaces import bake_surface_clearance
        report['surface_clearance']=bake_surface_clearance(scene,rig,garment_objects,reference_path=output/'surface_solver_reference.npz')
        report['limitations'][3]='Embedded bounded hair/garment shapes against body capsules and actual triangles/layer order; residuals and fallback reported. No cloth dynamics, same-component self-collision or continuous collision guarantee.'
    # All-frame Blender evaluation catches dependency, quaternion and hierarchy errors.
    maximum = 0.
    for f, positions in enumerate(expected, 1):
        scene.frame_set(f)
        for n, position in positions.items():
            error = (rig.pose.bones[n].head - Vector(position)).length
            if not math.isfinite(error):
                raise ValueError('Nonfinite target pose')
            maximum = max(maximum, error)
    if maximum > .0001:
        raise ValueError('Evaluated rig does not match baked target: ' + str(maximum))
    # Import actual app-produced FBX as an independently inspectable reference.
    before = set(scene.objects)
    bpy.ops.import_scene.fbx(filepath=str(motion_path.with_suffix('.fbx')), use_anim=True, anim_offset=0)
    reference = bpy.data.collections.new('MotionBridge source FBX - hidden reference')
    scene.collection.children.link(reference)
    for obj in set(scene.objects) - before:
        for collection in list(obj.users_collection):
            collection.objects.unlink(obj)
        reference.objects.link(obj)
    reference.hide_render = True
    reference.hide_viewport = True
    # Fixed front camera lets the user inspect movement without old camera animation.
    bpy.ops.object.camera_add(location=(0, -4., .65))
    camera = bpy.context.object
    camera.name = 'MotionBridge dance comparison camera'
    camera.rotation_euler = (Vector((0, 0, .64)) - camera.location).to_track_quat('-Z', 'Y').to_euler()
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 1.6
    if world_mode:
        cloud = np.array([p for record in expected for p in record.values()])
        lo, hi = cloud.min(axis=0), cloud.max(axis=0)
        center = Vector(((lo+hi)/2).tolist())
        heading = C @ world[0,index['Hips']] @ np.array([0.,0.,1.])
        front = Vector((float(heading[0]),float(heading[1]),0.)).normalized()
        camera.location = center + front*4 + Vector((0,0,.08))
        camera.rotation_euler = (center-camera.location).to_track_quat('-Z','Y').to_euler()
        camera.data.ortho_scale = max(1.6, float(hi[2]-lo[2])*1.5,
                                     float(np.linalg.norm(hi[:2]-lo[:2]))*720/540*1.4)
    scene.camera = camera
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 8
    scene.cycles.use_denoising = False
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 4
    scene.render.resolution_x, scene.render.resolution_y = 540, 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.use_freestyle = False
    scene.render.use_compositing = False
    scene['MotionBridge_motion_source'] = str(motion_path)
    scene['MotionBridge_limitations'] = '\n'.join(report['limitations'])
    scene.frame_set(min(len(poses), round(4 * fps) + 1))
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces.active.region_3d.view_perspective = 'CAMERA'
                area.spaces.active.shading.type = 'MATERIAL'
    report.update({'frames': len(poses), 'fps': fps, 'mapped_bones': mapping,
                   'evaluated_joint_error_max_m': maximum,
                   'support_lift_range_m': [float(np.min(np.asarray(lifts)[...,2] if world_mode else lifts)),
                                            float(np.max(np.asarray(lifts)[...,2] if world_mode else lifts))],
                   'original_sha256_after': digest(source)})
    report['root_mode'] = motion.get('root_mode','in_place')
    if world_mode:
        report['expected_root_positions'] = (root_offsets + np.array(bone_rest['全ての親'].translation)).tolist()
    report['target_endpoint_ik']={'enabled':enhanced,'method':'face_reference_clavicle_contact_ik_v2' if calibrated else 'head_bimanual_relative_and_contact_ik_v1',
        'mean_hand_goal_residual_before_rate_limit_m':float(np.mean(endpoint_errors)) if endpoint_errors else None,
        'rate_limited_samples':ik_rate_limited_samples,
        'note':'Target-proportion fitting, not source capture accuracy. Unreachable targets clamp without stretching.'}
    report['world_contact_ik']={'enabled':world_contact_enabled,
        'note':'Bounded target leg IK during source-estimated support. Horizontal root is the scaled source track; optional target_ground_fit reports sole baseline corrections.'}
    if target_contact_points:
        points=np.asarray(target_contact_points)
        steps=np.linalg.norm(np.diff(points[:,:,[0,1]],axis=0),axis=-1)
        core=(contact_weights[1:]>=.999)&(contact_weights[:-1]>=.999)
        report['target_endpoint_ik']['mean_support_horizontal_step_m']=float(steps[core].mean()) if core.any() else None
    report['original_unchanged'] = report['original_sha256_before'] == report['original_sha256_after']
    blend = output / 'Nahida_MotionBridge_Dance.blend'
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(blend),compress=True)
    report['output'] = str(blend)
    report['preview_frame'] = scene.frame_current
    report['preview_time_seconds'] = (scene.frame_current - 1) / fps
    (output / 'retarget_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('RETARGET_SAVED', blend, 'evaluation_error', maximum, flush=True)
    scene.render.filepath = str(output / 'character_preview.png')
    bpy.ops.render.render(write_still=True)


if __name__ == '__main__':
    main()
