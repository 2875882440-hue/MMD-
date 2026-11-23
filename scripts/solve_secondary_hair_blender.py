"""Hair-only, force-driven spring solve and saved-scene acceptance audit.

New output directories only. Source PMX, source scene, body actions, skin and
morph data are never overwritten. No inference or body-retarget code is called.
"""
from pathlib import Path
import argparse,hashlib,json,math,sys,time
import bpy
import numpy as np
from mathutils import Matrix,Quaternion,Vector,Euler

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from motionbridge.secondary_plan import compile_profile
from motionbridge.secondary_motion import _inside
from motionbridge.secondary_hair import HairSolver,select_hair,proxy_contacts,matrix_euler,orthonormal
from bind_secondary_model_blender import validate_geometry,mesh_digest


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def digest_actions(names=None):
    result={}
    for action in bpy.data.actions:
        if names is not None and action.name not in names:continue
        curves=[]
        if hasattr(action,'fcurves'):curves.extend(action.fcurves)
        for layer in action.layers:
            for strip in layer.strips:
                for bag in getattr(strip,'channelbags',[]):curves.extend(bag.fcurves)
        rows=sorted((f.data_path,f.array_index,[(tuple(k.co),tuple(k.handle_left),tuple(k.handle_right),
                     k.interpolation,k.easing,k.type) for k in f.keyframe_points]) for f in curves)
        result[action.name]=hashlib.sha256(json.dumps(rows).encode()).hexdigest()
    return result


def set_frame(frame):
    base=math.floor(frame)
    bpy.context.scene.frame_set(base,subframe=frame-base)


def protected_matrices(rig,names,frames):
    arrays=[]
    for frame in frames:
        set_frame(frame)
        arrays.append(np.asarray([rig.matrix_world@rig.pose.bones[name].matrix for name in names],dtype=np.float32))
    return np.stack(arrays)


def camera_setup(scene):
    data=bpy.data.cameras.new('MBH_ComparisonCamera')
    camera=bpy.data.objects.new('MBH_ComparisonCamera',data);scene.collection.objects.link(camera)
    camera.location=(1.7,4.5,1.25)
    target=Vector((0,0,.86))
    camera.rotation_euler=(target-camera.location).to_track_quat('-Z','Y').to_euler()
    data.type='ORTHO';data.ortho_scale=1.95;scene.camera=camera
    scene.render.engine='BLENDER_WORKBENCH'
    scene.display.shading.light='STUDIO';scene.display.shading.color_type='TEXTURE'
    scene.display.shading.show_shadows=True;scene.display.shading.show_cavity=True
    scene.display.shading.background_type='WORLD'
    if scene.world is None:scene.world=bpy.data.worlds.new('MBH_ComparisonWorld')
    scene.world.color=(.07,.07,.07)
    scene.render.resolution_x=640;scene.render.resolution_y=800;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG'
    return {'location':list(camera.location),'rotation':list(camera.rotation_euler),
            'type':data.type,'ortho_scale':data.ortho_scale,'resolution':[640,800]}


def make_nodes(plan,ids,rig,settings):
    lookup={bone:i for i,bone in enumerate(ids)};by_bone={}
    for rigid in plan['rigid_bodies']:
        if rigid['bone_id'] in lookup:
            if rigid['bone_id'] in by_bone:raise ValueError('发束骨骼存在多个刚体，需专用适配')
            by_bone[rigid['bone_id']]=rigid
    nodes=[];used_joints=set()
    for bone in ids:
        entry=plan['bones'][bone];name=entry['blender_name'];rest=rig.data.bones[name]
        parent=rest.parent
        if parent is None:raise ValueError('发束缺少固定父级')
        parent_id=next((b['id'] for b in plan['bones'] if b['blender_name']==parent.name),None)
        if parent_id != entry['parent']:raise ValueError('Blender发束父级与PMX不同，拒绝推测适配')
        parent_index=lookup.get(parent_id,-1)
        if parent_index>=lookup[bone]:raise ValueError('发束未按父子顺序排列')
        rigid=by_bone[bone]
        candidates=[]
        for joint in plan['joints']:
            ends=joint['rigid_ids']
            if rigid['id'] not in ends:continue
            other=ends[1] if ends[0]==rigid['id'] else ends[0]
            if other is not None and plan['rigid_bodies'][other]['bone_id']==parent_id:
                candidates.append(joint)
        if not candidates:
            # Rin's right second ponytail is parented to the head in the
            # skeleton but its native angular joint follows the first ponytail.
            # Keep its skeleton head attachment and use that explicit joint's
            # physical parent for orientation, without translating body/bones.
            candidates=[j for j in plan['joints'] if j['rigid_ids'][1]==rigid['id']
                        and j['rigid_ids'][0] is not None
                        and plan['rigid_bodies'][j['rigid_ids'][0]]['bone_id'] in lookup
                        and lookup[plan['rigid_bodies'][j['rigid_ids'][0]]['bone_id']] < lookup[bone]]
        if len(candidates)!=1:raise ValueError(f'发束父级原生关节缺失或歧义: {name}')
        joint=candidates[0];used_joints.add(joint['id'])
        other=next(index for index in joint['rigid_ids'] if index!=rigid['id'])
        angular_parent_id=plan['rigid_bodies'][other]['bone_id']
        angular_parent_name=plan['bones'][angular_parent_id]['blender_name']
        angular_parent=rig.data.bones[angular_parent_name]
        angular_local=angular_parent.matrix_local.inverted()@rest.matrix_local
        local=parent.matrix_local.inverted()@rest.matrix_local
        joint_rotation=Euler(joint['blender']['rotation_rad'],joint['blender']['rotation_order']).to_matrix()
        frame=angular_parent.matrix_local.to_3x3().inverted()@joint_rotation
        shape=rigid['blender']['shape']
        radius=shape.get('radius_m',min(shape.get('half_extents_m',[.01]*3)))
        nodes.append({'bone_id':bone,'name':name,'parent':parent_index,'external_parent':parent.name,
                      'angular_parent':lookup.get(angular_parent_id,-1),'angular_external_parent':angular_parent_name,
                      'angular_local_rotation':[list(row) for row in angular_local.to_3x3()],
                      'local_head':list(local.translation),'local_rotation':[list(row) for row in local.to_3x3()],
                      'joint_frame_parent':[list(row) for row in frame], 'length':rest.length,
                      'angular_min':joint['blender']['angular_min_rad'],'angular_max':joint['blender']['angular_max_rad'],
                      'radius':radius*settings['strand_radius_scale'],'native_rigid_id':rigid['id'],'native_joint_id':joint['id']})
    ignored=[j['id'] for j in plan['joints'] if j['id'] not in used_joints and any(
        i is not None and plan['rigid_bodies'][i]['bone_id'] in lookup for i in j['rigid_ids'])]
    return nodes,ignored


def body_specs(plan,rig,settings,hair_ids):
    specs=[]
    for index in settings['body_collider_ids']:
        row=plan['rigid_bodies'][index]
        if row['native']['mode'] != 0 or row['bone_id'] is None or row['bone_id'] in hair_ids:
            raise ValueError('身体碰撞代理必须绑定原生运动学身体骨骼')
        bone=plan['bones'][row['bone_id']]['blender_name']
        p=row['blender'];shape=p['shape']
        matrix=Euler(p['rotation_rad'],p['rotation_order']).to_matrix().to_4x4()
        matrix.translation=p['position_m']
        offset=rig.data.bones[bone].matrix_local.inverted()@matrix
        kind={'sphere':0,'box':1,'capsule':2}[shape['kind']]
        size=shape['half_extents_m'] if kind==1 else [shape['radius_m'],shape.get('cylinder_height_m',0),0]
        specs.append({'native_rigid_id':index,'bone':bone,'offset':np.asarray(offset), 'kind':kind,'size':size})
    if not specs:raise ValueError('必须配置身体碰撞代理')
    return specs


def capture_drivers(rig,nodes,specs,frames):
    names=sorted({n['external_parent'] for n in nodes if n['parent']<0}|{s['bone'] for s in specs})
    driver={name:[] for name in names};hair=[]
    for frame in frames:
        set_frame(frame)
        for name in names:
            mat=rig.matrix_world@rig.pose.bones[name].matrix
            # Remove float orthogonality drift, preserving source translation.
            normalized=mat.to_quaternion().normalized().to_matrix().to_4x4();normalized.translation=mat.translation
            driver[name].append(np.asarray(normalized))
        hair.append(np.asarray([rig.matrix_world@rig.pose.bones[n['name']].matrix for n in nodes]))
    return {name:np.asarray(rows) for name,rows in driver.items()},np.asarray(hair)


def interpolate_drivers(drivers,frame_index,fraction):
    result={}
    previous=max(0,frame_index-1)
    for name,values in drivers.items():
        first=Matrix(values[previous]);second=Matrix(values[frame_index])
        rot=first.to_quaternion().slerp(second.to_quaternion(),fraction).to_matrix().to_4x4()
        rot.translation=first.translation.lerp(second.translation,fraction)
        result[name]=np.asarray(rot)
    return result


def colliders_at(specs,drivers):
    matrices=np.asarray([drivers[s['bone']]@s['offset'] for s in specs])
    return dict(centers=matrices[:,:3,3],rotations=matrices[:,:3,:3],
                kinds=np.array([s['kind'] for s in specs]),sizes=np.array([s['size'] for s in specs]))


def run_solver(nodes,settings,drivers,specs,fps,out):
    solver=HairSolver(nodes,settings);substeps=settings['substeps'];dt=1/fps/substeps
    fixed=interpolate_drivers(drivers,0,1);body=colliders_at(specs,fixed)
    warmup_steps=math.ceil(settings['warmup_seconds']/dt)
    for _ in range(warmup_steps):solver.step(fixed,body,dt)
    warmup_speed=np.linalg.norm(solver.velocities,axis=1)
    write(out/'warmup.json',{'steps':warmup_steps,'seconds':settings['warmup_seconds'],
        'tip_speed_p95_m_s':float(np.percentile(warmup_speed,95)),'tip_speed_max_m_s':float(warmup_speed.max())})
    if warmup_speed.max()>.04:
        raise ValueError('静止预热未收敛（最大末端速度>0.04m/s），拒绝发布该配置')
    heads=[];rotations=[];tails=[];angles=[]
    start=time.perf_counter();count=len(next(iter(drivers.values())))
    for frame in range(count):
        for substep in range(substeps):
            driver=interpolate_drivers(drivers,frame,(substep+1)/substeps)
            h,r,t,a=solver.step(driver,colliders_at(specs,driver),dt)
        heads.append(h);rotations.append(r);tails.append(t);angles.append(a)
        if frame%90==0:
            write(out/'solver-progress.json',{'frames':frame+1,'total':count,'seconds':time.perf_counter()-start})
            print('HAIR_SOLVE_PROGRESS',frame+1,count,round(time.perf_counter()-start,2),flush=True)
    arrays=dict(heads=np.asarray(heads),rotations=np.asarray(rotations),tails=np.asarray(tails),angles=np.asarray(angles))
    np.savez_compressed(out/'hair-simulation.npz',**arrays)
    return arrays,{'seconds':time.perf_counter()-start,'warmup_steps':warmup_steps,
        'warmup_tip_speed_p95_m_s':float(np.percentile(warmup_speed,95)),
        'warmup_tip_speed_max_m_s':float(warmup_speed.max()),'gravity_m_s2':settings['gravity_m_s2'],
        'force_integrations':warmup_steps+count*substeps,'contacts_applied':solver.contacts_applied,
        'contact_worsening_rollbacks':solver.contact_rollbacks,
        'maximum_length_error_m':solver.maximum_length_error,'maximum_angular_limit_error_rad':solver.maximum_limit_error}


def create_physics_actions(rig,nodes,arrays,frames):
    collection=bpy.data.collections.new('MBH_HairDynamics');bpy.context.scene.collection.children.link(collection)
    for index,node in enumerate(nodes):
        bone=rig.pose.bones[node['name']]
        if any(c.name.startswith('MBH_') for c in bone.constraints):raise ValueError('已存在头发物理，拒绝重复叠加')
        target=bpy.data.objects.new(f'MBH_Target_{node["bone_id"]}',None);collection.objects.link(target)
        target.rotation_mode='QUATERNION';target.empty_display_size=.01;target.hide_render=True
        action=bpy.data.actions.new(f'MBH_Rotation_{node["bone_id"]}')
        slot=action.slots.new(id_type='OBJECT',name=target.name)
        layer=action.layers.new('Hair dynamics');strip=layer.strips.new(type='KEYFRAME');bag=strip.channelbags.new(slot)
        rotations=[];last=None
        for matrix in arrays['rotations'][:,index]:
            q=Matrix(matrix).to_quaternion().normalized()
            if last is not None and q.dot(last)<0:q.negate()
            rotations.append(list(q));last=q
        for component in range(4):
            curve=bag.fcurves.new(data_path='rotation_quaternion',index=component)
            curve.keyframe_points.add(len(frames))
            curve.keyframe_points.foreach_set('co',np.column_stack((frames,np.asarray(rotations)[:,component])).ravel())
            for key in curve.keyframe_points:key.interpolation='LINEAR'
            curve.update()
        target.animation_data_create();target.animation_data.action=action;target.animation_data.action_slot=slot
        constraint=bone.constraints.new('COPY_ROTATION');constraint.name='MBH_HairDynamics'
        constraint.target=target;constraint.owner_space='WORLD';constraint.target_space='WORLD';constraint.mix_mode='REPLACE'
    rig['motionbridge_hair_state']='experimental_force_driven_hair_only'


def proxy_metrics(heads,tails,nodes,specs,drivers):
    count=0;frames=0;worst=0.;total=0.
    fractions=np.array([.35,.65,1.])
    radii=np.repeat(np.array([n['radius'] for n in nodes]),len(fractions))
    for index,(h,t) in enumerate(zip(heads,tails)):
        points=(h[:,None,:]+fractions[None,:,None]*(t-h)[:,None,:]).reshape(-1,3)
        body=colliders_at(specs,{name:array[index] for name,array in drivers.items()})
        distance,_=proxy_contacts(points,**body)
        depth=np.maximum(radii-distance.min(axis=1),0)
        count+=int(np.sum(depth>.001));frames+=int(np.any(depth>.001));worst=max(worst,float(depth.max()));total+=float(depth.sum())
    return {'samples_deeper_than_1mm':count,'frames_deeper_than_1mm':frames,'maximum_depth_m':worst,
            'summed_depth_m':total,'method':'3 segment samples vs 20 authored moving body proxies, no mesh guarantee'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--profile',default='rin');parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--enable-hair',action='store_true')
    parser.add_argument('--frame-limit',type=int)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    if not args.enable_hair:raise ValueError('必须显式--enable-hair；当前为实验求解器')
    out=args.output_dir.resolve()
    if not out.is_relative_to(ROOT/'output/secondary_motion') or out.exists():raise ValueError('请使用本副本output/secondary_motion内的新子目录')
    if not args.profile.isascii() or not args.profile.replace('_','').replace('-','').isalnum():raise ValueError('非法角色标识')
    source=_inside(ROOT,args.scene);source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    profile=json.loads((ROOT/f'profiles/secondary_motion/{args.profile}.json').read_text(encoding='utf-8'))
    plan=compile_profile(profile,ROOT);settings=profile['hair_dynamics'];ids=select_hair(plan,settings)
    bpy.ops.wm.open_mainfile(filepath=str(source),load_ui=False,use_scripts=False)
    scene=bpy.context.scene
    if scene.rigidbody_world and scene.rigidbody_world.enabled:raise ValueError('源工程已启用其他仿真，拒绝混算')
    rig=bpy.data.objects[plan['bindings']['armature']];mesh=bpy.data.objects[plan['bindings']['mesh']]
    geometry=validate_geometry(mesh,rig,plan)
    nodes,ignored_joints=make_nodes(plan,ids,rig,settings);specs=body_specs(plan,rig,settings,ids)
    frames=list(range(scene.frame_start,scene.frame_end+1))
    if args.frame_limit:frames=frames[:args.frame_limit];scene.frame_end=frames[-1]
    audit_frames=sorted(set(frames+[f+.5 for f in frames[:-1]]))
    names=[plan['bones'][i]['blender_name'] for i in plan['protected_bone_ids']]
    source_actions=digest_actions()
    # Do not silently replace pre-existing hair animation channels.
    for action in bpy.data.actions:
        for layer in action.layers:
            for strip in layer.strips:
                for bag in getattr(strip,'channelbags',[]):
                    for curve in bag.fcurves:
                        if any(f'pose.bones["{n["name"]}"]' in curve.data_path for n in nodes):
                            raise ValueError('源工程已有发束动画，需明确组合策略，当前拒绝覆盖')
    before_world=protected_matrices(rig,names,audit_frames)
    set_frame(frames[0]);source_mesh=mesh_digest(mesh)
    drivers,source_hair=capture_drivers(rig,nodes,specs,frames)
    # Current adapter supports unanimated native rest-local hair chains only.
    for f in range(len(frames)):
        for i,node in enumerate(nodes):
            parent=(source_hair[f,node['parent']] if node['parent']>=0 else drivers[node['external_parent']][f])
            local=np.linalg.inv(parent)@source_hair[f,i]
            expected=np.eye(4);expected[:3,:3]=node['local_rotation'];expected[:3,3]=node['local_head']
            if np.max(np.abs(local-expected))>2e-5:raise ValueError(f'发束存在未适配的动画/约束: {node["name"]}')
    out.mkdir(parents=True);write(out/'model-plan.json',plan);write(out/'hair-nodes.json',nodes)
    write(out/'profile.json',profile);camera=camera_setup(scene);write(out/'camera.json',camera)
    set_frame(frames[0]);bpy.ops.wm.save_as_mainfile(filepath=str(out/'before.blend'))
    arrays,solve_metrics=run_solver(nodes,settings,drivers,specs,scene.render.fps/scene.render.fps_base,out)
    create_physics_actions(rig,nodes,arrays,frames)
    set_frame(frames[0]);bpy.ops.wm.save_as_mainfile(filepath=str(out/'after.blend'))
    bpy.ops.wm.open_mainfile(filepath=str(out/'after.blend'),load_ui=False,use_scripts=False)
    rig=bpy.data.objects[plan['bindings']['armature']];mesh=bpy.data.objects[plan['bindings']['mesh']]
    after_world=protected_matrices(rig,names,audit_frames)
    if not np.array_equal(before_world,after_world):raise AssertionError(f'主动作世界矩阵变化: {np.max(np.abs(before_world-after_world))}')
    if digest_actions(set(source_actions))!=source_actions:raise AssertionError('原动作曲线变化')
    set_frame(frames[0])
    if mesh_digest(mesh)!=source_mesh:raise AssertionError('原蒙皮或形态键变化')
    actual_heads=[];actual_tails=[];angle_change=[];position_error=[];actual_angular_error=[]
    for index,frame in enumerate(frames):
        set_frame(frame);h=[];t=[]
        for i,node in enumerate(nodes):
            bone=rig.pose.bones[node['name']];matrix=rig.matrix_world@bone.matrix
            h.append(list(matrix.translation));t.append(list(rig.matrix_world@bone.tail))
            difference=matrix.to_quaternion().rotation_difference(Matrix(source_hair[index,i]).to_quaternion()).angle
            angle_change.append(min(difference,2*math.pi-difference))
            position_error.append(np.max(np.abs(np.asarray(matrix.to_3x3())-arrays['rotations'][index,i])))
            parent_name=(nodes[node['angular_parent']]['name'] if node['angular_parent']>=0 else node['angular_external_parent'])
            parent_rotation=orthonormal(np.asarray((rig.matrix_world@rig.pose.bones[parent_name].matrix).to_3x3()))
            joint=parent_rotation@orthonormal(np.asarray(node['joint_frame_parent']))
            reference=parent_rotation@orthonormal(np.asarray(node['angular_local_rotation']))
            measured=matrix_euler(joint.T@orthonormal(np.asarray(matrix.to_3x3()))@reference.T@joint)
            actual_angular_error.append(float(np.maximum(np.asarray(node['angular_min'])-measured,
                                                         measured-np.asarray(node['angular_max'])).max()))
        actual_heads.append(h);actual_tails.append(t)
    actual_heads=np.asarray(actual_heads);actual_tails=np.asarray(actual_tails)
    source_heads=source_hair[:,:,:3,3]
    source_tails=source_heads+source_hair[:,:,:3,1]*np.array([n['length'] for n in nodes])[None,:,None]
    displacement=np.linalg.norm(actual_tails-source_tails,axis=2)
    if displacement.max()<.001:raise AssertionError('施力后未产生可测发束变化')
    if max(position_error)>2e-5:raise AssertionError('保存回读发束旋转未匹配求解结果')
    if max(actual_angular_error)>2e-5:raise AssertionError('保存回读后发束角限位越界')
    lengths=np.linalg.norm(actual_tails-actual_heads,axis=2)
    length_error=float(np.max(np.abs(lengths-np.array([n['length'] for n in nodes]))))
    if length_error>2e-6:raise AssertionError('真实骨段长度变化')
    assert source_sha==hashlib.sha256(source.read_bytes()).hexdigest()
    report={'passed':True,'phase':'hair_force_solve_saved_scene_verified','simulation_performed':True,
        'source_scene':str(source),'source_sha256':source_sha,'source_unchanged':True,
        'source_actions':source_actions,'source_actions_unchanged':True,
        'mesh_and_morphs_sha256':source_mesh,'skin_and_81_shape_keys_unchanged':True,
        'protected_bones':len(names),'integer_frames':len(frames),'integer_and_half_frames':len(audit_frames),
        'world_before_sha256':hashlib.sha256(before_world.tobytes()).hexdigest(),
        'world_after_sha256':hashlib.sha256(after_world.tobytes()).hexdigest(),'protected_max_matrix_error':0.,
        'hair_bones':len(nodes),'body_proxies':len(specs),'solver':solve_metrics,
        'actual_length_max_error_m':length_error,'saved_rotation_max_error':max(position_error),
        'actual_angular_limit_max_error_rad':max(0.,max(actual_angular_error)),
        'hair_tip_change_max_m':float(displacement.max()),'hair_tip_change_p95_m':float(np.percentile(displacement,95)),
        'hair_global_rotation_change_max_deg':math.degrees(max(angle_change)),
        'proxy_contacts_before':proxy_metrics(source_heads,source_tails,nodes,specs,drivers),
        'proxy_contacts_after':proxy_metrics(actual_heads,actual_tails,nodes,specs,drivers),
        'native_cross_or_duplicate_joints_not_solved':ignored_joints,'camera':camera,'geometry':geometry,
        'limitations':['实验性的单向弹簧骨链，不是PMX完整六自由度/Bullet仿真；没有身体反作用力。',
                       '只解父级发束关节角限位及固定长度；跨束/重复关节、扭转弹簧和头发自碰撞未解。',
                       '身体碰撞是显式代理，保留原掩码并对本轮发束/身体覆盖；不保证真实发丝网格或衣物无穿插。',
                       '角度/长度约束优先，遇到碰撞不可兼容会残留，报告保留实际代理穿入数。',
                       '只验证Rin原生静态发束与已有舞蹈；其他角色、已有发束动画和交互式实时播放尚未适配。',
                       '未执行FBX导出；当前Blender保存独立发束旋转目标动作与约束。']}
    write(out/'physics-validation.json',report)
    print('HAIR_PHYSICS_PASSED',json.dumps({k:report[k] for k in ['hair_bones','integer_frames','protected_bones',
        'protected_max_matrix_error','hair_tip_change_max_m','hair_tip_change_p95_m','proxy_contacts_before','proxy_contacts_after']}),flush=True)


if __name__=='__main__':main()
