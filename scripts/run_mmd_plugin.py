"""Rin MMD Tools adapter. Keep artist-authored physics in the isolated plugin process."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from motionbridge.secondary_plan import compile_profile
from motionbridge.secondary_surface import MDDWriter
from motionbridge.mmd_hair_cache import stabilize
from solve_secondary_hair_blender import digest_actions, mesh_digest
from bind_secondary_model_blender import validate_geometry

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')

def at(scene, frame):
    scene.frame_set(int(frame), subframe=float(frame % 1))

def matrices(rig, names):
    return np.array([rig.matrix_world @ rig.pose.bones[n].matrix for n in names], dtype=np.float32)

def coordinates(mesh):
    ev = mesh.evaluated_get(bpy.context.evaluated_depsgraph_get())
    result = np.empty((len(ev.data.vertices), 3), dtype=np.float32)
    ev.data.vertices.foreach_get('co', result.reshape(-1))
    return result

def simulate(scene, src_rig, src_mesh, profile, names, frames, reference, path, fps):
    simulation = bpy.data.scenes.new('MotionBridge PMX simulation')
    bpy.context.window.scene = simulation
    simulation.render.fps = scene.render.fps
    simulation.render.fps_base = scene.render.fps_base
    simulation.frame_start, simulation.frame_end = scene.frame_start, scene.frame_end
    at(simulation, frames[0])
    previous = set(bpy.data.objects)
    pmx = (ROOT/profile['model']['path']).resolve()
    if not pmx.is_relative_to(ROOT) or sha(pmx) != profile['model']['sha256']:
        raise ValueError('PMX 指纹与角色配置不匹配')
    if bpy.ops.mmd_tools.import_model(filepath=str(pmx), scale=.08, types={'MESH','ARMATURE','MORPHS','PHYSICS'},
            clean_model=False, remove_doubles=False, rename_bones=False) != {'FINISHED'}:
        raise ValueError('MMD Tools 未完成 PMX 导入')
    actors = set(bpy.data.objects) - previous
    rig, = [o for o in actors if o.type == 'ARMATURE']
    mesh, = [o for o in actors if o.type == 'MESH' and len(o.data.vertices) == len(src_mesh.data.vertices)]
    if not np.array_equal([v.co[:] for v in mesh.data.vertices], [v.co[:] for v in src_mesh.data.vertices]) or not np.array_equal(
            [p.vertices[:] for p in mesh.data.polygons], [p.vertices[:] for p in src_mesh.data.polygons]):
        raise ValueError('插件模型与角色工程的顶点或面顺序不同')
    anim = src_rig.animation_data
    if not anim or not anim.action or any(not track.mute for track in anim.nla_tracks):
        raise ValueError('需要单一活动角色动作')
    for pb in rig.pose.bones:
        original = src_rig.pose.bones.get(pb.name)
        if original:
            pb.rotation_mode = original.rotation_mode
            for con in pb.constraints:
                other = original.constraints.get(con.name)
                if other:
                    con.mute, con.influence = other.mute, other.influence
    rig.animation_data_create()
    rig.animation_data.action = anim.action
    rig.animation_data.action_slot = anim.action_slot
    world = simulation.rigidbody_world
    if not world:
        raise ValueError('角色没有作者物理')
    world.enabled = False
    for i, frame in enumerate(frames):
        at(simulation, frame)
        if not np.array_equal(reference[i], matrices(rig,names)):
            raise ValueError(f'加载动作时主骨不一致，第 {frame:g} 帧')
    at(simulation,frames[0])
    world.point_cache.frame_end = scene.frame_end
    world.point_cache.frame_start = scene.frame_start
    bpy.context.view_layer.objects.active = rig.parent
    if bpy.ops.mmd_tools.build_rig() != {'FINISHED'} or not world.enabled:
        raise ValueError('MMD Tools 未能建立物理')
    world.point_cache.frame_end = scene.frame_end
    world.point_cache.frame_start = scene.frame_start
    cache = np.lib.format.open_memmap(path,mode='w+',shape=(len(frames),len(mesh.data.vertices),3),dtype=np.float32)
    for i, frame in enumerate(frames):
        at(simulation,frame)
        if not np.array_equal(reference[i],matrices(rig,names)):
            raise ValueError(f'插件改变主骨，第 {frame:g} 帧')
        cache[i] = coordinates(mesh)
        if not np.isfinite(cache[i]).all():
            raise ValueError('插件返回非有限顶点坐标')
    cache.flush()
    return cache

def run(source, expected_sha, out, phase):
    if not source.is_relative_to(ROOT) or not out.is_relative_to(ROOT/'output/secondary_motion') or sha(source)!=expected_sha:
        raise ValueError('工程路径或源指纹不匹配')
    modules = [add.module for add in bpy.context.preferences.addons if add.module=='bl_ext.user_default.mmd_tools']
    if bpy.app.version[:2] != (5,2) or not modules:
        raise ValueError('需要在独立的 Blender 5.2 环境启用 MMD Tools')
    if phase=='bake':
        out.mkdir(parents=True,exist_ok=False)
    elif not (out/'baked.json').is_file():
        raise ValueError('缺少本次烘焙凭证')
    profile=json.loads((ROOT/'profiles/secondary_motion/rin.json').read_text(encoding='utf8'))
    plan=compile_profile(profile,ROOT)
    if phase=='bake':
        write(out/'profile.json',profile);write(out/'model-plan.json',plan)
    bpy.ops.wm.open_mainfile(filepath=str(source),load_ui=False,use_scripts=False)
    scene=bpy.context.scene;rig=scene.objects[plan['bindings']['armature']];mesh=scene.objects[plan['bindings']['mesh']]
    validate_geometry(mesh,rig,plan)
    if any(m.type=='MESH_CACHE' for m in mesh.modifiers) or not np.array_equal(mesh.matrix_world,np.eye(4)):
        raise ValueError('请使用未叠加顶点缓存的角色动作工程')
    start,end=scene.frame_start,scene.frame_end
    fps=scene.render.fps/scene.render.fps_base
    frames=np.arange(start,end+.25,.5)
    names=[plan['bones'][i]['blender_name'] for i in plan['protected_bone_ids']]
    if len(names)!=48 or len(frames)<1:raise ValueError('Rin 主骨或帧范围不匹配')
    at(scene,start);skin=mesh_digest(mesh);actions=digest_actions()
    fixed=np.unique([v for region in plan['regions'] if region['role'] in ('body','boots') for v in region['vertex_ids']])
    animated=np.setdiff1d(np.arange(len(mesh.data.vertices)),fixed)
    ref=[];fixed_hash=[]
    for frame in frames:
        at(scene,frame);ref.append(matrices(rig,names))
        fixed_hash.append(hashlib.sha256(coordinates(mesh)[fixed].tobytes()).digest())
    ref=np.stack(ref);at(scene,start)
    bpy.context.preferences.filepaths.save_version=0
    if phase=='bake':
        bpy.ops.wm.save_as_mainfile(filepath=str(out/'before.blend'))
        bake(out,scene,rig,mesh,profile,plan,names,frames,ref,fps,expected_sha)
        return
    receipt=json.loads((out/'baked.json').read_text(encoding='utf8'))
    if (receipt['source_sha256']!=expected_sha or receipt['mdd_sha256']!=sha(out/'surface-final.mdd')
            or receipt['before_sha256']!=sha(out/'before.blend')):
        raise ValueError('插件缓存或输入工程与烘焙凭证不符')
    apply_cache(source,expected_sha,out,plan,names,start,end,frames,actions,skin,animated,fixed,ref,fixed_hash)

def bake(out,scene,rig,mesh,profile,plan,names,frames,ref,fps,expected_sha):
    raw_path=out/'plugin-raw.npy'
    raw=simulate(scene,rig,mesh,profile,names,frames,ref,raw_path,fps)
    head_id=next(b['id'] for b in plan['bones'] if b['blender_name']=='頭')
    head=ref[:,names.index('頭')].astype(float)
    writer=MDDWriter(out/'surface-final.mdd',len(frames),len(mesh.data.vertices),fps*2)
    # Kernel and contact fade both need neighbors on each side of a block boundary.
    overlap=math.ceil(3*fps*2*.035)+math.ceil(fps*2*.1)+2
    for left in range(0,len(frames),64):
        right=min(len(frames),left+64);a=max(0,left-overlap);b=min(len(frames),right+overlap)
        ids,filtered,*_=stabilize(raw[a:b],head[a:b],plan,head_id,2*fps)
        for i in range(left,right):
            pts=raw[i].copy();pts[ids]=filtered[i-a];writer.append(pts)
    writer.close();del raw
    raw_path.unlink()
    write(out/'baked.json',dict(source_sha256=expected_sha,mdd_sha256=sha(out/'surface-final.mdd'),before_sha256=sha(out/'before.blend')))

def apply_cache(source,expected_sha,out,plan,names,start,end,frames,actions,skin,animated,fixed,ref,fixed_hash):
    bpy.ops.wm.open_mainfile(filepath=str(out/'before.blend'),load_ui=False,use_scripts=False)
    scene=bpy.context.scene;rig=scene.objects[plan['bindings']['armature']];mesh=scene.objects[plan['bindings']['mesh']]
    group=mesh.vertex_groups.new(name='MBP_MMDCache');group.add(animated.tolist(),1,'REPLACE')
    mod=mesh.modifiers.new('MB4_DynamicSurfaceCache','MESH_CACHE')
    mod.cache_format='MDD';mod.filepath='//surface-final.mdd';mod.vertex_group=group.name
    mod.frame_start=2*start;mod.frame_scale=2;mod.deform_mode='OVERWRITE';mod.interpolation='LINEAR'
    at(scene,start);bpy.ops.wm.save_as_mainfile(filepath=str(out/'after.blend'))
    bpy.ops.wm.open_mainfile(filepath=str(out/'after.blend'),load_ui=False,use_scripts=False)
    scene=bpy.context.scene;rig=scene.objects[plan['bindings']['armature']];mesh=scene.objects[plan['bindings']['mesh']]
    at(scene,start)
    if digest_actions(set(actions))!=actions or mesh_digest(mesh)!=skin:raise ValueError('保存回读后原曲线、蒙皮或形态键变化')
    mdd=np.memmap(out/'surface-final.mdd',mode='r',dtype='>f4',offset=8+4*len(frames),shape=(len(frames),len(mesh.data.vertices),3))
    maximum=0.
    for i,frame in enumerate(frames):
        at(scene,frame);pos=coordinates(mesh)
        if not np.array_equal(matrices(rig,names),ref[i]) or hashlib.sha256(pos[fixed].tobytes()).digest()!=fixed_hash[i]:
            raise ValueError(f'保存回读后身体或主骨变化，第 {frame:g} 帧')
        err=float(np.abs(pos[animated]-mdd[i,animated]).max());maximum=max(maximum,err)
        if not np.isfinite(err) or err>2e-6:raise ValueError(f'MDD 回读不符，第 {frame:g} 帧，误差 {err:g} m')
    if sha(source)!=expected_sha:raise ValueError('运行期间源工程变化')
    write(out/'physics-validation.json',dict(provider='mmd_tools',source_sha256=expected_sha,passed=True,
          protection_passed=True,collision_quality_passed=None,hair_stabilized=True,frames=end-start+1,
          samples=len(frames),protected_bones=len(names),protected_actions=len(actions),max_cache_error_m=maximum,
          blender=bpy.app.version_string))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--scene',type=Path,required=True);parser.add_argument('--source-sha',required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--phase',choices=['bake','apply'],required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    run(args.scene.resolve(),args.source_sha,args.out.resolve(),args.phase)
    # A one-shot bake owns the Bullet world. Blender 5.2.4 can dereference released
    # constraints at WM_exit. All files are closed before returning from run().
    # Only the fresh apply process can validate and publish a baked result.
    if args.phase == 'bake':
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
