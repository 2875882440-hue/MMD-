"""Read source dependencies; never writes the input scene or runs simulation."""
from pathlib import Path
import argparse,hashlib,json,sys
import bpy
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from solve_secondary_hair_blender import digest_actions,mesh_digest
parser=argparse.ArgumentParser();parser.add_argument('--scene',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--purpose',choices=['simulate','export'],default='simulate');parser.add_argument('--profile',default='rin')
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);source=args.scene.resolve();out=args.output.resolve()
if not source.is_relative_to(ROOT) or not source.is_file() or not (out.is_relative_to(ROOT/'output/secondary_motion') or out.is_relative_to(ROOT/'validation/secondary-motion-dev5')) or out.exists():raise ValueError('Invalid preflight paths')
profile=json.loads((ROOT/f'profiles/secondary_motion/{args.profile}.json').read_text(encoding='utf-8'));bindings=profile['bindings']
digest=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
model=(ROOT/profile['model']['path']).resolve()
if not model.is_relative_to(ROOT) or not model.is_file() or digest(model)!=profile['model']['sha256']:raise ValueError('模型文件与配置指纹不符，拒绝复用旧缓存')
before=digest(source);bpy.ops.wm.open_mainfile(filepath=str(source),load_ui=False,use_scripts=False)
scene=bpy.context.scene;rig=bpy.data.objects.get(bindings['armature']);mesh=bpy.data.objects.get(bindings['mesh'])
if rig is None or mesh is None:raise ValueError('Source does not match configured character')
dependencies={};packed=0;unsupported=[]
def dependency(path):
 p=Path(bpy.path.abspath(path)).resolve()
 if not p.is_file():raise ValueError(f'Missing source dependency: {p}')
 dependencies[str(p)]=digest(p)
for image in bpy.data.images:
 if image.packed_file:packed+=1
 elif image.filepath:dependency(image.filepath)
for library in bpy.data.libraries:dependency(library.filepath)
for obj in scene.objects:
 for modifier in obj.modifiers:
  if modifier.type=='MESH_CACHE':dependency(modifier.filepath)
  if modifier.type in ('CLOTH','SOFT_BODY','FLUID','PARTICLE_SYSTEM'):unsupported.append(obj.name+':'+modifier.type)
for cache in bpy.data.cache_files:dependency(cache.filepath)
if unsupported:raise ValueError('Use a baked result, not live simulation modifiers: '+','.join(unsupported))
if args.purpose=='simulate' and mesh.modifiers.get('MB4_DynamicSurfaceCache'):raise ValueError('This is already a surface result; use its original motion input to recompute physics')
quality={'status':'not_certified','collision_quality_passed':None,'source_report':None}
for folder in [source.parent,source.parent.parent]:
 path=folder/'surface-audit.json'
 if path.is_file():
  report=json.loads(path.read_text(encoding='utf-8'));quality={'status':'experimental_rejected' if report.get('collision_quality_passed') is False else 'not_certified','collision_quality_passed':report.get('collision_quality_passed'),'source_report':str(path)};break
scene.frame_set(scene.frame_start)
result={'source':str(source),'source_sha256':before,'dependencies':dependencies,'packed_images':packed,'libraries':len(bpy.data.libraries),
 'blender_version':bpy.app.version_string,'blender_build':bpy.app.build_hash.decode(),'frame_start':scene.frame_start,'frame_end':scene.frame_end,'fps':scene.render.fps/scene.render.fps_base,
 'source_actions':digest_actions(),'skin_morph_sha256':mesh_digest(mesh),'shape_keys':len(mesh.data.shape_keys.key_blocks) if mesh.data.shape_keys else 0,
 'rigid_object_count':sum(o.type=='MESH' and o.name.startswith('MB4_RigidAttachment_') for o in scene.objects),'physics_quality':quality,'alembic_available':bool(getattr(bpy.app.build_options,'alembic',False))}
assert digest(source)==before
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('SECONDARY_PREFLIGHT_PASSED',result['blender_version'],result['packed_images'],len(dependencies),flush=True)
