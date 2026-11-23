"""Self-contained editable Blender snapshot; original curve/skin/key protection."""
from pathlib import Path
import argparse,hashlib,json,shutil,sys,time
import bpy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from solve_secondary_hair_blender import digest_actions,mesh_digest,protected_matrices
from motionbridge.blender_secondary_rigid import evaluated_mesh_records
parser=argparse.ArgumentParser();parser.add_argument('--scene',type=Path,required=True);parser.add_argument('--input-dir',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--take',type=int)
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);source=args.scene.resolve();case=args.input_dir.resolve();out=args.output_dir.resolve()
if not source.is_relative_to(ROOT) or not out.is_relative_to(ROOT/'output/secondary_motion') or out.exists():raise ValueError('New isolated package directory required')
plan=json.loads((case/'model-plan.json').read_text(encoding='utf-8'));names=[plan['bones'][i]['blender_name'] for i in plan['protected_bone_ids']]
digest=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
source_sha=digest(source);started=time.perf_counter();bpy.ops.wm.open_mainfile(filepath=str(source),load_ui=False,use_scripts=False)
scene=bpy.context.scene;start=scene.frame_start;end=min(scene.frame_end,start+args.take-1) if args.take else scene.frame_end;frames=np.arange(start,end+.01,.5).tolist()
rig=bpy.data.objects[plan['bindings']['armature']];mesh=bpy.data.objects[plan['bindings']['mesh']]
before=protected_matrices(rig,names,frames);scene.frame_set(start);actions=digest_actions();skin=mesh_digest(mesh);shape_count=len(mesh.data.shape_keys.key_blocks)
samples=sorted(set([float(start),(start+end)//2+.5 if start<end else float(start),float(end)]));geometry={}
def geometry_snapshot():
 result={}
 for frame in samples:
  scene.frame_set(int(frame),subframe=frame-int(frame));sha=hashlib.sha256()
  for obj in sorted([o for o in scene.objects if o.type=='MESH' and (o.name==mesh.name or o.name.startswith('MB4_RigidAttachment_'))],key=lambda o:o.name):
   p,_,f,_=evaluated_mesh_records(obj);sha.update(obj.name.encode());sha.update(p.tobytes());sha.update(f.tobytes())
  result[str(frame)]=sha.hexdigest()
 return result
geometry=geometry_snapshot();out.mkdir();cache_dir=out/'cache';cache_dir.mkdir();textures=out/'textures';textures.mkdir();copied={}
bpy.ops.file.make_paths_absolute()
def cache_copy(path):
 path=Path(path).resolve();sha=digest(path);name=sha+path.suffix.lower();target=cache_dir/name
 if not target.exists():shutil.copyfile(path,target)
 copied[name]={'sha256':sha,'bytes':target.stat().st_size,'source':str(path)};return str(target)
for obj in scene.objects:
 for modifier in obj.modifiers:
  if modifier.type=='MESH_CACHE':modifier.filepath=cache_copy(bpy.path.abspath(modifier.filepath))
for cache in bpy.data.cache_files:cache.filepath=cache_copy(bpy.path.abspath(cache.filepath))
if bpy.data.libraries:raise ValueError('Linked libraries need explicit dependency packaging; this version refuses to claim portability')
image_assets={}
for image in bpy.data.images:
 if image.packed_file:content=bytes(image.packed_file.data)
 elif image.filepath and Path(bpy.path.abspath(image.filepath)).is_file():content=Path(bpy.path.abspath(image.filepath)).read_bytes()
 else:continue
 sha=hashlib.sha256(content).hexdigest();extension=Path(image.filepath).suffix.lower()
 if extension not in ('.bmp','.png','.jpg','.jpeg','.tga','.tif','.tiff','.exr','.hdr'):extension='.bin'
 target=textures/(sha+extension)
 if not target.exists():target.write_bytes(content)
 image_assets[image.name]={'file':target.relative_to(out).as_posix(),'sha256':sha,'original_filename':Path(image.filepath).name}
materials=[]
for material in bpy.data.materials:
 material_record={'name':material.name,'diffuse_color':list(material.diffuse_color),'roughness':material.roughness,'metallic':material.metallic,'image_nodes':[]}
 if material.use_nodes:
  for node in material.node_tree.nodes:
   if node.type=='TEX_IMAGE' and node.image:material_record['image_nodes'].append({'node':node.name,'image':node.image.name,'asset':image_assets.get(node.image.name)})
 materials.append(material_record)
(out/'materials.json').write_text(json.dumps({'scope':'asset mapping for manual relinking; not a C4D shader conversion','materials':materials,'images':image_assets},ensure_ascii=False,indent=2),encoding='utf-8')
scene.frame_set(start);scene['MB5_export_quality_note']='Snapshot only; export fidelity does not approve collision quality.'
bpy.ops.file.pack_all();bpy.context.preferences.filepaths.save_version=0
bpy.ops.wm.save_as_mainfile(filepath=str(out/'scene.blend'),relative_remap=False)
for obj in scene.objects:
 for modifier in obj.modifiers:
  if modifier.type=='MESH_CACHE':modifier.filepath='//'+Path(modifier.filepath).relative_to(out).as_posix()
for cache in bpy.data.cache_files:cache.filepath='//'+Path(cache.filepath).relative_to(out).as_posix()
bpy.ops.wm.save_as_mainfile(filepath=str(out/'scene.blend'),relative_remap=False)
bpy.ops.wm.open_mainfile(filepath=str(out/'scene.blend'),load_ui=False,use_scripts=False);scene=bpy.context.scene;rig=bpy.data.objects[plan['bindings']['armature']];mesh=bpy.data.objects[plan['bindings']['mesh']]
after=protected_matrices(rig,names,frames);assert np.array_equal(before,after),'Package changed protected matrices'
scene.frame_set(start);assert digest_actions(set(actions))==actions and mesh_digest(mesh)==skin,'Package changed original curves/skin/morphs'
assert geometry_snapshot()==geometry,'Relocated package geometry changed'
for name,info in copied.items():assert digest(cache_dir/name)==info['sha256']
assert digest(source)==source_sha
report={'passed':True,'source':str(source),'source_sha256':source_sha,'source_file_unchanged':True,'protected_bones':len(names),'matrix_samples':len(frames),
 'protected_matrix_max_error':0,'protected_matrix_sha256':hashlib.sha256(before.tobytes()).hexdigest(),'original_actions_unchanged':len(actions),'original_skin_and_shape_keys_unchanged':True,
 'shape_keys':shape_count,'geometry_sample_frames':samples,'geometry_hashes':geometry,'external_caches':copied,'packed_images':len(image_assets),'linked_libraries':0,
 'self_contained_for_tested_character':True,'physics_quality_approval':False,'C4D_runtime_verified':False,'seconds':time.perf_counter()-started}
(out/'blend-package-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('BLEND_PACKAGE_PASSED',len(frames),len(copied),len(image_assets),flush=True)
