"""Bake actual visible geometry to Alembic and validate every half frame."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import bpy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from solve_secondary_hair_blender import digest_actions,mesh_digest
from motionbridge.blender_secondary_rigid import evaluated_mesh_records
parser=argparse.ArgumentParser();parser.add_argument('--scene',type=Path,required=True);parser.add_argument('--input-dir',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--take',type=int);parser.add_argument('--render-check',action='store_true')
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);source=args.scene.resolve();out=args.output_dir.resolve();case=args.input_dir.resolve()
if not source.is_relative_to(ROOT) or not out.is_relative_to(ROOT/'output/secondary_motion') or out.exists():raise ValueError('New isolated export directory required')
plan=json.loads((case/'model-plan.json').read_text(encoding='utf-8'));source_sha=hashlib.file_digest(source.open('rb'),'sha256').hexdigest();started=time.perf_counter()
bpy.ops.wm.open_mainfile(filepath=str(source),load_ui=False,use_scripts=False);scene=bpy.context.scene;mesh=bpy.data.objects[plan['bindings']['mesh']]
start=scene.frame_start;end=min(scene.frame_end,start+args.take-1) if args.take else scene.frame_end;fps=scene.render.fps/scene.render.fps_base
frames=np.arange(start,end+.01,.5).tolist();scene.frame_set(start);actions=digest_actions();skin=mesh_digest(mesh);shape_count=len(mesh.data.shape_keys.key_blocks)
actors=sorted([o for o in scene.objects if o.type=='MESH' and not o.hide_render and (o.name==mesh.name or o.name.startswith('MB4_RigidAttachment_'))],key=lambda o:o.name)
camera_name=scene.camera.name if scene.camera else None;world_name=scene.world.name if scene.world else None
out.mkdir();refdir=out/'reference';refdir.mkdir();refs={};metadata={}
def uv_records(obj):
 ev=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());data=ev.to_mesh()
 try:
  if not data.uv_layers:return None
  uv=np.empty(len(data.loops)*2,dtype=np.float32);data.uv_layers[0].data.foreach_get('uv',uv)
  vertices=np.empty(len(data.loops),dtype=np.int32);data.loops.foreach_get('vertex_index',vertices)
  rows=np.column_stack((vertices,uv.reshape(-1,2)));return rows[np.lexsort((rows[:,2],rows[:,1],rows[:,0]))]
 finally:ev.to_mesh_clear()
for i,obj in enumerate(actors):
 points,_,faces,_=evaluated_mesh_records(obj);refs[obj.name]=np.lib.format.open_memmap(refdir/f'{i:03d}.npy',mode='w+',dtype=np.float32,shape=(len(frames),len(points),3))
 uv=uv_records(obj)
 if uv is not None:np.save(refdir/f'{i:03d}-uv.npy',uv)
 metadata[obj.name]={'reference':f'{i:03d}.npy','vertices':len(points),'triangles':len(faces),'uv':f'{i:03d}-uv.npy' if uv is not None else None}
for f,frame in enumerate(frames):
 scene.frame_set(int(frame),subframe=frame-int(frame))
 for obj in actors:
  points,_,_,_=evaluated_mesh_records(obj)
  if len(points)!=metadata[obj.name]['vertices']:raise ValueError('Time-varying topology is not supported by this validator')
  refs[obj.name][f]=points
 if f%150==0:print('ALEMBIC_REFERENCE',f+1,len(frames),flush=True)
for a in refs.values():a.flush()
bpy.ops.object.select_all(action='DESELECT')
for obj in actors:obj.select_set(True)
scene.frame_set(start)
bpy.ops.wm.alembic_export(filepath=str(out/'geometry.abc'),start=start,end=end,xsamples=2,gsamples=2,sh_open=0,sh_close=1,
 selected=True,flatten=True,uvs=True,normals=True,face_sets=True,triangulate=True,use_instancing=False,export_hair=False,export_particles=False,as_background_job=False)
scene.frame_set(start);assert digest_actions(set(actions))==actions and mesh_digest(mesh)==skin
material_names=[m.name for m in bpy.data.materials]
(refdir/'metadata.json').write_text(json.dumps({'frames':frames,'objects':metadata},ensure_ascii=False,indent=2),encoding='utf-8')
del actors;refs={n:np.load(refdir/m['reference'],mmap_mode='r') for n,m in metadata.items()}
bpy.ops.wm.read_factory_settings(use_empty=True);scene=bpy.context.scene;scene.render.fps=round(fps);scene.render.fps_base=round(fps)/fps
bpy.ops.wm.alembic_import(filepath=str(out/'geometry.abc'),set_frame_range=False,always_add_cache_reader=True,as_background_job=False)
imported={o.name:o for o in scene.objects if o.type=='MESH'}
assert set(metadata)==set(imported),f'Alembic object identity mismatch: {set(metadata)^set(imported)}'
uv_error=0.;geometry_error=0.;per_object={n:0. for n in metadata}
scene.frame_set(start)
for name,obj in imported.items():
 uv=uv_records(obj);path=metadata[name]['uv']
 if path:
  expected=np.load(refdir/path);assert uv is not None and uv.shape==expected.shape,'UV topology changed'
  uv_error=max(uv_error,float(np.max(np.abs(uv-expected))))
for f,frame in enumerate(frames):
 scene.frame_set(int(frame),subframe=frame-int(frame))
 for name,obj in imported.items():
  points,_,faces,_=evaluated_mesh_records(obj);expected=refs[name][f]
  assert points.shape==expected.shape and len(faces)==metadata[name]['triangles'],'Alembic topology changed'
  error=float(np.linalg.norm(points-expected,axis=1).max(initial=0));per_object[name]=max(per_object[name],error);geometry_error=max(geometry_error,error)
 if f%150==0:print('ALEMBIC_READBACK',f+1,len(frames),geometry_error,flush=True)
passed=geometry_error<1e-5 and uv_error<1e-6
# A useful Blender readback package, restoring original packed materials/camera.
dummy_slots={n:[m.name if m else '' for m in o.data.materials] for n,o in imported.items()}
with bpy.data.libraries.load(str(source),link=False) as (available,loaded):
 loaded.materials=[n for n in material_names if n in available.materials]
 loaded.objects=[camera_name] if camera_name and camera_name in available.objects else []
 loaded.worlds=[world_name] if world_name and world_name in available.worlds else []
materials=dict(zip([n for n in material_names],loaded.materials))
for name,obj in imported.items():
 for i,material in enumerate(dummy_slots[name]):
  if material in materials and materials[material]:obj.data.materials[i]=materials[material]
if loaded.objects:
 camera=loaded.objects[0];scene.collection.objects.link(camera);scene.camera=camera
if loaded.worlds:scene.world=loaded.worlds[0]
scene.render.engine='BLENDER_WORKBENCH';scene.render.resolution_x=640;scene.render.resolution_y=800;scene.render.resolution_percentage=100
scene.display.shading.light='STUDIO';scene.display.shading.color_type='TEXTURE';scene.display.shading.show_shadows=True;scene.display.shading.show_cavity=True;scene.display.shading.background_type='WORLD'
scene.frame_start=start;scene.frame_end=end;scene.frame_set(start);bpy.context.preferences.filepaths.save_version=0
bpy.ops.file.pack_all();bpy.ops.wm.save_as_mainfile(filepath=str(out/'readback.blend'),relative_remap=False)
for cache in bpy.data.cache_files:cache.filepath='//geometry.abc'
bpy.ops.wm.save_as_mainfile(filepath=str(out/'readback.blend'),relative_remap=False)
bpy.ops.wm.open_mainfile(filepath=str(out/'readback.blend'),load_ui=False,use_scripts=False)
scene=bpy.context.scene;imported={o.name:o for o in scene.objects if o.type=='MESH' and o.name in metadata}
saved_error=0.
for frame in sorted(set([float(start),(start+end)//2+.5 if start<end else float(start),float(end)])):
 scene.frame_set(int(frame),subframe=frame-int(frame));f=round((frame-start)*2)
 for name,obj in imported.items():
  points,_,_,_=evaluated_mesh_records(obj);saved_error=max(saved_error,float(np.linalg.norm(points-refs[name][f],axis=1).max(initial=0)))
assert saved_error<1e-5,'Saved Alembic readback scene did not resolve its relative cache'

if args.render_check:
 shots=out/'readback-shots';shots.mkdir()
 for frame in sorted(set([start,(start+end)//2,min(end,849),end])):
  scene.frame_set(frame);scene.render.filepath=str(shots/f'{frame:04d}.png');bpy.ops.render.render(write_still=True)
report={'passed':passed,'format':'Alembic actual evaluated geometry','source':str(source),'source_sha256':source_sha,
 'source_file_unchanged':hashlib.file_digest(source.open('rb'),'sha256').hexdigest()==source_sha,'source_actions_skin_and_morphs_unchanged':True,'source_shape_keys':shape_count,
 'objects':len(imported),'integer_frames':end-start+1,'sample_frames':len(frames),'samples_per_frame':2,'shutter':[0,1],'fps':fps,
 'saved_readback_geometry_max_error_m':saved_error,'maximum_world_vertex_error_m':geometry_error,'maximum_uv_error':uv_error,'per_object_maximum_world_error_m':per_object,
 'original_bones_or_shape_keys_in_alembic':False,'materials':'face sets + packed original materials restored in readback.blend; C4D shaders not verified',
 'physics_quality_approval':False,'C4D_runtime_verified':False,'seconds':time.perf_counter()-started}
(out/'alembic-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');assert passed,report
print('ALEMBIC_GEOMETRY_PASSED',geometry_error,uv_error,len(imported),flush=True)
