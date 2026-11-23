"""Capture immutable posed driver surfaces; source motion/data never written."""
from pathlib import Path
import argparse,hashlib,json,sys
import bpy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from motionbridge.secondary_plan import compile_profile
from motionbridge.secondary_motion import _inside
from motionbridge.secondary_surface import cloth_partition,rigid_components,MDDWriter
from motionbridge.secondary_surface_config import surface_settings
from motionbridge.secondary_contact_constraints import contact_active_faces
from solve_secondary_hair_blender import digest_actions,protected_matrices,mesh_digest,set_frame
from bind_secondary_model_blender import validate_geometry
parser=argparse.ArgumentParser();parser.add_argument('--scene',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--profile',default='rin')
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);out=args.output_dir.resolve()
if not out.is_relative_to(ROOT/'output/secondary_motion') or out.exists():raise ValueError('必须使用本副本内的新独立输出目录')
source=_inside(ROOT,args.scene);source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
profile=json.loads((ROOT/f'profiles/secondary_motion/{args.profile}.json').read_text(encoding='utf-8'))
plan=compile_profile(profile,ROOT);cfg=surface_settings(profile)
from motionbridge.secondary_segmentation import require_review
require_review(plan)
bpy.ops.wm.open_mainfile(filepath=str(source),load_ui=False,use_scripts=False)
scene=bpy.context.scene;rig=bpy.data.objects[plan['bindings']['armature']];mesh=bpy.data.objects[plan['bindings']['mesh']]
validate_geometry(mesh,rig,plan);frames=list(range(scene.frame_start,scene.frame_end+1));fps=scene.render.fps/scene.render.fps_base
assert len(frames)>0 and np.allclose(mesh.matrix_world,np.eye(4))
out.mkdir(parents=True);bpy.context.preferences.filepaths.save_version=0
scene.frame_set(frames[0]);actions=digest_actions();skin=mesh_digest(mesh)
# Move the active save context before ANY native point-cache modifier exists.
bpy.ops.wm.save_as_mainfile(filepath=str(out/'before.blend'))
names=[plan['bones'][i]['blender_name'] for i in plan['protected_bone_ids']]
audit_frames=sorted(set(frames+[f+.5 for f in frames[:-1]]))
protected=protected_matrices(rig,names,audit_frames);np.save(out/'protected-before.npy',protected)
ids,faces,pins,source_faces=cloth_partition(plan)
hard,excluded,hard_faces=rigid_components(plan,cfg['ornament_bone_tokens'],maximum_rigid_extent=cfg['maximum_rigid_extent_m'])
protected_vertices={v for r in plan['regions'] if r['role'] in ('body','boots') for v in r['vertex_ids']}
body_materials=cfg['body_materials']
material_names=plan['geometry']['material_names'];material_ids={material_names.index(n) for n in body_materials}
body_source_faces=np.flatnonzero(np.isin(plan['geometry']['face_material_ids'],list(material_ids)))
body_tri=np.asarray(plan['geometry']['triangles'],int)[body_source_faces];body_ids=np.unique(body_tri)
body_faces=np.searchsorted(body_ids,body_tri)
input_positions=np.lib.format.open_memmap(out/'source-positions.npy',mode='w+',dtype=np.float32,
                                        shape=(len(frames),len(mesh.data.vertices),3))
for index,frame in enumerate(frames):
 scene.frame_set(frame);evaluated=mesh.evaluated_get(bpy.context.evaluated_depsgraph_get());data=evaluated.to_mesh()
 if len(data.vertices)!=len(mesh.data.vertices):raise ValueError('求解驱动拓扑发生变化')
 points=np.empty(len(data.vertices)*3,dtype=np.float32);data.vertices.foreach_get('co',points)
 evaluated.to_mesh_clear();input_positions[index]=points.reshape(-1,3)
 if index%90==0:print('SURFACE_CAPTURE',index+1,len(frames),flush=True)
input_positions.flush()
# Add only pinned triangles that actually conflict with the floor, with a
# topological collar. Unrelated fully pinned torso panels stay kinematic.
unsafe=np.zeros(len(ids),bool)
for index in range(len(frames)):unsafe|=input_positions[index,ids,2]<cfg['floor_z_m']+cfg['body_gap_m']
active_faces=contact_active_faces(pins,faces,unsafe,np.isin(ids,list(protected_vertices)),np.isin(source_faces,hard_faces),cfg['contact_pin_collar'])
selected=np.unique(faces[active_faces]);cloth_ids=ids[selected];cloth_faces=np.searchsorted(selected,faces[active_faces]);cloth_pins=pins[selected]
cloth_source_faces=source_faces[active_faces]
metadata={'source':str(source),'source_sha256':source_sha,'frame_start':frames[0],'frame_end':frames[-1],
 'frames':len(frames),'fps':fps,'original_actions':actions,'skin_morph_sha256':skin,'shape_keys':len(mesh.data.shape_keys.key_blocks),
 'protected_names':names,'protected_matrix_sha256':hashlib.sha256(protected.tobytes()).hexdigest(),
 'body_materials':body_materials,'body_source_triangles':len(body_faces),'cloth_dynamic_triangles':len(cloth_faces),
 'cloth_vertices':len(cloth_ids),'all_cloth_vertices':len(ids),'all_cloth_triangles':len(faces),
 'fully_pinned_cloth_triangles_kept_kinematic':int(np.sum(~active_faces)),
 'rigid_components':hard,'excluded_rigid_components':excluded,'hard_faces':hard_faces,
 'body_source_face_ids':body_source_faces.tolist(),'cloth_source_face_ids':cloth_source_faces.tolist()}
np.savez(out/'partitions.npz',cloth_ids=cloth_ids,cloth_faces=cloth_faces,cloth_pins=cloth_pins,
         cloth_source_faces=cloth_source_faces,body_ids=body_ids,body_faces=body_faces,body_source_faces=body_source_faces)
for name,value in [('input.json',metadata),('model-plan.json',plan),('profile.json',profile)]:
 (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
assert source_sha==hashlib.sha256(source.read_bytes()).hexdigest()
print('SURFACE_CAPTURE_COMPLETE',json.dumps({k:metadata[k] for k in ['frames','body_source_triangles','cloth_dynamic_triangles','cloth_vertices']}),flush=True)
