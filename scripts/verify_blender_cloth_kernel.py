"""Real Blender cloth/floor/triangle and MDD timing controls, isolated scene."""
from pathlib import Path
import sys,json,numpy as np
import bpy
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT))
from motionbridge.secondary_surface import MDDWriter
OUT=ROOT/'validation/secondary-motion-dev4/native-kernel-v2';OUT.mkdir(parents=True,exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.preferences.filepaths.save_version=0
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'kernel.blend'))
mesh=bpy.data.meshes.new('cache-test');mesh.from_pydata([(0,0,0),(1,0,0),(0,1,0)],[],[(0,1,2)])
obj=bpy.data.objects.new('cache-test',mesh);bpy.context.scene.collection.objects.link(obj)
path=OUT/'timing.mdd'
if path.exists():raise ValueError('Use a fresh native-kernel directory')
writer=MDDWriter(path,3,3,30)
for z in (0,1,2):writer.append(np.array([[0,0,z],[1,0,z],[0,1,z]],float))
writer.close();cache=obj.modifiers.new('MDD','MESH_CACHE');cache.filepath=str(path);cache.frame_start=1
timing=[]
for frame in (1,2,3):
 bpy.context.scene.frame_set(frame);ev=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());m=ev.to_mesh()
 timing.append(m.vertices[0].co.z);ev.to_mesh_clear()
assert np.allclose(timing,[0,1,2]),timing
bpy.data.objects.remove(obj,do_unlink=True)
def collider(name,vertices,faces):
 mesh=bpy.data.meshes.new(name);mesh.from_pydata(vertices,[],faces);mesh.update()
 obj=bpy.data.objects.new(name,mesh);bpy.context.scene.collection.objects.link(obj)
 obj.modifiers.new('Actual triangle collision','COLLISION');obj.collision.thickness_outer=.003;obj.collision.thickness_inner=.003
 return obj
floor=collider('floor',[(-5,-5,0),(5,-5,0),(5,5,0),(-5,5,0)],[(0,1,2),(0,2,3)])
cube=collider('body-triangles',[(-.25,-.25,0),(.25,-.25,0),(.25,.25,0),(-.25,.25,0),
    (-.25,-.25,.5),(.25,-.25,.5),(.25,.25,.5),(-.25,.25,.5)],
    [(4,5,6),(4,6,7),(0,2,1),(0,3,2),(0,1,5),(0,5,4),(3,7,6),(3,6,2),
     (0,4,7),(0,7,3),(1,2,6),(1,6,5)])
n=14;vertices=[((x/(n-1)-.5)*1.4,(y/(n-1)-.5)*1.4,1.0) for y in range(n) for x in range(n)]
faces=[]
for y in range(n-1):
 for x in range(n-1):
  a=y*n+x;faces.extend([(a,a+1,a+n+1),(a,a+n+1,a+n)])
mesh=bpy.data.meshes.new('cloth');mesh.from_pydata(vertices,[],faces);mesh.update()
obj=bpy.data.objects.new('cloth',mesh);bpy.context.scene.collection.objects.link(obj)
cloth=obj.modifiers.new('Force-driven actual cloth triangles','CLOTH');cloth.settings.quality=8
cloth.settings.mass=.05;cloth.settings.tension_stiffness=40;cloth.settings.compression_stiffness=40
cloth.settings.shear_stiffness=25;cloth.settings.bending_stiffness=.25;cloth.settings.air_damping=2
cloth.collision_settings.use_collision=True;cloth.collision_settings.distance_min=.003;cloth.collision_settings.collision_quality=6
cloth.collision_settings.use_self_collision=True;cloth.collision_settings.self_distance_min=.002
cloth.point_cache.frame_start=1;cloth.point_cache.frame_end=90
bpy.context.scene.render.fps=30
track=[]
for frame in range(1,91):
 bpy.context.scene.frame_set(frame);ev=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());m=ev.to_mesh()
 points=np.array([v.co[:] for v in m.vertices]);ev.to_mesh_clear();track.append(points)
 if frame%30==0:print('KERNEL_FRAME',frame,float(points[:,2].min()),float(points[:,2].max()),flush=True)
assert points[:,2].min()>-.02 and points[:,2].max()>.49
assert points[:,2].min()<.05,'No gravity/floor contact occurred'
over_cube=(np.abs(points[:,0])<.22)&(np.abs(points[:,1])<.22)
assert over_cube.any() and points[over_cube,2].min()>.495,'Cloth penetrated actual cube volume'
assert np.any((points[over_cube,2]>.5)&(points[over_cube,2]<.6)),'No actual triangle support occurred'
report={'passed':True,'blender':bpy.app.version_string,'mdd_frame_start_1_values':timing,
 'cloth_triangles':len(faces),'body_triangles':12,'floor_triangles':2,
 'gravity_displacement_verified':True,'cloth_min_z_m':float(points[:,2].min()),'cloth_max_z_m':float(points[:,2].max()),
 'body_collision_method':'native Blender Cloth against actual cube triangles, no proxies',
 'continuous_collision_certification':False}
(OUT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('NATIVE_CLOTH_KERNEL_PASSED',json.dumps(report),flush=True)
