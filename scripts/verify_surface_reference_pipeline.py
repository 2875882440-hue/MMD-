"""Real Blender producer/save/reopen/reference/tamper regression on tiny meshes."""
from pathlib import Path
import copy,hashlib,json,sys
import bpy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motionbridge.blender_surfaces import bake_surface_clearance,audit_surface_clearance,body_capsules
DEST=ROOT/'research/v069/reference-pipeline';DEST.mkdir(exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
scene=bpy.context.scene;scene.frame_start=1;scene.frame_end=9;scene.render.fps=30
arm=bpy.data.armatures.new('ReferenceRig');rig=bpy.data.objects.new('ReferenceRig',arm);scene.collection.objects.link(rig)
bpy.context.view_layer.objects.active=rig;rig.select_set(True);bpy.ops.object.mode_set(mode='EDIT')
names=set(n for a,b,_ in body_capsules(rig) for n in (a,b))|{'BodyStatic','ClothControl'}
for i,name in enumerate(sorted(names)):
    bone=arm.edit_bones.new(name);bone.head=(2,0,i*.1) if name not in {'BodyStatic','ClothControl'} else (0,0,0)
    bone.tail=bone.head+__import__('mathutils').Vector((0,0,.1))
bpy.ops.object.mode_set(mode='OBJECT')
cube=[[-.05,-.05,-.02],[.05,-.05,-.02],[.05,.05,-.02],[-.05,.05,-.02],[-.05,-.05,.02],[.05,-.05,.02],[.05,.05,.02],[-.05,.05,.02]]
faces=[[4,5,6],[4,6,7],[0,2,1],[0,3,2],[0,1,5],[0,5,4],[3,7,6],[3,6,2],[0,4,7],[0,7,3],[1,2,6],[1,6,5]]
def mesh(name,vertices,triangles,bone):
    data=bpy.data.meshes.new(name);data.from_pydata(vertices,[],triangles);data.update()
    obj=bpy.data.objects.new(name,data);scene.collection.objects.link(obj)
    group=obj.vertex_groups.new(name=bone);group.add(list(range(len(vertices))),1.,'REPLACE')
    modifier=obj.modifiers.new('Skin','ARMATURE');modifier.object=rig
    return obj
body=mesh('Nahida_mesh',cube,faces,'BodyStatic');body.data.materials.append(bpy.data.materials.new('体'))
cloth=mesh('ReferenceCloth',[[-.01,-.01,.03],[.01,-.01,.03],[.01,.01,.03],[-.01,.01,.03]],[[0,1,2],[0,2,3]],'ClothControl')
for f in (1,9):
    rig.pose.bones['ClothControl'].location=(0,-.015,0)
    rig.pose.bones['ClothControl'].keyframe_insert('location',frame=f)
reference=DEST/'surface_solver_reference.npz'
report=bake_surface_clearance(scene,rig,[cloth],reference_path=reference)
assert report['objects'][0]['triangle_contact']['after']['body_samples_over_1mm']<report['objects'][0]['triangle_contact']['before']['body_samples_over_1mm']
target=DEST/'reference.blend';bpy.ops.wm.save_as_mainfile(filepath=str(target));digest=hashlib.sha256(target.read_bytes()).hexdigest()
bpy.ops.wm.open_mainfile(filepath=str(target),load_ui=False);scene=bpy.context.scene;rig=bpy.data.objects['ReferenceRig']
audit=audit_surface_clearance(scene,rig,report,reference_path=reference);assert audit['passed'],audit
checks={'new_pre_embedding_reference':audit}
try:audit_surface_clearance(scene,rig,report,reference_path=DEST/'missing.npz')
except ValueError:checks['missing_reference_rejected']=True
else:raise AssertionError('Missing reference accepted')
wrong=copy.deepcopy(report);wrong['solver_reference']['sha256']='0'*64
try:audit_surface_clearance(scene,rig,wrong,reference_path=reference)
except ValueError:checks['wrong_hash_rejected']=True
else:raise AssertionError('Wrong hash accepted')
with np.load(reference,allow_pickle=False) as archive:points={name:archive[name].copy() for name in archive.files}
points['ReferenceCloth'][4,0,0]+=.00002;tampered=DEST/'tampered.npz';np.savez_compressed(tampered,**points)
wrong=copy.deepcopy(report);wrong['solver_reference']['sha256']=hashlib.sha256(tampered.read_bytes()).hexdigest()
audit=audit_surface_clearance(scene,rig,wrong,reference_path=tampered);assert not audit['passed']
checks['wrong_vertices_rejected_even_with_matching_hash']=True
legacy=copy.deepcopy(report);legacy.pop('solver_reference')
audit=audit_surface_clearance(scene,rig,legacy);assert audit['passed']
ref=audit['objects'][0]['triangle_contacts']['geometry_reference']
assert ref['mode']=='legacy_saved_absolute_key_lbs' and not ref['original_solver_track_available']
checks['legacy_reference']=audit
assert hashlib.sha256(target.read_bytes()).hexdigest()==digest
(DEST/'verification.json').write_text(json.dumps({'passed':True,'checks':checks,'saved_scene_unchanged':True},indent=2),encoding='utf-8')
print('SAVED_SURFACE_REFERENCE_PIPELINE_PASSED',flush=True)
