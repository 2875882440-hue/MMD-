"""Validate the reviewed source adapter on a real animated character, no physics."""
import argparse, hashlib, json, math, sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')];sys.dont_write_bytecode=True
from motionbridge.secondary_plan import compile_profile
from motionbridge.secondary_source_map import load_model_document
from motionbridge.blender_source_surface import BlenderSourceSurfaceAdapter
from bind_secondary_model_blender import mesh_digest,curves_digest
from solve_secondary_hair_blender import set_frame,protected_matrices


def render_review(plan,review,mapping,out):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=8
    scene.render.resolution_x=800;scene.render.resolution_y=1000;scene.render.resolution_percentage=100
    scene.world=bpy.data.worlds.new('ReviewWorld');scene.world.use_nodes=True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value=(.5,.5,.5,1)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=.65
    scene.view_settings.view_transform='Standard'
    colors={'protected_surface':(.34,.36,.4,1),'outer_cloth':(.08,.35,.9,1),'inner_cloth':(.65,.2,.65,1),
            'rigid_ornament':(1,.48,.025,1),'flexible_trim':(.04,.72,.35,1),'hair':(.13,.15,.17,1)}
    objects=[];tri=plan['geometry']['triangles'];pos=plan['geometry']['positions_m'];physical=set(mapping.physical_face_ids.tolist())
    for part in review['parts']:
        fs=[f for f in part['face_ids'] if f in physical]
        if not fs:continue
        vs=sorted({v for f in fs for v in tri[f]});local={v:i for i,v in enumerate(vs)}
        data=bpy.data.meshes.new(part['id']);data.from_pydata([pos[v] for v in vs],[],[[local[v] for v in tri[f]] for f in fs]);data.update()
        obj=bpy.data.objects.new(part['id'],data);scene.collection.objects.link(obj)
        color=colors[part['category']];mat=bpy.data.materials.get(part['category'])
        if mat is None:
            mat=bpy.data.materials.new(part['category']);mat.diffuse_color=color;mat.use_nodes=True
            mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value=color
            mat.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value=.65
        data.materials.append(mat);objects.append((obj,part['category']))
    for loc,power in [((3,-4,5),450),((-3,2,4),300)]:
        data=bpy.data.lights.new('ReviewLight','AREA');data.energy=power;data.size=5
        obj=bpy.data.objects.new('ReviewLight',data);scene.collection.objects.link(obj);obj.location=loc
        obj.rotation_euler=(Vector((0,0,1))-obj.location).to_track_quat('-Z','Y').to_euler()
    data=bpy.data.cameras.new('ReviewCamera');cam=bpy.data.objects.new('ReviewCamera',data);scene.collection.objects.link(cam)
    scene.camera=cam;data.type='ORTHO'
    for name,y,z,scale,hide in [('classification-front',-4,.86,1.9,False),('classification-back',4,.86,1.9,False),
                               ('hardware-front',-4,1.25,.85,True)]:
        data.ortho_scale=scale;cam.location=(0,y,z);cam.rotation_euler=(Vector((0,0,z))-cam.location).to_track_quat('-Z','Y').to_euler()
        for obj,category in objects:obj.hide_render=hide and category!='rigid_ornament'
        scene.render.filepath=str(out/(name+'.png'));bpy.ops.render.render(write_still=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profile',required=True);parser.add_argument('--scene',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--render',action='store_true')
    a=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);out=(ROOT/a.output).resolve();source=(ROOT/a.scene).resolve()
    if not out.is_relative_to(ROOT/'validation') or (out.exists() and (not out.is_dir() or any(out.iterdir()))) or not source.is_relative_to(ROOT):raise ValueError('仅允许新/空验证目录和本开发副本内源场景')
    profile=load_model_document(ROOT,{'path':a.profile,'sha256':hashlib.sha256((ROOT/a.profile).read_bytes()).hexdigest()})
    plan=compile_profile(profile,ROOT);review=plan['surface_segmentation']['review'];manifest=load_model_document(ROOT,profile['source_surface_map'])
    before_sha=hashlib.file_digest(source.open('rb'),'sha256').hexdigest()
    out.mkdir(parents=True,exist_ok=True);bpy.ops.wm.open_mainfile(filepath=str(source),load_ui=False,use_scripts=False)
    scene=bpy.context.scene;mesh=bpy.data.objects[plan['bindings']['mesh']];rig=bpy.data.objects[plan['bindings']['armature']]
    frames=sorted(set([float(x) for x in np.linspace(scene.frame_start,scene.frame_end,17).astype(int)] +
                     [float(x)+.5 for x in np.linspace(scene.frame_start,scene.frame_end-1,16).astype(int)]))
    names=[plan['bones'][i]['blender_name'] for i in plan['protected_bone_ids']]
    set_frame(frames[0]);before={'actions':curves_digest(),'skin_morph':mesh_digest(mesh)};matrices=protected_matrices(rig,names,frames)
    adapter=BlenderSourceSurfaceAdapter(mesh,rig,plan,review,manifest);mapping=adapter.mapping
    rows=[];max_delta_body=0.;mutable={p['id'] for p in review['parts'] if p['category']=='outer_cloth'}
    cloth_vs={v for p in review['parts'] if p['id'] in mutable for f in p['face_ids'] for v in plan['geometry']['triangles'][f]}
    candidates=sorted(set(mapping.canonical_vertex[list(cloth_vs)])-set(mapping.fixed_canonical))
    aliased={w for v,w in mapping.pairs if v!=w};test_vertex=int(next(v for v in candidates if v in aliased))
    for frame in frames:
        set_frame(frame);captured=adapter.capture();x=captured['source_positions'];delta=np.zeros_like(x);delta[test_vertex,0]=.001
        returned=mapping.expand_delta(x,delta)
        body_error=float(np.abs(returned[mapping.fixed_vertices]-x[mapping.fixed_vertices]).max())
        max_delta_body=max(max_delta_body,body_error)
        assert body_error==0 and mapping.check_positions(returned)<2e-6
        rows.append({'frame':frame,'display_error_m':captured['display_alias_error_m'],'delta_body_error_m':body_error})
        print('REAL_SOURCE_ADAPTER_SAMPLE',frame,flush=True)
    after_matrices=protected_matrices(rig,names,frames);set_frame(frames[0])
    after={'actions':curves_digest(),'skin_morph':mesh_digest(mesh)}
    assert before==after and np.array_equal(matrices,after_matrices)
    assert before_sha==hashlib.file_digest(source.open('rb'),'sha256').hexdigest()
    report={'source':str(source),'source_sha256':before_sha,'model_sha256':plan['model_sha256'],
       'review_sha256':plan['surface_segmentation']['review_gate']['review_sha256'],'map':mapping.summary(),
       'native_validation':adapter.validation,'samples':rows,'protected_bones':len(names),
       'protected_matrix_max_error':float(np.abs(after_matrices-matrices).max()),'before':before,'after':after,
       'source_file_unchanged':True,'delta_roundtrip':{'source_vertex':test_vertex,'applied_m':.001,'body_error_m':max_delta_body,
         'scope':'mapping perturbation only; no force, simulation or physical quality claim'},
       'simulation_performed':False,'candidate_generated':False,'native_scene_saved':False}
    (out/'adapter-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    if a.render:render_review(plan,review,mapping,out)
    print('REAL_SOURCE_ADAPTER_VALIDATED',len(rows),flush=True)


if __name__=='__main__':main()
