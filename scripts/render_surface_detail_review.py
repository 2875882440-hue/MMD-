"""Read-only old/new detail views with identical old-pose-derived framing."""
from pathlib import Path
import argparse,hashlib,json,sys
import bpy,numpy as np
from mathutils import Matrix,Vector
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motionbridge.surface_directed import connected_surfaces
from motionbridge.blender_triangle_contacts import evaluated_points
parser=argparse.ArgumentParser()
parser.add_argument('old',type=Path);parser.add_argument('new',type=Path);parser.add_argument('output',type=Path)
parser.add_argument('--component',type=int,default=2)
parser.add_argument('--frames',default='215,216,217,240,241,242,243,244,245')
parser.add_argument('--scale',type=float,default=.25)
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
args.output=args.output.resolve()
args.output.mkdir(parents=True,exist_ok=True);frames=list(map(int,args.frames.split(',')))
paths={'v068':args.old.resolve(),'v069':args.new.resolve()}
hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
reference=ROOT/'research/global-motion-regression/nahida-world/Nahida_MotionBridge_Dance.blend'
framing={};records=[]
for label,path in paths.items():
    bpy.ops.wm.open_mainfile(filepath=str(path),load_ui=False)
    scene=bpy.context.scene;obj=bpy.data.objects['Nahida flexible garments - clearance bake']
    rest=np.array([v.co[:] for v in obj.data.vertices]);edges=np.array([e.vertices[:] for e in obj.data.edges])
    mask=connected_surfaces(rest,edges)==args.component
    if not mask.any():raise ValueError('Missing requested detail component')
    camera_name=scene.camera.name
    with bpy.data.libraries.load(str(reference),link=False) as (src,dst):dst.objects=[camera_name]
    scene.collection.objects.link(dst.objects[0]);scene.camera=dst.objects[0]
    scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=8
    scene.render.threads_mode='FIXED';scene.render.threads=2
    scene.render.resolution_x=540;scene.render.resolution_y=540;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.use_compositing=False;scene.render.use_sequencer=False
    for f in frames:
        if not scene.frame_start<=f<=scene.frame_end:raise ValueError('Detail frame outside animation')
        scene.frame_set(f)
        if label=='v068':
            points=evaluated_points(obj)[mask];center=(points.min(0)+points.max(0))/2
            rotation=scene.camera.matrix_world.to_quaternion().to_matrix().to_4x4()
            direction=rotation.to_quaternion()@Vector((0,0,-1))
            rotation.translation=Vector(center)-direction*4
            framing[f]=[list(row) for row in rotation]
        scene.camera.matrix_world=Matrix(framing[f]);scene.camera.data.type='ORTHO';scene.camera.data.ortho_scale=args.scale
        scene.render.filepath=str(args.output/f'{label}-{f:04d}.png')
        bpy.ops.render.render(write_still=True)
        records.append({'version':label,'frame':f,'file':scene.render.filepath})
assert all(hashlib.sha256(path.read_bytes()).hexdigest()==hashes[label] for label,path in paths.items())
report={'scene_saved':False,'old_pose_framing_for_both':True,'component':args.component,'scale_m':args.scale,
        'camera_by_frame':framing,'frames':records,'source_hashes_unchanged':hashes,
        'note':'Diagnostic close-up only, same old-pose camera framing; no geometry or source project changed.'}
(args.output/'detail-render-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('SURFACE_DETAIL_REVIEW_COMPLETE',flush=True)
