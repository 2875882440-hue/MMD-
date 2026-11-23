"""Verify the independently copied comparison cache really deforms geometry."""
from pathlib import Path
import argparse,hashlib,json,sys
import bpy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT))
from motionbridge.blender_secondary_rigid import evaluated_mesh_records
parser=argparse.ArgumentParser();parser.add_argument('--pair-dir',type=Path,required=True)
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);pair=args.pair_dir.resolve()
if not pair.is_relative_to(ROOT/'output/secondary_motion') or (pair/'cache-verification.json').exists():raise ValueError('新目录验证必需')
bpy.ops.wm.open_mainfile(filepath=str(pair/'after.blend'),load_ui=False,use_scripts=False)
scene=bpy.context.scene;meshes=[o for o in scene.objects if o.type=='MESH' and o.modifiers.get('MB4_DynamicSurfaceCache')]
assert len(meshes)==1;mesh=meshes[0];modifier=mesh.modifiers['MB4_DynamicSurfaceCache']
path=Path(bpy.path.abspath(modifier.filepath)).resolve();assert path==pair/'surface-final.mdd' and path.is_file()
digest=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
assert digest(path)==digest(pair.parent/'surface-final.mdd')
count=scene.frame_end-scene.frame_start+1
reference=np.memmap(path,dtype='>f4',mode='r',offset=8+4*count,shape=(count,len(mesh.data.vertices),3))
sample_frames=sorted(set(np.linspace(scene.frame_start,scene.frame_end,17).round().astype(int).tolist()+[min(scene.frame_end,f) for f in [30,60,571,889]]))
error=0.
for frame in sample_frames:
 scene.frame_set(frame);points,ids,_,_=evaluated_mesh_records(mesh)
 error=max(error,float(np.linalg.norm(points-reference[frame-scene.frame_start,ids],axis=1).max()))
assert error<1e-5,f'派生对照工程实际缓存加载错误: {error}'
report={'passed':True,'cache_file':str(path),'sha256':digest(path),'copied_cache_sha_matches_source':True,
 'actual_evaluated_geometry_max_error_m':error,'sample_frames':sample_frames,'scope':'all original mesh vertices retained in modifier output at listed integer frames; rigid objects verified separately'}
(pair/'cache-verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('COMPARISON_CACHE_PASSED',json.dumps(report),flush=True)
