"""Independently verify comparison scenes after their camera was changed."""
from pathlib import Path
import argparse,hashlib,json,sys
import bpy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from solve_secondary_hair_blender import protected_matrices,digest_actions,mesh_digest
parser=argparse.ArgumentParser();parser.add_argument('--input-dir',type=Path,required=True);parser.add_argument('--pair-dir',type=Path,required=True)
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);root=args.input_dir.resolve();pair=args.pair_dir.resolve()
if not pair.is_relative_to(ROOT/'output/secondary_motion') or (pair/'protection-verification.json').exists():raise ValueError('新验证报告必需')
meta=json.loads((root/'input.json').read_text(encoding='utf-8'));plan=json.loads((root/'model-plan.json').read_text(encoding='utf-8'))
results={}
for variant in ('before','after'):
 bpy.ops.wm.open_mainfile(filepath=str(pair/f'{variant}.blend'),load_ui=False,use_scripts=False)
 scene=bpy.context.scene;rig=bpy.data.objects[plan['bindings']['armature']];mesh=bpy.data.objects[plan['bindings']['mesh']]
 frames=np.arange(scene.frame_start,scene.frame_end+.01,.5).tolist()
 matrices=protected_matrices(rig,meta['protected_names'],frames);baseline=np.load(root/'protected-before.npy',mmap_mode='r')
 assert np.array_equal(matrices,baseline),'改变相机后主骨骼改变'
 scene.frame_set(scene.frame_start);assert digest_actions(set(meta['original_actions']))==meta['original_actions']
 assert mesh_digest(mesh)==meta['skin_morph_sha256']
 results[variant]={'samples':len(frames),'bones':len(meta['protected_names']),'matrix_max_error':0,'matrix_sha256':hashlib.sha256(matrices.tobytes()).hexdigest(),'source_actions_unchanged':len(meta['original_actions']),'skin_morph_sha256':mesh_digest(mesh),'shape_keys':len(mesh.data.shape_keys.key_blocks)}
 print('COMPARISON_PROTECTION',variant,flush=True)
assert results['before']==results['after']
(pair/'protection-verification.json').write_text(json.dumps({'passed':True,'variants':results},indent=2),encoding='utf-8')
print('COMPARISON_PROTECTION_PASSED',flush=True)
