"""Small real-Blender geometry acceptance cases, no user scene touched."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motionbridge.blender_triangle_contacts import TriangleSurface,mesh_contact_tracks

cube=np.array([[-.05,-.05,-.02],[.05,-.05,-.02],[.05,.05,-.02],[-.05,.05,-.02],
               [-.05,-.05,.02],[.05,-.05,.02],[.05,.05,.02],[-.05,.05,.02]])
cube_faces=np.array([[4,5,6],[4,6,7],[0,2,1],[0,3,2],[0,1,5],[0,5,4],
                     [3,7,6],[3,6,2],[0,4,7],[0,7,3],[1,2,6],[1,6,5]])
sheet=np.array([[-.01,-.01,.03],[.01,-.01,.03],[.01,.01,.03],[-.01,.01,.03]])
faces=np.array([[0,1,2],[0,2,3]]);edges=np.array([[0,1],[1,2],[2,3],[3,0],[0,2]])
caps=np.tile(np.array([[[2.,0,-1],[2,0,1]]]),(9,1,1,1))
results={}

def solve(rest,raw,es,fs,body_points):
    before=raw.copy();surface=TriangleSurface(body_points,cube_faces)
    final,report=mesh_contact_tracks(raw,raw.copy(),caps,np.full((1,len(rest)),.02),es,rest,fs,
                                    body_points,cube_faces,lambda _:surface,30)
    assert np.array_equal(before,raw)
    assert report['combined_guard']['accepted']
    assert report['maximum_extra_shift_m']<=.03+1e-7
    return final,report

raw=np.tile(sheet,(9,1,1));raw[:,:,2]=.015
final,report=solve(sheet,raw,edges,faces,cube)
assert report['after']['body_samples_over_1mm']<report['before']['body_samples_over_1mm']
assert final[:,:,2].min()>.02
results['body_contact']={'before':report['before'],'after':report['after']}

clear=raw.copy();clear[:,:,2]=.04
final,report=solve(sheet,clear,edges,faces,cube)
assert np.allclose(final,clear,atol=1e-8)
results['clear_control_unchanged']=True

rest=np.concatenate([sheet, sheet+[0,0,-.02]])
raw=np.tile(rest,(9,1,1));raw[:,:4,2]=-.004;raw[:,4:,2]=0
final,report=solve(rest,raw,np.concatenate([edges,edges+4]),np.concatenate([faces,faces+4]),cube+[1,0,0])
assert report['before']['layer_contact_samples_over_1mm']>0
assert report['after']['layer_contact_samples_over_1mm']<report['before']['layer_contact_samples_over_1mm']
assert final[:,:4,2].mean()>final[:,4:,2].mean()
results['reversed_parallel_layers']={'before':report['before'],'after':report['after']}
out=ROOT/'research/v069/solver-blender-tests.json'
out.write_text(json.dumps({'passed':True,'cases':results},indent=2),encoding='utf-8')
print('TRIANGLE_SOLVER_BLENDER_TESTS_PASSED',flush=True)
