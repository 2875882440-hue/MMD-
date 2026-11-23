import json,copy
from pathlib import Path
import numpy as np
import pytest
from motionbridge.secondary_plan import compile_profile
from motionbridge.secondary_source_map import load_model_document,SourceSurfaceMap
from motionbridge.secondary_surface_mounts import compile_mounts,project_mounts

@pytest.fixture(scope='module')
def real_mounts():
 r=Path(__file__).resolve().parents[1];profile=json.loads((r/'profiles/secondary_motion/rin.json').read_text(encoding='utf8'));p=compile_profile(profile,r)
 return p,p['surface_segmentation']['review'],load_model_document(r,profile['source_surface_map']),load_model_document(r,profile['surface_mounts'])

def test_real_mounts_keep_both_clothing_layers(real_mounts):
 p,r,m,c=real_mounts;out=compile_mounts(p,r,m,c)
 assert len(out['mounts'])==5 and sum(len(x['anchors']) for x in out['mounts'])==158
 assert all(x['cloth_parts']==['m11-f9610','m18-f37772'] for x in out['mounts'][:4])
 assert not out['simulation_ready']

@pytest.mark.parametrize('case',['fingerprint','duplicate','one_layer','foreign_vertex','no_evidence'])
def test_rejects_invented_or_incomplete_mount(real_mounts,case):
 p,r,m,c=copy.deepcopy(real_mounts)
 if case=='fingerprint':c['review_sha256']='bad'
 elif case=='duplicate':c['mounts'].append(c['mounts'][0])
 elif case=='one_layer':c['mounts'][0]['anchors']=[a for a in c['mounts'][0]['anchors'] if a['cloth_part']=='m18-f37772']
 elif case=='foreign_vertex':c['mounts'][0]['anchors'][0]['cloth_vertex_id']=0
 else:c['mounts'][0]['evidence']=''
 with pytest.raises(ValueError):compile_mounts(p,r,m,c)

def test_two_way_mount_and_unchanged_inputs(real_mounts):
 p,r,m,c=real_mounts;mapping=SourceSurfaceMap(p,r,m);compiled=compile_mounts(p,r,m,c);x=mapping.rest.copy();saved=x.copy()
 metrics={row['part_id']:[1.,1.] for row in compiled['mounts']};offsets={pid:[0,0,.004] for pid in metrics}
 out,proof=project_mounts(x,mapping,compiled,p['source_connections']['source_root_vertex_ids'],np.ones(len(x)),metrics,offsets)
 assert proof['passed'] and proof['cloth_response_m']>0 and proof['rendered_residual_m']<=2e-6
 assert np.array_equal(x,saved) and np.array_equal(out[mapping.fixed_vertices],x[mapping.fixed_vertices])
 assert any(v>0 for v in proof['rigid_response_m'].values())

@pytest.mark.parametrize('case',['negative_metric','nan_mass','bad_budget','fixed_prediction'])
def test_invalid_projection_is_atomic(real_mounts,case):
 p,r,m,c=real_mounts;mapping=SourceSurfaceMap(p,r,m);compiled=p['surface_mounts'];x=mapping.rest.copy();saved=x.copy();weights=np.ones(len(x));metrics={row['part_id']:[1.,1.] for row in compiled['mounts']};offsets={};iterations=128
 pid=next(iter(metrics))
 if case=='negative_metric':metrics[pid]=[-1,1]
 elif case=='nan_mass':weights[0]=np.nan
 elif case=='bad_budget':iterations=0
 else:metrics[pid]=[0,0];offsets[pid]=[0,0,.004]
 with pytest.raises(ValueError):project_mounts(x,mapping,compiled,p['source_connections']['source_root_vertex_ids'],weights,metrics,offsets,iterations)
 assert np.array_equal(x,saved)
