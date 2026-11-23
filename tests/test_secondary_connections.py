import copy,json
from pathlib import Path
import numpy as np
import pytest
from motionbridge.secondary_plan import compile_profile
from motionbridge.secondary_source_map import load_model_document,SourceSurfaceMap
from motionbridge.secondary_connections import compile_connections,project_source_stitches

@pytest.fixture(scope='module')
def real():
 r=Path(__file__).resolve().parents[1];profile=json.loads((r/'profiles/secondary_motion/rin.json').read_text(encoding='utf8'));p=compile_profile(profile,r)
 return p,p['surface_segmentation']['review'],load_model_document(r,profile['source_surface_map']),load_model_document(r,profile['source_connections'])

def test_actual_roots_joints_and_seams(real):
 p,r,m,c=real;compiled=compile_connections(p,r,m,c)
 assert 0<compiled['cloth_root_vertex_count']<2593 and len(compiled['explicit_root_parts'])==7 and compiled['stitch_group_count']==6
 assert len(compiled['rigid_joints'])==2 and not compiled['simulation_ready']
 assert all(j['parent_bone_id']==16 for j in compiled['rigid_joints'])
 assert not compiled['contact_exclusions_added']

@pytest.mark.parametrize('bad',['hash','duplicate','dynamic_parent','wrong_part','wrong_stitch','duplicate_stitch'])
def test_rejects_wrong_actual_connections(real,bad):
 p,r,m,c=copy.deepcopy(real)
 if bad=='hash':c['review_sha256']='bad'
 elif bad=='duplicate':c['rigid_joints'].append(c['rigid_joints'][0])
 elif bad=='dynamic_parent':c['rigid_joints']=[{'part_id':'neck_pendant','joint_id':131}]
 elif bad=='wrong_part':c['rigid_joints'][0]['part_id']='m11-f7924'
 elif bad=='wrong_stitch':c['stitches'][0]['vertex_pairs'][0][0]=0
 else:c['stitches'].append(c['stitches'][0])
 with pytest.raises(ValueError):compile_connections(p,r,m,c)

def test_bidirectional_stitch_and_source_root_atomicity(real):
 p,r,m,c=real;mapping=SourceSurfaceMap(p,r,m);compiled=compile_connections(p,r,m,c);x=mapping.rest.copy()
 a,b=compiled['stitch_vertex_pairs'][0];pred=x.copy();pred[a,2]+=.001
 original=pred.copy();out=project_source_stitches(x,pred,np.ones(len(x)),compiled,mapping)
 assert np.array_equal(pred,original) and np.array_equal(out[mapping.fixed_vertices],x[mapping.fixed_vertices])
 assert max(np.linalg.norm(out[u]-out[v]) for u,v in compiled['stitch_vertex_pairs'])<=2e-6
 conflict=x.copy();conflict[mapping.canonical_vertex==a,2]+=.001
 with pytest.raises(ValueError):project_source_stitches(conflict,conflict,np.zeros(len(x)),compiled,mapping)
 assert np.array_equal(pred,original)

@pytest.mark.parametrize('bad',['nan','negative_mass','bad_shape','budget'])
def test_rejects_invalid_projection(real,bad):
 p,r,m,c=real;mapping=SourceSurfaceMap(p,r,m);x=mapping.rest.copy();pred=x.copy();w=np.ones(len(x));n=32
 if bad=='nan':pred[0,0]=np.nan
 elif bad=='negative_mass':w[0]=-1
 elif bad=='bad_shape':pred=pred[:-1]
 else:n=0
 with pytest.raises(ValueError):project_source_stitches(x,pred,w,p['source_connections'],mapping,n)
