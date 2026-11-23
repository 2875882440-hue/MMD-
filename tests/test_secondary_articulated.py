import copy,json
from pathlib import Path
import pytest
from motionbridge.secondary_plan import compile_profile
from motionbridge.secondary_source_map import load_model_document
from motionbridge.secondary_articulated import compile_articulated

@pytest.fixture(scope='module')
def island():
 r=Path(__file__).resolve().parents[1];p=json.loads((r/'profiles/secondary_motion/rin.json').read_text(encoding='utf8'))
 return compile_profile(p,r),load_model_document(r,p['articulated_connections'])

def test_native_closed_island_keeps_hidden_pendant_parent(island):
 p,c=island;out=compile_articulated(p,c)
 assert out['dynamic_rigid_ids']==[135,136,137,138] and out['fixed_rigid_ids']==[2]
 assert [j['id'] for j in out['joints']]==[128,129,130,131,132,133]
 assert not out['simulation_ready'] and not out['individual_render_links_bound']

@pytest.mark.parametrize('case',['hash','duplicate','kinematic','missing_parent','drop_cross_joint','spring','bad_id'])
def test_rejects_incomplete_or_unsafe_island(island,case):
 p,c=copy.deepcopy(island)
 if case=='hash':c['model_sha256']='wrong'
 elif case=='duplicate':c['dynamic_rigid_ids'].append(138)
 elif case=='kinematic':c['dynamic_rigid_ids'].append(2)
 elif case=='missing_parent':c['dynamic_rigid_ids']=[138];c['joint_ids']=[131]
 elif case=='drop_cross_joint':c['joint_ids'].remove(132)
 elif case=='spring':p['joints'][131]['blender']['linear_spring_import_value'][0]=1
 else:c['dynamic_rigid_ids']=[-1]
 with pytest.raises(ValueError):compile_articulated(p,c)
