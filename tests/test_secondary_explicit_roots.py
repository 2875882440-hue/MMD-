import copy
import pytest
from test_secondary_connections import real
from motionbridge.secondary_connections import compile_connections

def test_waist_weight_variant_is_not_automatically_pinned(real):
 p,r,m,c=real;result=compile_connections(p,r,m,c)
 assert result['cloth_root_vertex_count']==109
 assert not {16536,28964}&set(result['source_root_vertex_ids'])
 assert len(result['explicit_root_parts'])==7

@pytest.mark.parametrize('case',['missing','duplicate','foreign_vertex','empty','no_evidence'])
def test_rejects_unreviewed_root_segments(real,case):
 p,r,m,c=copy.deepcopy(real)
 if case=='missing':c['roots']=[]
 elif case=='duplicate':c['roots'].append(c['roots'][0])
 elif case=='foreign_vertex':c['roots'][0]['vertex_ids']=[0,1]
 elif case=='empty':c['roots'][0]['vertex_ids']=[]
 else:c['roots'][0]['evidence']=''
 with pytest.raises(ValueError):compile_connections(p,r,m,c)
