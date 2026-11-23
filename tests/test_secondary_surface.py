from pathlib import Path
import json,struct
import numpy as np
import pytest
from motionbridge.secondary_surface import cloth_partition,rigid_fit,rigid_components,MDDWriter,closest_segments,floor_depth
from motionbridge.secondary_surface import new_child_output
ROOT=Path(__file__).resolve().parents[1]


def test_new_output_tag_cannot_escape_project_directory(tmp_path):
    for tag in ('../old','..','a/b','a\\b','C:/outside'):
        with pytest.raises(ValueError):new_child_output(tmp_path,tag)
    assert new_child_output(tmp_path,'safe-new')==tmp_path/'safe-new'


def test_new_output_tag_preserves_existing_results(tmp_path):
    (tmp_path/'old').mkdir();(tmp_path/'old/keep.txt').write_text('keep')
    with pytest.raises(ValueError):new_child_output(tmp_path,'old')
    assert (tmp_path/'old/keep.txt').read_text()=='keep'


@pytest.fixture(scope='module')
def plan():return json.loads((ROOT/'output/secondary_motion/dev3/rin-dance-contact/model-plan.json').read_text(encoding='utf-8'))


def test_cloth_partition_uses_actual_faces_and_protects_shared_body(plan):
    ids,faces,pins,source=cloth_partition(plan)
    assert len(ids)==10854 and len(faces)==len(source)>10000
    assert np.array_equal(ids[faces],np.array(plan['geometry']['triangles'])[source])
    body={v for r in plan['regions'] if r['role'] in ('body','boots') for v in r['vertex_ids']}
    assert all(pin==1 for v,pin in zip(ids,pins) if v in body)
    assert (pins==0).any() and (pins==1).any()


def test_rigid_attachment_removes_skin_shear_without_reflecting_shape():
    rest=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,.2]],float)
    deformed=rest@np.array([[1,.4,0],[0,1.2,0],[0,0,1]])+[4,2,1]
    final,rotation,translation=rigid_fit(rest,deformed)
    assert np.linalg.det(rotation)==pytest.approx(1)
    assert final.mean(0)==pytest.approx(deformed.mean(0))
    assert np.linalg.norm(final[:,None]-final[None,:],axis=2)==pytest.approx(np.linalg.norm(rest[:,None]-rest[None,:],axis=2),abs=1e-12)


@pytest.mark.parametrize('bad',[np.zeros((2,3)),np.zeros((4,3)),np.full((3,3),np.nan)])
def test_degenerate_rigid_pieces_are_rejected(bad):
    with pytest.raises(ValueError):rigid_fit(bad,bad)


def test_rigid_partition_excludes_protected_body_and_finds_hair_ornaments(plan):
    parts,excluded,faces=rigid_components(plan,['髮飾','髪飾'])
    assert len(parts)>8 and len(faces)>5000
    protected={v for r in plan['regions'] if r['role'] in ('body','boots') for v in r['vertex_ids']}
    assert all(not set(p['vertex_ids'])&protected for p in parts)
    assert all(p['reason'] in ('shares_protected_body_vertices','degenerate_component','flexible_trim_requires_cloth_follow') for p in excluded)


def test_mdd_is_complete_big_endian_and_does_not_overwrite(tmp_path):
    path=tmp_path/'cache.mdd';writer=MDDWriter(path,2,3,30)
    first=np.arange(9).reshape(3,3);writer.append(first);writer.append(first+1);writer.close()
    data=path.read_bytes();assert struct.unpack('>2i',data[:8])==(2,3)
    assert np.frombuffer(data[8:16],dtype='>f4')==pytest.approx([0,1/30])
    assert np.array_equal(np.frombuffer(data[16:52],dtype='>f4').reshape(3,3),first)
    with pytest.raises(FileExistsError):MDDWriter(path,2,3,30)


def test_partial_and_nonfinite_cache_is_not_success(tmp_path):
    writer=MDDWriter(tmp_path/'partial.mdd',2,3,30)
    with pytest.raises(ValueError):writer.append(np.full((3,3),np.nan))
    with pytest.raises(ValueError):writer.close()


@pytest.mark.parametrize('b0,b1,distance',[([.5,-1,0],[.5,1,0],0),([.5,1,0],[2,1,0],1),([2,0,0],[2,0,0],1)])
def test_hair_segment_contacts_include_crossing_parallel_and_point_cases(b0,b1,distance):
    a,b,s,t=closest_segments([[0,0,0]],[[1,0,0]],[b0],[b1])
    assert np.linalg.norm(a-b)==pytest.approx(distance)
    assert 0<=s[0]<=1 and 0<=t[0]<=1


def test_floor_contact_is_a_solver_depth_not_a_root_translation():
    points=np.array([[0,0,-.1],[1,0,.01],[2,0,1.]])
    original=points.copy();assert floor_depth(points,.02)==pytest.approx([.12,.01,0])
    assert np.array_equal(points,original)
