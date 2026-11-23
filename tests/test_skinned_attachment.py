import copy
import numpy as np
import pytest
from test_secondary_coupled_surface import FixtureMap
from test_secondary_surface_mounts import real_mounts
from motionbridge.secondary_mount_reference import mount_mode, mount_reference
from motionbridge.secondary_coupled_surface import MountedClothFrame,solve_coupled_surface
from motionbridge.secondary_coupled_contacts import SourceTriangleContacts
from motionbridge.secondary_surface_mounts import compile_mounts,project_mounts


def build(mode='source_skinned_attachment',scale=1.12,offset=None):
    m=FixtureMap();m.fixed_vertices=m.fixed_canonical
    x=m.rest.copy();x[:7,0]*=scale
    row={'part_id':'tab','canonical_vertex_ids':[4,5,6], 'anchors':[[4,0],[5,1],[6,2]],
         'motion_mode':mode,'motion_mode_evidence':'synthetic matching skin seam'}
    compiled={'mounts':[row]};conn={'source_root_vertex_ids':[7,8,9],'stitch_vertex_pairs':[]}
    f=MountedClothFrame(x,m,compiled,conn,np.ones(10),{'tab':[1.,1.]},offset)
    return f,SourceTriangleContacts(m,compiled,conn,gap=.002,floor_z=-.1),compiled,conn


def test_source_mode_unforced_retains_input_shape():
    f,_,_,_=build();np.testing.assert_allclose(f.positions(),f.source,atol=1e-15)
    assert f.mount_reference_audit['tab']['reference_source_fit_error_m']==0
    assert not f.mount_reference_audit['tab']['constant_static_shape_across_frames']


def test_rest_mode_retains_static_shape_and_reports_source_mismatch():
    f,_,_,_=build(mode='rest_rigid')
    assert f.mount_reference_audit['tab']['source_rigid_fit_error_m']>.001
    assert np.linalg.norm(f.point(4)-f.point(5))==pytest.approx(.1)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError,match='Unknown'):build(mode='auto')


@pytest.mark.parametrize('evidence',[None,'', '  ',123])
def test_source_mode_requires_explicit_evidence(evidence):
    with pytest.raises(ValueError,match='evidence'):
        mount_mode({'motion_mode':'source_skinned_attachment','motion_mode_evidence':evidence})


def test_legacy_mount_defaults_to_rigid():
    assert mount_mode({'part_id':'legacy'})=='rest_rigid'


def test_reference_rebuilt_from_next_source_not_previous_simulation():
    a,_,_,_=build(scale=1.12);a.states['tab'][1]+=[.3,.2,.1]
    b,_,_,_=build(scale=.9)
    assert np.linalg.norm(a.local['tab'][4]-a.local['tab'][5])==pytest.approx(.112)
    assert np.linalg.norm(b.local['tab'][4]-b.local['tab'][5])==pytest.approx(.09)
    np.testing.assert_allclose(b.positions(),b.source,atol=1e-15)


def test_prediction_does_not_change_source_reference():
    a,_,_,_=build();b,_,_,_=build(offset={'tab':[0,0,.01]})
    np.testing.assert_array_equal(a.reference,b.reference)
    np.testing.assert_array_equal(a.source,b.source)


def test_actual_contact_force_preserves_source_shape_and_primary_points():
    f,c,_,_=build();source=f.source.copy();f.x[:4,2]-=.019;f.states['tab'][1][2]-=.019
    out,report=solve_coupled_surface(f,c)
    assert report['candidate_committed'],report
    assert out[:7,2].min()>=.002-2e-6
    before=source[4:7];after=out[4:7]
    np.testing.assert_allclose(np.linalg.norm(after[:,None]-after,axis=2),np.linalg.norm(before[:,None]-before,axis=2),atol=1e-12)
    np.testing.assert_array_equal(out[7:],source[7:]);np.testing.assert_array_equal(f.source,source)
    assert report['candidate_shape_audit']['attachment_residual_m']<=2e-6


def test_source_reference_does_not_hide_later_cloth_stretch():
    f,_,_,_=build();f.x[3,0]+=1
    assert f.audit()['edges_over_25_percent']>0


def test_mount_only_solver_honors_explicit_source_mode():
    f,_,compiled,conn=build()
    out,report=project_mounts(f.source,f.mapping,compiled,conn['source_root_vertex_ids'],np.ones(10),{'tab':[1.,1.]})
    np.testing.assert_allclose(out,f.source,atol=1e-15)
    assert report['mount_references']['tab']['motion_mode']=='source_skinned_attachment'
    assert report['source_rigid_fit_error_m']['tab']>.001


def test_failed_source_mode_candidate_rolls_back():
    f,c,_,_=build();f.x[3,2]+=.1;incoming=f.positions().copy()
    out,report=solve_coupled_surface(f,c,iterations=1)
    assert not report['candidate_committed']
    np.testing.assert_array_equal(out,f.source);np.testing.assert_array_equal(f.positions(),incoming)


def test_compiled_rin_modes_are_explicit_and_keep_all_anchors(real_mounts):
    p,review,m,c=real_mounts;compiled=compile_mounts(p,review,m,c)
    assert sum(len(v['anchors']) for v in compiled['mounts'])==158
    assert sum(v['motion_mode']=='source_skinned_attachment' for v in compiled['mounts'])==4
    assert sum(v['motion_mode']=='rest_rigid' for v in compiled['mounts'])==1
    assert not compiled['simulation_ready']


def test_compiler_rejects_unreviewed_mode(real_mounts):
    p,r,m,c=copy.deepcopy(real_mounts);c['mounts'][0]['motion_mode']='sliding'
    with pytest.raises(ValueError,match='Unknown'):compile_mounts(p,r,m,c)


def test_compiler_preserves_legacy_mode_without_auto_detection(real_mounts):
    p,r,m,c=copy.deepcopy(real_mounts)
    for row in c['mounts']:row.pop('motion_mode',None);row.pop('motion_mode_evidence',None)
    assert all(v['motion_mode']=='rest_rigid' for v in compile_mounts(p,r,m,c)['mounts'])
