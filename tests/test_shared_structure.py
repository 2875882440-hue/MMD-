import numpy as np
from test_secondary_coupled_surface import fixture

def shared():
    f,_=fixture()
    f.edges=np.array([[4,3]]);f.edge_lengths=f.lengths(f.source,f.edges);f.edge_colors=f.color(f.edges)
    f.bends=np.empty((0,2),int);f.bend_lengths=np.empty(0);f.bend_colors=[]
    f.x[3]+=[.02,.01,.03]
    return f

def test_shared_edge_reacts_on_cloth_and_whole_rigid():
    f=shared();before=f.positions();err=abs(np.linalg.norm(f.point(4)-f.point(3))-f.edge_lengths[0])
    f.project_structure();after=f.positions()
    assert abs(np.linalg.norm(f.point(4)-f.point(3))-f.edge_lengths[0])<err
    assert np.linalg.norm(after[3]-before[3])>0 and np.linalg.norm(after[4]-before[4])>0

def test_rigid_shadow_node_is_not_independently_changed():
    f=shared();shadow=f.x[4:7].copy();source=f.source.copy();f.project_structure()
    np.testing.assert_array_equal(f.x[4:7],shadow);np.testing.assert_array_equal(f.source,source)

def test_rigid_internal_lengths_preserved_under_shared_edge_force():
    f=shared();before=f.positions()[4:7];f.project_structure();after=f.positions()[4:7]
    np.testing.assert_allclose(np.linalg.norm(after[:,None]-after[None],axis=2),np.linalg.norm(before[:,None]-before[None],axis=2),atol=1e-12)

def test_zero_rigid_metric_only_moves_free_cloth():
    f=shared();f.metrics['tab'][:]=0;before=f.positions();f.project_structure()
    np.testing.assert_array_equal(f.positions()[4:7],before[4:7]);np.testing.assert_array_equal(f.positions()[7:],before[7:])
    assert np.linalg.norm(f.point(3)-before[3])>0

def test_same_body_distance_does_not_deform_body():
    f=shared();f.edges=np.array([[4,5]]);f.edge_lengths=f.lengths(f.source,f.edges);f.edge_colors=f.color(f.edges)
    before=f.positions();f.project_structure();np.testing.assert_allclose(f.positions(),before,atol=1e-14)
