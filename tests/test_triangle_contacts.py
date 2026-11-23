import numpy as np
import pytest
from motionbridge.triangle_contacts import geometry_reference_error
from motionbridge.triangle_contacts import closest_triangle,contact_samples,sample_points,surface_contacts,scatter_corrections,bound_surface,preserve_capsule_clearance,regularize_contact_delta,support_safe_track,contact_track_gate,select_contact_components,proper_triangle_crossings


def test_closest_triangle_covers_interior_edges_corners_and_degenerate_faces():
    tri=np.array([[0.,0,0],[1,0,0],[0,1,0]])
    p=np.array([[.2,.3,1],[.8,.8,1],[-1,-1,0],[.4,1,0]])
    triangles=np.repeat(tri[None],4,axis=0);triangles[-1,2]=triangles[-1,1]
    assert np.allclose(closest_triangle(p,triangles),[[.2,.3,0],[.5,.5,0],[0,0,0],[.4,0,0]])
    with pytest.raises(ValueError):closest_triangle(p,np.zeros((2,3,3)))


def test_interior_sampling_detects_contact_missed_by_corners_and_welds_duplicates():
    rest=np.array([[-1.,-1,.1],[1,-1,.1],[0,1,.1],[-1,-1,.1]])
    edges=np.array([[0,1],[1,2],[2,0]]);faces=np.array([[0,1,2],[3,1,2]])
    ids,weights,groups=contact_samples(rest,edges,faces)
    p=sample_points(rest,ids,weights)
    assert len(p)==7 and groups[0]==groups[3]
    assert any(np.allclose(x,[0,-1/3,.1]) for x in p)
    assert np.allclose(weights.sum(1),1)


def test_overlap_allowance_is_per_queried_triangle_and_distant_rest_grants_none():
    # Cloth starts intentionally 2mm inside face 0. Later a different body
    # patch crosses it; face 0's rest overlap must not exempt that new patch.
    rt=np.array([[[-1.,-1,0],[1,-1,0],[0,1,0]], [[-1,-1,1],[1,-1,1],[0,1,1]]])
    rest=np.array([[0.,0,-.002]]);p=rest.copy()
    def near0(points,radius):return np.array([[0.,0,0]]),np.array([0])
    depth,_=surface_contacts(p,rest,rt,rt,near0)
    assert depth[0]==0
    moved=rt.copy();moved[1,:,2]=0
    def near1(points,radius):return np.array([[0.,0,0]]),np.array([1])
    depth,normals=surface_contacts(p,rest,rt,moved,near1)
    assert np.allclose(depth,.0035) and np.allclose(normals,[[0,0,1]])


def test_degenerate_and_missing_triangle_queries_do_not_invent_forces():
    p=np.zeros((2,3));tri=np.zeros((1,3,3))
    query=lambda points,radius:(np.zeros_like(points),np.array([-1,0]))
    d,n=surface_contacts(p,p,tri,tri,query)
    assert not d.any() and np.isfinite(n).all()
    with pytest.raises(ValueError):surface_contacts(p,p,tri,tri,query,gap=.1,radius=.01)


def test_contact_distribution_preserves_duplicate_seams_and_inputs_with_hard_bounds():
    base=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,0]])
    edges=np.array([[0,1],[1,2],[2,3]]);faces=np.array([[0,1,2],[3,1,2]])
    ids,w,g=contact_samples(base,edges,faces);copy=base.copy()
    d=np.full(len(ids),.1);n=np.tile([0,0,1.],(len(ids),1))
    delta=scatter_corrections(d,n,ids,w,g,edges)
    assert np.allclose(delta[0],delta[3]) and np.array_equal(base,copy)
    p=bound_surface(base,base,base+delta*100,extra=.01,total=.12)
    assert np.linalg.norm(p-base,axis=1).max()<=.01+1e-12
    previous=base+np.array([0,0,.11])
    p=bound_surface(base,previous,previous+delta*100,extra=.02,total=.12)
    assert np.linalg.norm(p-base,axis=1).max()<=.12+1e-12


def test_support_planes_prevent_opposite_branch_and_retain_existing_overlaps():
    from motionbridge.surface_clearance import capsule_distance
    caps=np.array([[[0.,0,-1],[0,0,1]],[[.12,0,-1],[.12,0,1]]])
    baseline=np.array([[.05,0,0],[.04,0,0],[.05,0,1.03]])
    limits=np.full((2,3),.045);raw=baseline.copy()
    candidate=baseline+[-.025,.015,0]
    result=preserve_capsule_clearance(raw,baseline,candidate,caps,limits)
    for c,limit in zip(caps,limits):
        old=capsule_distance(baseline,*c)[0];new=capsule_distance(result,*c)[0]
        assert np.all(new>=np.minimum(old,limit)-1e-7)
    assert np.linalg.norm(result-baseline,axis=1).max()<=.03+1e-12
    assert np.array_equal(baseline,raw)


def test_rest_layer_order_is_winding_invariant_and_preserves_attachments():
    tri=np.array([[[-1.,-1,0],[1,-1,0],[0,1,0]]])
    rest=np.array([[0.,0,.02],[0,0,.0001]])
    crossed=np.array([[0.,0,-.004],[0,0,-.004]])
    query=lambda points,radius:(np.zeros_like(points),np.zeros(len(points),int))
    depth,n=surface_contacts(crossed,rest,tri,tri,query,rest_order=True)
    depth2,n2=surface_contacts(crossed,rest,tri[:,::-1],tri[:,::-1],query,rest_order=True)
    assert np.allclose(depth,[.0055,0]) and np.allclose(depth,depth2)
    assert np.allclose(n[0],n2[0])


def gate_fixture():
    raw=np.tile(np.array([[0.,0,0],[.02,0,0],[0,.02,0]]),(9,1,1))
    caps=np.tile(np.array([[[2.,0,-1],[2,0,1]]]),(9,1,1,1))
    limits=np.full((1,3),.03);edges=np.array([[0,1],[1,2],[2,0]])
    return raw,caps,limits,edges


def test_contact_gate_bounds_small_new_strain_and_rejects_new_spikes_and_penetration():
    raw,caps,limits,edges=gate_fixture();p=raw.copy();p[:,:,2]=.002
    p[:,1,0]+=.0003
    gate=contact_track_gate(raw,raw,p,caps,limits,edges,30)
    assert gate['accepted']
    p[:,1,0]+=.003
    assert 'incremental_edge_error' in contact_track_gate(raw,raw,p,caps,limits,edges,30)['rejection_reasons']
    p=raw.copy();p[4,:,2]=.005
    assert 'vertex_acceleration_max' in contact_track_gate(raw,raw,p,caps,limits,edges,30)['rejection_reasons']
    p=raw.copy();p[:,:,0]+=1.999
    assert 'new_capsule_contacts' in contact_track_gate(raw,raw,p,caps,limits,edges,30)['rejection_reasons']


def test_strain_regularizer_reduces_new_edge_error_without_changing_baseline():
    raw,caps,limits,edges=gate_fixture();p=raw[0].copy();p[1,0]+=.02
    baseline=raw[0].copy()
    result=regularize_contact_delta(raw[0],baseline,p,caps[0],limits,edges,np.arange(3))
    lengths=lambda x:np.linalg.norm(x[edges[:,0]]-x[edges[:,1]],axis=1)
    assert abs(lengths(result)-lengths(baseline)).max()<.00051
    assert np.array_equal(baseline,raw[0])


def test_combined_gate_allows_bounded_clean_edges_but_rejects_impulsive_change():
    raw,caps,limits,edges=gate_fixture();p=raw.copy();p[:,1,0]+=.0003
    final,report=select_contact_components(raw,raw,p,caps,limits,edges,raw[0],30)
    assert report['combined']['accepted']
    assert np.allclose(final,p)
    p=raw.copy();p[4,:,2]=.005
    final,report=select_contact_components(raw,raw,p,caps,limits,edges,raw[0],30)
    assert np.array_equal(final,raw) and report['components'][0]['strength']==0


def test_support_ceiling_is_static_welded_and_preserves_all_frame_capsule_bounds():
    from motionbridge.surface_clearance import capsule_distance
    rest=np.array([[.05,0,-.1],[.06,0,0],[.06,0,.1],[.05,0,-.1]])
    baseline=np.tile(rest,(9,1,1));caps=np.tile(np.array([[[0.,0,-1],[0,0,1]]]),(9,1,1,1))
    limits=np.full((1,4),.05);candidate=baseline.copy()
    candidate[:,:,2]+=.01;candidate[4,[0,3],0]-=.002
    edges=np.array([[0,1],[1,2],[2,3]])
    result,report=support_safe_track(baseline,candidate,caps,limits,edges,rest)
    assert not report['time_varying_weights']
    assert np.allclose(result[:,0],baseline[:,0]) and np.allclose(result[:,3],baseline[:,3])
    for f in range(9):
        assert np.all(capsule_distance(result[f],*caps[f,0])[0]>=.05-1e-10)
    assert np.array_equal(baseline,np.tile(rest,(9,1,1)))


def test_triangle_confirmation_separates_real_crossing_from_coplanar_and_grazing():
    a=np.array([[-1.,-1,0],[1,-1,0],[0,1,0]])
    cross=np.array([[0.,0,-1],[0,.5,1],[0,1,-1]])
    left=np.tile(a,(4,1,1));right=np.stack([cross,a,cross*[1,1,1e-7],np.zeros((3,3))])
    expected=[True,False,False,False]
    assert proper_triangle_crossings(left,right).tolist()==expected
    assert proper_triangle_crossings(right,left).tolist()==expected
    rotation=np.array([[0.,0,1],[1,0,0],[0,1,0]])
    assert proper_triangle_crossings(left@rotation+[4,5,6],right@rotation+[4,5,6]).tolist()==expected


def test_all_vertex_reference_rejects_wrong_geometry_even_if_contacts_could_drop():
    reference=np.zeros((4,3));actual=reference.copy();actual[2,1]=3e-6
    assert geometry_reference_error(actual,reference)[1]
    actual[2,1]=2e-5
    assert not geometry_reference_error(actual,reference)[1]
    with pytest.raises(ValueError):geometry_reference_error(actual,reference[:2])
