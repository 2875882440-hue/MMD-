import copy
from types import SimpleNamespace
import numpy as np
import pytest
from motionbridge.secondary_coupled_surface import MountedClothFrame, cloth_structure, solve_coupled_surface
from motionbridge.secondary_coupled_contacts import SourceTriangleContacts, aabb_pairs, sweep_aabb_pairs


class FixtureMap:
    def __init__(self):
        # A two-triangle sheet, a small rigid tab bonded at three actual points,
        # and a real body triangle. Completely synthetic engineering fixture.
        self.rest=np.array([[0,0,.02],[.1,0,.02],[0,.1,.02],[.1,.1,.02],
                            [0,0,.02],[.1,0,.02],[0,.1,.02],
                            [-1,-1,0],[2,-1,0],[-1,2,0]],float)
        self.triangles=np.array([[0,1,2],[1,3,2],[4,5,6],[7,8,9]],int)
        self.parts=[{'id':'cloth','category':'outer_cloth','face_ids':[0,1]},
                    {'id':'tab','category':'rigid_ornament','face_ids':[2]},
                    {'id':'body','category':'protected_surface','face_ids':[3]}]
        self.canonical_vertex=np.arange(10);self.physical_triangles=self.triangles
        self.physical_face_ids=np.arange(4);self.fixed_canonical=np.array([7,8,9])
    def check_positions(self,x):
        if np.shape(x)!=self.rest.shape or not np.isfinite(x).all():raise ValueError('bad positions')
    def expand_delta(self,source,delta):
        if np.any(delta[self.fixed_canonical]):raise ValueError('protected delta')
        return source+delta


def fixture(floor=-.1):
    mapping=FixtureMap();mounts={'mounts':[{'part_id':'tab','canonical_vertex_ids':[4,5,6],
                                          'anchors':[[4,0],[5,1],[6,2]]}]}
    connections={'source_root_vertex_ids':[7,8,9],'stitch_vertex_pairs':[]}
    frame=MountedClothFrame(mapping.rest,mapping,mounts,connections,np.ones(10),{'tab':[1.,1.]})
    contact=SourceTriangleContacts(mapping,mounts,connections,gap=.002,floor_z=floor)
    return frame,contact


def test_cloth_structure_contains_shared_hinge_opposite_points():
    s=cloth_structure(FixtureMap())
    assert len(s['edges'])==5 and s['bends']==[(0,3)]
    assert not s['nonmanifold_edges'] and 'not_signed_dihedral' in s['bending_model']


def test_structure_moves_neighbour_not_only_attached_point():
    f,_=fixture();f.x[1,2]+=.01;before=f.x.copy()
    for _ in range(12):f.project_structure()
    assert np.linalg.norm(f.x[3]-before[3])>1e-5
    assert f.audit()['edges_over_25_percent']==0
    np.testing.assert_array_equal(f.x[7:],f.source[7:])


def test_rest_bonded_contacts_are_not_category_exclusions():
    f,c=fixture();r=c.scan(f.source)
    assert not r['blocking_reason'] and r['audit']['bonded_features']>0
    assert r['audit']['physical_triangle_count']==4
    assert r['audit']['proper_crossings']==0


def test_real_triangle_body_contact_structure_and_mount_same_state():
    f,c=fixture();f.x[:4,2]-=.019;f.states['tab'][1][2]-=.019
    incoming=f.positions();source=f.source.copy()
    out,r=solve_coupled_surface(f,c)
    assert r['candidate_committed'],r
    assert out[:7,2].min()>=.002-2e-6
    assert r['candidate_shape_audit']['attachment_residual_m']<=2e-6
    assert r['candidate_shape_audit']['edge_error_m']<=2e-6
    assert np.max(np.linalg.norm(out-incoming,axis=1))>1e-4
    np.testing.assert_array_equal(out[7:],source[7:])
    np.testing.assert_array_equal(f.source,source)
    assert not r['full_physics_accepted'] and not r['continuous_collision_checked']


def test_mount_rigid_shape_is_preserved_when_contacts_act():
    f,c=fixture();f.x[:4,2]-=.019;f.states['tab'][1][2]-=.019
    out,r=solve_coupled_surface(f,c);assert r['candidate_committed'],r
    np.testing.assert_allclose(np.linalg.norm(out[4:7,None]-out[None,4:7],axis=-1),
        np.linalg.norm(f.mapping.rest[4:7,None]-f.mapping.rest[None,4:7],axis=-1),atol=1e-12)


def test_fixed_floor_conflict_returns_source_and_restores_prediction():
    f,c=fixture(floor=.001);f.x[3,2]+=.002;incoming=f.x.copy()
    out,r=solve_coupled_surface(f,c,iterations=2)
    assert r['rolled_back'] and not r['candidate_committed']
    np.testing.assert_array_equal(out,f.source);np.testing.assert_array_equal(f.x,incoming)


def test_no_finite_budget_partial_publication():
    f,c=fixture();f.x[3,2]+=.02;incoming=f.positions().copy()
    out,r=solve_coupled_surface(f,c,iterations=1,tolerance=1e-12)
    assert not r['candidate_committed']
    np.testing.assert_array_equal(out,f.source);np.testing.assert_array_equal(f.positions(),incoming)


@pytest.mark.parametrize('iterations,tolerance',[(0,2e-6),(257,2e-6),(128,3e-6),(128,float('nan'))])
def test_cannot_relax_budget_or_threshold(iterations,tolerance):
    f,c=fixture()
    with pytest.raises(ValueError):solve_coupled_surface(f,c,iterations=iterations,tolerance=tolerance)


def test_existing_intersection_records_faces_and_refuses_exit_guess():
    f,c=fixture();x=f.source.copy();x[7:]=[[-.1,.05,-.1],[.2,.05,-.1],[.05,.05,.1]]
    r=c.scan(x)
    assert r['blocking_reason']=='intersecting_source_or_candidate_requires_authored_exit_side'
    assert r['audit']['proper_crossings']>0 and r['audit']['crossing_examples']
    assert 'outer_cloth/protected_surface' in r['audit']['proper_crossings_by_channel']


def test_ambiguous_unbonded_coplanar_surface_not_silently_ignored():
    f,c=fixture();x=f.source.copy();x[7:]=[[-1,-1,.02],[2,-1,.02],[-1,2,.02]]
    r=c.scan(x)
    assert r['blocking_reason']=='ambiguous_zero_distance_contact_requires_review'
    assert r['audit']['ambiguous_features']>0


def test_broadphase_budget_never_silently_samples():
    f,c=fixture()
    with pytest.raises(ValueError,match='budget'):
        aabb_pairs(f.source,f.mapping.triangles,.1,budget=1)


@pytest.mark.parametrize('distance',[.0005,.0015,.003])
def test_sweep_includes_separated_parallel_contact(distance):
    x=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,distance],[1,0,distance],[0,1,distance]])
    pairs=sweep_aabb_pairs(x,np.array([[0,1,2],[3,4,5]]),.002,100)
    assert ((0,1) in pairs)==(distance<=.002)


def test_sweep_matches_exhaustive_aabb_without_sampling():
    rng=np.random.default_rng(1120);x=rng.uniform(-1,1,(300,3));tri=np.arange(300).reshape(-1,3)
    # Small-fixture provider pads both boxes by gap, sweep pads by gap/2.
    assert set(sweep_aabb_pairs(x,tri,.002,10000))==set(aabb_pairs(x,tri,.001,10000))
    with pytest.raises(ValueError,match='budget'):sweep_aabb_pairs(x,tri,.002,1)


def test_bad_pair_provider_rejected():
    f,c=fixture();c.provider=lambda *a:[(0,1),(0,1)]
    with pytest.raises(ValueError,match='duplicate'):c.scan(f.source)


def test_body_and_cloth_clearances_remain_separate():
    f,c=fixture();c.body_gap=.025
    r=c.scan(f.source)
    assert not r['blocking_reason']
    assert r['audit']['maximum_deficit_m']==pytest.approx(.005)
    assert any(x['offset']==.025 for x in r['constraints'])


def test_final_contacts_rebuilt_after_structure_and_last_push():
    f,c=fixture();f.x[:4,2]-=.019;f.states['tab'][1][2]-=.019
    calls=[];scan=c.scan
    def recorded(x):calls.append(x.copy());return scan(x)
    c.scan=recorded;out,r=solve_coupled_surface(f,c)
    assert r['candidate_committed'] and len(calls)>=4
    np.testing.assert_array_equal(calls[-1],out)


def test_nonmanifold_bending_fails_closed():
    m=FixtureMap();m.triangles=np.r_[m.triangles,[[0,1,3],[0,1,2]]]
    m.physical_face_ids=np.arange(6);m.parts[0]['face_ids']+=[4,5]
    assert cloth_structure(m)['nonmanifold_edges']


@pytest.mark.parametrize('mass,metric,offset',[(float('nan'),[1,1],[0,0,0]),(1,[-1,1],[0,0,0]),(1,[0,1],[0,0,.01])])
def test_invalid_dof_inputs_are_rejected(mass,metric,offset):
    f,c=fixture()
    with pytest.raises(ValueError):MountedClothFrame(f.source,f.mapping,{'mounts':f.mounts},
        {'source_root_vertex_ids':[7,8,9],'stitch_vertex_pairs':[]},np.full(10,mass),{'tab':metric},{'tab':offset})
