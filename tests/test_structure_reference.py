import copy
import numpy as np
import pytest
from test_secondary_coupled_surface import FixtureMap
from motionbridge.secondary_coupled_surface import MountedClothFrame, solve_coupled_surface
from motionbridge.secondary_structure_reference import check_reference


def make(scale=1.05, offset=None, stitches=None, fixed=False):
    m=FixtureMap()
    # Native cloth face shares two tab vertices; original anchors remain.
    m.triangles[0]=[4,5,2]
    x=m.rest.copy(); x[:7,0]*=scale
    mounts={'mounts':[{'part_id':'tab','canonical_vertex_ids':[4,5,6], 'anchors':[[4,0],[5,1],[6,2]]}]}
    connections={'source_root_vertex_ids':[0,7,8,9] if fixed else [7,8,9], 'stitch_vertex_pairs':stitches or []}
    return MountedClothFrame(x,m,mounts,connections,np.ones(10),{'tab':[1.,1.]},offset)


def test_rigid_and_shared_cloth_use_consistent_reference():
    f=make(); i=np.flatnonzero(np.all(f.edges==[4,5],axis=1))[0]
    assert f.edge_lengths[i]==pytest.approx(.1)
    assert check_reference(f,2e-6)['passed']
    assert check_reference(f,2e-6)['same_body_constraints_checked']>0


def test_reference_propagates_authored_mounts_without_mutating_source():
    f=make(); original=f.source.copy()
    for a,b in f.mounts[0]['anchors']: np.testing.assert_allclose(f.reference[a],f.reference[b],atol=1e-14)
    np.testing.assert_array_equal(f.source,original)
    assert not np.array_equal(f.reference,f.source)


def test_prediction_does_not_reset_material_targets():
    a=make();b=make(offset={'tab':[.2,.3,.4]})
    np.testing.assert_array_equal(a.reference,b.reference)
    np.testing.assert_array_equal(a.edge_lengths,b.edge_lengths)
    assert not np.array_equal(a.positions(),b.positions())


def test_unattached_cloth_keeps_source_reference():
    f=make(); np.testing.assert_array_equal(f.reference[3],f.source[3])
    np.testing.assert_array_equal(f.reference[7:],f.source[7:])


def test_corrupted_internal_target_fails_before_contact_scan():
    f=make();i=np.flatnonzero(np.all(f.edges==[4,5],axis=1))[0];f.edge_lengths[i]+=.001
    class NeverScan:
        def scan(self,x): raise AssertionError('infeasible reference must stop before scan')
    state=copy.deepcopy(f.states);out,r=solve_coupled_surface(f,NeverScan())
    assert r['reason']=='structural_reference_infeasible' and r['iterations_used']==0
    np.testing.assert_array_equal(out,f.source)
    for pid in state: np.testing.assert_array_equal(f.states[pid][1],state[pid][1])


def test_source_deformation_gate_not_hidden_by_new_targets():
    f=make(scale=2.)
    assert f.audit()['edges_over_25_percent']>0
    assert any(x['reason']=='rigid_invariant_exceeds_source_deformation_gate' for x in check_reference(f,2e-6)['issues'])


def test_coincidence_of_distinct_rigid_points_is_infeasible():
    f=make(stitches=[[0,1]])
    assert any(x['reason']=='distinct_rigid_points_forced_coincident' for x in check_reference(f,2e-6)['issues'])


def test_fixed_reference_conflict_is_not_silently_moved():
    f=make(fixed=True)
    assert any(x['reason']=='reference_moves_protected_point' for x in check_reference(f,2e-6)['issues'])
    np.testing.assert_array_equal(f.source,f.x)


def test_simulated_motion_cannot_rewrite_reference():
    f=make();reference=f.reference.copy();lengths=f.edge_lengths.copy()
    f.x[3]+=[.02,.01,.03]; f.project_connections();f.project_structure()
    np.testing.assert_array_equal(f.reference,reference);np.testing.assert_array_equal(f.edge_lengths,lengths)


def test_bending_targets_share_same_reference_policy():
    f=make();np.testing.assert_allclose(f.bend_lengths,f.lengths(f.reference,f.bends))
    assert f.structure['bending_model']=='opposite_vertex_distance_not_signed_dihedral'


def test_attachment_tolerance_is_included_in_infeasibility_bound():
    f=make();f.edges=np.array([[0,1]]);f.edge_lengths=np.array([.1])
    f.source[1]=f.source[0]+[.1/1.25001,0,0]
    r=check_reference(f,2e-6)
    assert r['passed'],r
    assert r['rigid_source_differences'][0]['attachment_slack_m']==4e-6


def test_reference_does_not_alias_input_or_rest_arrays():
    f=make();source=f.source.copy();rest=f.mapping.rest.copy()
    f.reference[0]+=1
    np.testing.assert_array_equal(f.source,source);np.testing.assert_array_equal(f.mapping.rest,rest)
