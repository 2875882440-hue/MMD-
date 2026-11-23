from types import SimpleNamespace
import numpy as np
from motionbridge.secondary_structure_reference import build_reference

def fixture(weights=(1.,1.,1.),pairs=None):
    source=np.array([[0.,0,0],[.00035,0,0],[.02,.01,0]])
    f=SimpleNamespace(source=source,weights=np.array(weights),stitches=pairs or [[0,1]],mounts=[],owner={},
                      mapping=SimpleNamespace(canonical_vertex=np.arange(3)))
    ref,bindings,issues=build_reference(f,source.copy())
    return f,ref,issues

def test_equal_mass_stitch_has_common_midpoint():
    f,r,e=fixture();np.testing.assert_allclose(r[:2],[[.000175,0,0]]*2);assert not e

def test_material_lengths_to_shared_neighbour_are_consistent():
    f,r,e=fixture();assert abs(np.linalg.norm(r[0]-r[2])-np.linalg.norm(r[1]-r[2]))<1e-15
    assert abs(np.linalg.norm(f.source[0]-f.source[2])-np.linalg.norm(f.source[1]-f.source[2]))>6e-6

def test_unequal_inverse_masses_preserve_mass_center():
    f,r,e=fixture((1.,3.,1.));np.testing.assert_allclose(r[0],[.0000875,0,0])

def test_locked_endpoint_reference_not_moved():
    f,r,e=fixture((0.,1.,1.));np.testing.assert_array_equal(r[0],f.source[0]);np.testing.assert_array_equal(r[1],f.source[0])

def test_conflicting_immovable_points_fail_closed():
    f,r,e=fixture((0.,0.,1.));assert e[0]['reason']=='incompatible_immovable_stitch_reference'
    np.testing.assert_array_equal(r,f.source)

def test_transitive_stitch_group_has_one_reference():
    f,r,e=fixture(pairs=[[0,1],[1,2]]);np.testing.assert_allclose(r,np.tile(f.source.mean(0),(3,1)))

def test_source_and_unconnected_points_unchanged():
    f,r,e=fixture();np.testing.assert_array_equal(f.source[1],[.00035,0,0]);np.testing.assert_array_equal(r[2],f.source[2])

def test_reference_does_not_alias_input():
    f,r,e=fixture();saved=f.source.copy();r[:]=9;np.testing.assert_array_equal(f.source,saved)
