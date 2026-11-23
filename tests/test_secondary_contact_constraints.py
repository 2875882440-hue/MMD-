import numpy as np
import pytest
from motionbridge.secondary_contact_constraints import contact_pin_release,FloorContactGraph,bind_surface_points,transport_surface_points,rigid_floor_envelope

def test_contact_release_only_relaxes_incompatible_garment_and_keeps_body():
 pins=np.ones(6);faces=np.array([[0,1,2],[2,3,4],[3,4,5]]);unsafe=np.array([1,0,0,0,0,1],bool);protected=np.array([0,0,0,0,0,1],bool)
 released,depth=contact_pin_release(pins,faces,unsafe,protected,rings=3)
 assert released[0]==0 and 0<released[1]<1 and released[5]==1
 assert np.array_equal(pins,np.ones(6))
 safe,_=contact_pin_release(pins,faces,np.zeros(6,bool));assert np.array_equal(safe,pins)

def test_floor_corrects_only_garment_not_body_or_original_arrays():
 rest=np.array([[0,0,0],[.1,0,0],[0,.1,0]],np.float32);posed=rest+[0,0,-.02];posed[0,2]=-.03;copy=posed.copy()
 solver=FloorContactGraph(rest,[[0,1,2]],[True,False,False]);final,report=solver.solve(posed,0)
 assert np.array_equal(final[0],posed[0]) and np.all(final[1:,2]>=.002-1e-7)
 assert np.array_equal(posed,copy) and report['contact_vertices']==2

def test_floor_above_origin_and_no_contact_is_bitwise_identity():
 rest=np.array([[0,0,1],[.1,0,1],[0,.1,1]],np.float32);solver=FloorContactGraph(rest,[[0,1,2]])
 final,_=solver.solve(rest,.95);assert np.array_equal(final,rest)
 final,_=solver.solve(rest,1.2);assert np.all(final[:,2]>=1.202-1e-6)

def test_floor_welds_matching_uv_seams_without_changing_protected_data():
 rest=np.array([[0,0,0],[.1,0,0],[0,.1,0],[0,0,0],[.1,0,0],[0,-.1,0]],float)
 solver=FloorContactGraph(rest,[[0,1,2],[3,4,5]],weld_keys=['same']*6)
 final,_=solver.solve(rest-[0,0,.02],0)
 assert np.array_equal(final[0],final[3]) and np.array_equal(final[1],final[4])

@pytest.mark.parametrize('bad',[float('nan'),float('inf')])
def test_floor_rejects_nonfinite_constraint(bad):
 with pytest.raises(ValueError):FloorContactGraph(np.eye(3),[[0,1,2]]).solve(np.eye(3),bad)

def test_triangle_binding_follows_trim_beyond_the_old_three_centimeter_cutoff():
 rest=np.array([[0,0,0],[1,0,0],[0,1,0]],float);faces=np.array([[0,1,2]]);point=np.array([[.2,.2,.0626]])
 binding=bind_surface_points(point,rest,faces,max_distance=.2)
 assert binding['bound'][0] and binding['distance_m'][0]==pytest.approx(.0626)
 moved=transport_surface_points(binding,point,rest,rest+[.1,0,.2],faces)
 assert moved==pytest.approx(point+[.1,0,.2])
 assert np.array_equal(transport_surface_points(binding,point,rest,rest,faces),point)

def test_trim_offset_rotates_with_the_actual_triangle_patch():
 rest=np.array([[0,0,0],[1,0,0],[0,1,0]],float);faces=np.array([[0,1,2]]);point=np.array([[.2,.2,.08]])
 rotate=np.array([[0,0,1],[0,1,0],[-1,0,0]],float)
 binding=bind_surface_points(point,rest,faces)
 assert transport_surface_points(binding,point,rest,rest@rotate.T,faces)==pytest.approx(point@rotate.T)

def test_authored_skin_affinity_does_not_bind_to_the_opposite_overlapping_panel():
 rest=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,.004],[1,0,.004],[0,1,.004]],float);faces=np.array([[0,1,2],[3,4,5]])
 weights=[{'left':1}]*3+[{'right':1}]*3
 binding=bind_surface_points([[.2,.2,.001]],rest,faces,[{'right':1}],weights)
 assert binding['triangles'][0]==1

def test_degenerate_triangle_transport_falls_back_without_collapsing_trim():
 rest=np.array([[0,0,0],[1,0,0],[0,1,0]],float);faces=np.array([[0,1,2]]);point=np.array([[.2,.2,.08]])
 binding=bind_surface_points(point,rest,faces);collapsed=np.zeros_like(rest)
 final=transport_surface_points(binding,point,rest,collapsed,faces)
 assert np.isfinite(final).all() and final[0,2]==pytest.approx(.08)
 with pytest.raises(ValueError):bind_surface_points(point,collapsed,faces)

def rotation_y(angle):
 c,s=np.cos(angle),np.sin(angle);return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def test_rigid_floor_bound_catches_rotation_that_dips_between_safe_endpoints():
 local=np.array([[0,0,1],[0,.1,1],[.1,0,.9]],float);rotations=np.array([rotation_y(2*np.pi/3),rotation_y(4*np.pi/3)])
 centers=np.array([[0,0,.7],[0,0,.7]],float);before=centers.copy()
 assert min((local@rotations[0].T+centers[0])[:,2])>0
 assert min((local@rotations[1].T+centers[1])[:,2])>0
 final,lift=rigid_floor_envelope(local,rotations,centers,0)
 assert lift.max()>0 and np.array_equal(centers,before)
 for t in np.linspace(0,1,201):
  points=local@rotation_y(2*np.pi/3+t*2*np.pi/3).T+(1-t)*final[0]+t*final[1]
  assert points[:,2].min()>=.001-1e-9
  assert np.linalg.norm(points[1]-points[0])==pytest.approx(np.linalg.norm(local[1]-local[0]))

def test_rigid_floor_preserves_clear_trajectory_and_supports_nonzero_floor():
 local=np.array([[0,0,0],[1,0,0],[0,1,0]],float);r=np.repeat(np.eye(3)[None],2,axis=0);c=np.array([[0,0,2],[0,0,3]],float)
 clear,lift=rigid_floor_envelope(local,r,c,0);assert np.array_equal(clear,c) and not lift.any()
 final,lift=rigid_floor_envelope(local,r,c,2.5);assert final[:,2].min()>=2.501

def test_rigid_floor_rejects_shear_or_nonfinite_transforms():
 with pytest.raises(ValueError):rigid_floor_envelope(np.eye(3),np.zeros((2,3,3)),np.zeros((2,3)),0)
