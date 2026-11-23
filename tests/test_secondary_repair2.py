import numpy as np
import pytest
from motionbridge.secondary_contact_constraints import FloorContactGraph,mesh_edges,garment_edge_deformation,regularized_attachment_fit
from motionbridge.secondary_guard import actual_surface_gate

def mixed_scale_mesh():
 # A short seam edge attached to larger faces; averaging corrections lets the
 # short edge retain large relative error even with small absolute movement.
 x=np.array([[0,0,1],[.002,0,1],[.03,.03,1],[-.02,.03,1],[.03,-.03,1],[-.02,-.03,1]],float)
 f=np.array([[0,1,2],[0,2,3],[0,3,5],[0,5,4],[0,4,1]])
 return x,f

def test_short_seam_edge_is_repaired_with_uneven_triangle_scales():
 x,f=mixed_scale_mesh();candidate=x.copy();candidate[1]+=[.012,.003,0]
 final,report=FloorContactGraph(x,f).solve(candidate,0,iterations=256,reference_positions=x)
 assert garment_edge_deformation(x,final,mesh_edges(f),relative_limit=.06)['passed']
 assert report['contact_vertices']==0 and np.isfinite(final).all()

@pytest.mark.parametrize('offset',[0.,1.5,-.2])
def test_length_projection_and_floor_remain_joint_constraints(offset):
 x,f=mixed_scale_mesh();x[:,2]=offset+.01;candidate=x.copy();candidate[1]+=[.012,0,-.04]
 final,_=FloorContactGraph(x,f).solve(candidate,offset,iterations=256,reference_positions=x)
 assert final[:,2].min()>=offset+.002-1e-7
 assert garment_edge_deformation(x,final,mesh_edges(f))['passed']

def test_colored_edges_never_share_a_vertex_in_one_batch():
 x,f=mixed_scale_mesh();g=FloorContactGraph(x,f)
 for color in np.unique(g.edge_colors):
  vertices=g.edges[g.edge_colors==color].ravel()
  assert len(vertices)==len(np.unique(vertices))

def test_projection_keeps_protected_anchor_and_inputs_exact():
 x,f=mixed_scale_mesh();candidate=x.copy();candidate[1,0]=.03;saved=candidate.copy()
 final,_=FloorContactGraph(x,f,[True,False,False,False,False,False]).solve(candidate,0,iterations=256,reference_positions=x)
 assert np.array_equal(candidate,saved) and np.array_equal(final[0],saved[0])
 assert garment_edge_deformation(x,final,mesh_edges(f))['passed']

def test_uv_seams_keep_consistent_shape_under_colored_projection():
 x,f=mixed_scale_mesh();rest=np.concatenate([x,x]);faces=np.concatenate([f,f+len(x)]);candidate=rest.copy();candidate[[1,7],0]=.014
 final,_=FloorContactGraph(rest,faces,weld_keys=['seam']*len(rest)).solve(candidate,0,iterations=256,reference_positions=rest)
 assert np.array_equal(final[:6],final[6:])
 assert garment_edge_deformation(rest,final,mesh_edges(faces))['passed']

@pytest.mark.parametrize('channel',['body','cloth','hair'])
def test_attachment_regression_cannot_be_hidden_by_clean_cloth(channel):
 row=dict(samples=9,body_cloth_crossing_pairs=0,body_hair_crossing_pairs=0,cloth_hair_crossing_pairs=0,dynamic_floor_frames_over_1mm=0,dynamic_floor_vertex_samples_over_1mm=0,dynamic_floor_max_depth_m=0,accessory_body_crossing_pairs=0,accessory_cloth_crossing_pairs=0,accessory_hair_crossing_pairs=0)
 after=dict(row);key='accessory_'+channel+'_crossing_pairs';after[key]=1
 gate=actual_surface_gate(dict(before=row,after=after,protection_passed=True,attachment_audit_required=True))
 assert not gate['accepted'] and key in gate['rejection_reasons']

def test_missing_attachment_measurement_is_not_silently_accepted():
 row=dict(samples=9,body_cloth_crossing_pairs=0,body_hair_crossing_pairs=0,cloth_hair_crossing_pairs=0,dynamic_floor_frames_over_1mm=0,dynamic_floor_vertex_samples_over_1mm=0,dynamic_floor_max_depth_m=0)
 gate=actual_surface_gate(dict(before=row,after=dict(row),protection_passed=True,attachment_audit_required=True))
 assert not gate['accepted'] and 'attachment_contacts_unverified' in gate['rejection_reasons']

def test_trim_constraints_preserve_final_garment_even_at_a_welded_attachment():
 # Two panels sharing an authored attachment point: final cloth is immutable
 # while the free decorative triangle recovers its lengths above the floor.
 rest=np.array([[0,0,.02],[.03,0,.02],[0,.03,.02],[0,0,.02],[.01,0,.02],[0,.01,.02]],float)
 faces=np.array([[0,1,2],[3,4,5]]);candidate=rest.copy();candidate[4]+=[.015,0,-.03]
 final,_=FloorContactGraph(rest,faces,[True]*3+[False]*3,weld_keys=['attached']*6).solve(candidate,0,iterations=256,reference_positions=rest)
 assert np.array_equal(final[:3],candidate[:3]) and np.array_equal(final[0],final[3])
 assert final[3:,2].min()>=.002-1e-8
 assert garment_edge_deformation(rest,final,mesh_edges(faces))['passed']

@pytest.mark.parametrize('scale',[.001,1.,100.])
def test_thin_patch_cannot_turn_tiny_deformation_into_a_half_turn(scale):
 a=np.array([[-.02,0,0],[0,.00001,0],[.02,0,0]])*scale;b=a.copy();b[:,1]*=-1
 r,t=regularized_attachment_fit(a,b)
 assert np.arccos(np.clip((np.trace(r)-1)/2,-1,1))<.01
 assert np.linalg.det(r)==pytest.approx(1.) and np.allclose(r.T@r,np.eye(3))
 assert np.allclose(a.mean(0)@r.T+t,b.mean(0))

@pytest.mark.parametrize('patch',[np.eye(3),np.array([[0,0,0],[1,0,0],[2,0,0]]),np.ones((3,3))])
def test_rigid_translation_is_exact_for_full_rank_line_and_point_patches(patch):
 move=np.array([.1,-.2,.3]);r,t=regularized_attachment_fit(patch,patch+move)
 assert np.allclose(r,np.eye(3),atol=1e-12) and np.allclose(t,move,atol=1e-12)

def test_rotation_prior_is_coordinate_and_scale_invariant():
 a=np.array([[0,0,0],[.1,0,0],[0,.1,0]]);b=a.copy();b[:,2]=[0,.01,.02]
 r,t=regularized_attachment_fit(a,b);q=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
 rr,tt=regularized_attachment_fit(10*a@q.T,10*b@q.T)
 assert np.allclose(rr,q@r@q.T) and np.allclose(tt,10*q@t)

@pytest.mark.parametrize('prior',[0,-1,float('nan'),float('inf'),11])
def test_invalid_attachment_prior_is_rejected(prior):
 with pytest.raises(ValueError):regularized_attachment_fit(np.eye(3),np.eye(3),prior)
