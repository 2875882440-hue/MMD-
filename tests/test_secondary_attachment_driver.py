import copy
import json
from pathlib import Path
import numpy as np
import pytest
from attachment_fixture import make_fixture
from motionbridge.secondary_attachment_driver import (AttachmentSystem,exp_rotation,rotation_vector,
    triangle_frame,reject_legacy_assembly,driver_status)


def frame(mode='surface_rigid',mass=1.,inertia=.1):
    p,r,b=make_fixture(mode);system=AttachmentSystem(p,r,b)
    kwargs={'rigid_inverse_mass':{'ornament':mass},'rigid_inverse_inertia':{'ornament':np.eye(3)*inertia}} if mode=='surface_rigid' else {}
    if mode=='bone_socket':kwargs={'bone_rest':{0:np.eye(4)},'bone_current':{0:np.eye(4)}}
    return system,system.begin_frame(system.rest,np.ones(len(system.rest)),**kwargs)


def test_unforced_rest_is_exact_and_no_collision_claim_is_made():
    s,f=frame();before=f.positions();result=f.project_attachments()
    assert result['attachment_converged'] and result['iterations_used']==0
    assert not result['collision_checked'] and not result['physics_accepted']
    np.testing.assert_array_equal(f.positions(),before)


def test_rigid_force_reacts_on_cloth_without_stretching_the_ornament():
    s,f=frame();f.rigid['ornament']['center']+=np.array([0,0,.06])
    before=f.positions();center=f.rigid['ornament']['center'].copy();mass_center=before[:3].sum(0)+center
    report=f.project_attachments(iterations=96,tolerance_m=2e-6);after=f.positions()
    assert report['attachment_converged'],report
    assert np.linalg.norm(after[:3]-before[:3])>.005
    assert np.linalg.norm(f.rigid['ornament']['center']-center)>.005
    np.testing.assert_allclose(after[:3].sum(0)+f.rigid['ornament']['center'],mass_center,atol=2e-8)
    np.testing.assert_allclose(np.linalg.norm(after[3:6,None]-after[None,3:6],axis=-1),np.linalg.norm(s.rest[3:6,None]-s.rest[None,3:6],axis=-1),atol=1e-10)
    np.testing.assert_array_equal(after[[6,9,10,11]],s.rest[[6,9,10,11]])


def test_cloth_force_is_shared_instead_of_copied_to_the_ornament():
    s,f=frame();f.x[:3]+=[.035,0,0];incoming=f.x[:3].copy();center=f.rigid['ornament']['center'].copy()
    report=f.project_attachments(iterations=96,tolerance_m=2e-6)
    assert report['attachment_converged'],report
    assert np.linalg.norm(f.x[:3]-incoming)>.001 and np.linalg.norm(f.rigid['ornament']['center']-center)>.001
    assert np.linalg.norm(f.rigid['ornament']['center']-center)<.035


def test_zero_mass_ornament_pushes_only_free_cloth():
    s,f=frame(mass=0.,inertia=0.);center=f.rigid['ornament']['center'].copy();f.x[:3]+=[0,0,.02]
    result=f.project_attachments(iterations=96,tolerance_m=2e-6)
    assert result['attachment_converged'],result
    np.testing.assert_array_equal(f.rigid['ornament']['center'],center)


def test_bone_socket_follows_source_bone_and_ignores_external_displacement():
    p,r,b=make_fixture('bone_socket');s=AttachmentSystem(p,r,b)
    rest=np.eye(4);rest[:3,3]=[.2,.3,.1];current=np.eye(4);current[:3,:3]=exp_rotation([0,0,1.7]);current[:3,3]=[1,2,3]
    original=copy.deepcopy((rest,current));f=s.begin_frame(s.rest,np.ones(len(s.rest)),bone_rest={0:rest},bone_current={0:current})
    expected=s.rest[3:6]@current[:3,:3].T+current[:3,3]-current[:3,:3]@rest[:3,3]
    f.rigid['ornament']['center']+=100.;f.rigid['ornament']['rotation']=exp_rotation([1.,0,0])
    assert f.project_attachments()['attachment_converged']
    np.testing.assert_allclose(f.positions()[3:6],expected,atol=1e-12)
    np.testing.assert_array_equal(rest,original[0]);np.testing.assert_array_equal(current,original[1])
    np.testing.assert_array_equal(f.x[s.fixed],s.rest[s.fixed])


def test_fixed_model_scale_cancels_in_relative_bone_transform():
    p,r,b=make_fixture('bone_socket');s=AttachmentSystem(p,r,b);rest=np.eye(4);rest[:3,:3]*=.08
    current=rest.copy();rotation=exp_rotation([0,.6,0]);current[:3,:3]=rotation*.08;current[:3,3]=[1,0,0]
    f=s.begin_frame(s.rest,np.ones(len(s.rest)),bone_rest={0:rest},bone_current={0:current})
    np.testing.assert_allclose(f.positions()[3:6],s.rest[3:6]@rotation.T+[1,0,0],atol=1e-12)


@pytest.mark.parametrize('scale_error,accepted',[(5.5e-7,True),(1e-5,False)])
def test_native_float32_axis_roundoff_is_distinguished_from_actual_scale(scale_error,accepted):
    p,r,b=make_fixture('bone_socket');s=AttachmentSystem(p,r,b);current=np.eye(4)
    current[:3,:3]=exp_rotation([.3,.1,.5])@np.diag([1+scale_error,1+scale_error,1+scale_error])
    if not accepted:
        with pytest.raises(ValueError,match='缩放'):s.begin_frame(s.rest,np.ones(len(s.rest)),bone_rest={0:np.eye(4)},bone_current={0:current})
    else:
        f=s.begin_frame(s.rest,np.ones(len(s.rest)),bone_rest={0:np.eye(4)},bone_current={0:current})
        np.testing.assert_allclose(f.rigid['ornament']['rotation'].T@f.rigid['ornament']['rotation'],np.eye(3),atol=1e-12)


def test_surface_seam_moves_only_declared_trim_anchors_and_target_patch():
    s,f=frame('surface_seam');f.x[3:6]+=[0,0,.04];incoming=f.x.copy()
    report=f.project_attachments(iterations=96,tolerance_m=2e-6)
    assert report['attachment_converged'],report
    assert np.linalg.norm(f.x[3:6]-incoming[3:6])>.001 and np.linalg.norm(f.x[:3]-incoming[:3])>.001
    np.testing.assert_array_equal(f.x[6:],incoming[6:])


def test_source_frame_turn_and_translation_do_not_consume_attachment_limits():
    p,r,b=make_fixture();s=AttachmentSystem(p,r,b);rot=exp_rotation([.4,-.3,2.5])
    source=s.rest@rot.T+np.array([12,-7,3]);before=source.copy()
    f=s.begin_frame(source,np.ones(len(source)),{'ornament':1.},{'ornament':np.eye(3)})
    result=f.project_attachments(tolerance_m=1e-8)
    assert result['attachment_converged'] and result['iterations_used']==0
    np.testing.assert_array_equal(source,before)
    np.testing.assert_allclose(f.positions(),source,atol=1e-12)


def test_limits_are_relative_to_source_pose_and_impossible_attachment_is_rejected():
    s,f=frame();f.weights[:]=0
    f.rigid['ornament']['center']+=np.array([1.,0,0]);f.rigid['ornament']['rotation']=exp_rotation([0,1.2,0])
    f.mass['ornament']=0.;f.inertia['ornament'][:]=0
    result=f.project_attachments(iterations=2)
    assert not result['attachment_converged'] and not result['physics_accepted']
    ref=f.reference['ornament'];state=f.rigid['ornament']
    assert np.linalg.norm(state['center']-ref['center'])<=.25+1e-12
    assert np.linalg.norm(rotation_vector(state['rotation']@ref['rotation'].T))<=.8+1e-12


def test_rotating_offset_and_target_reaction_are_in_the_same_jacobian():
    s,f=frame('surface_seam');c=s.constraints[0]
    jac=sum(f._jacobian(c,('vertex',int(v)),1e-6) for v in c['target'])
    np.testing.assert_allclose(jac,-np.eye(3),atol=1e-8)
    # A normal movement at one triangle corner also rotates the 2cm offset;
    # a barycentric-only follower would have a zero x/z cross derivative.
    j=f._jacobian(c,('vertex',1),1e-6)
    assert abs(j[0,2])>.01


@pytest.mark.parametrize('kind',['nonfinite','negative_mass','indefinite_inertia','scaled_bone','missing_bone','missing_mass','stale_geometry'])
def test_bad_driver_input_fails_instead_of_guessing(kind):
    p,r,b=make_fixture('bone_socket' if kind in ('scaled_bone','missing_bone') else 'surface_rigid')
    if kind=='stale_geometry':
        p['geometry']['positions_m'][0][0]+=.01
        with pytest.raises(ValueError):AttachmentSystem(p,r,b)
        return
    s=AttachmentSystem(p,r,b);source=s.rest.copy();weights=np.ones(len(source));kwargs={}
    if kind in ('scaled_bone','missing_bone'):
        matrix=np.eye(4);matrix[0,0]=2 if kind=='scaled_bone' else 1
        kwargs={'bone_rest':{0:np.eye(4)},'bone_current':{0:matrix} if kind=='scaled_bone' else {}}
    else:
        kwargs={'rigid_inverse_mass':{} if kind=='missing_mass' else {'ornament':-1. if kind=='negative_mass' else 1.},
                'rigid_inverse_inertia':{'ornament':np.diag([1,1,-1]) if kind=='indefinite_inertia' else np.eye(3)}}
    if kind=='nonfinite':source[0,0]=float('nan')
    with pytest.raises(ValueError):s.begin_frame(source,weights,**kwargs)


def test_fixed_primary_vertex_mutation_is_rejected():
    s,f=frame();f.x[9,0]+=1
    with pytest.raises(ValueError,match='固定源顶点'):f.project_attachments()


def test_collapsed_target_has_no_nearest_neighbor_or_orientation_fallback():
    s,f=frame();f.x[2]=f.x[1]
    with pytest.raises(ValueError,match='退化'):f.project_attachments()


@pytest.mark.parametrize('angle',[0.,1e-9,.7,np.pi-1e-7,np.pi])
def test_rotation_limits_remain_proper_near_half_turn(angle):
    rotation=exp_rotation(np.array([1.,2.,3.])/np.sqrt(14)*angle)
    np.testing.assert_allclose(exp_rotation(rotation_vector(rotation)),rotation,atol=1e-7)


def test_real_rin_has_driver_core_but_no_invented_binding_or_unlocked_solver():
    from motionbridge.secondary_plan import compile_profile
    root=Path(__file__).resolve().parents[1];p=json.loads((root/'profiles/secondary_motion/rin.json').read_text(encoding='utf8'))
    plan=compile_profile(p,root);status=plan['secondary_contact_rules']['attachment_driver']
    assert status['core_implemented'] and not status['binding_structure_checked'] and not status['production_ready']
    assert not plan['secondary_contact_rules']['simulation_ready']


def test_legacy_assembly_cannot_reuse_unreviewed_cached_bindings():
    with pytest.raises(RuntimeError,match='旧最终衣料'):reject_legacy_assembly()
