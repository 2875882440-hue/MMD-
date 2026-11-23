import copy,json
from pathlib import Path
import pytest
from motionbridge.secondary_regression import review_measurements,regression_readiness,CHANNELS,review_archived_case,markdown_report
from test_secondary_workbench import work


def evidence():
    meta={'frame_start':1,'frame_end':2,'frames':2,'fps':30.,'shape_keys':81,
          'source_sha256':'a'*64,'protected_matrix_sha256':'b'*64,'original_actions':{'source':'c'*64}}
    def row(value):
        frames=[{'frame':f,'dynamic_floor_depth_m':0.,**{c:value if c=='body_cloth' else 4 for c in CHANNELS}} for f in (1.,1.5,2.)]
        r={'samples':3,'per_frame':frames,'dynamic_floor_frames_over_1mm':0,
           'dynamic_floor_vertex_samples_over_1mm':0,'dynamic_floor_max_depth_m':0.}
        for c in CHANNELS:r.update({c+'_crossing_pairs':sum(f[c] for f in frames),c+'_crossing_frames':3,c+'_max_pairs':max(f[c] for f in frames)})
        return r
    protect={'samples':3,'bones':48,'matrix_sha256':'b'*64,'source_actions':1,
             'all_source_actions_unchanged':True,'original_skin_and_shape_keys_unchanged':True}
    audit={'before':row(10),'after':row(9),'protection_passed':True,'attachment_audit_required':True,'frame_stride':1,
           'protected_body_boot_max_world_change_m':0.,'protection':{'before':copy.deepcopy(protect),'after':copy.deepcopy(protect)}}
    cloth={'frames':2,'fps':30.,'prewarm_convergence':{'passed':True,'trailing_frames':30,
        'maximum_frame_p95_m_s':.01,'maximum_vertex_speed_m_s':.02}}
    surface={'source_sha256':'a'*64,'frames':2,'garment_deformation_gate':{'passed':True,'relative_limit':.25,
       'maximum_additional_relative_edge_error':.1,'absolute_tolerance_m':5e-6,'frames_over_budget':0,'edge_samples_over_budget':0}}
    return meta,audit,cloth,surface


def test_numeric_success_is_not_new_native_or_visual_acceptance():
    report=review_measurements(*evidence())
    assert report['evidence_valid'] and report['numeric_regression_passed']
    assert not report['publication_accepted'] and not report['new_simulation'] and not report['visual_verified']


@pytest.mark.parametrize('defect',['missing_half','duplicate_frame','summary','nan','boolean','matrix','source','stride','shape_keys','body_changed'])
def test_report_integrity_failures_are_not_quality_passes(defect):
    m,a,c,s=evidence()
    if defect=='missing_half':a['after']['per_frame'].pop(1)
    if defect=='duplicate_frame':a['after']['per_frame'][1]['frame']=1.
    if defect=='summary':a['before']['body_cloth_crossing_pairs']+=1
    if defect=='nan':a['before']['per_frame'][0]['accessory_body']=float('nan')
    if defect=='boolean':a['after']['per_frame'][0]['accessory_body']=True
    if defect=='matrix':a['protection']['after']['matrix_sha256']='d'*64
    if defect=='source':s['source_sha256']='d'*64
    if defect=='stride':a['frame_stride']=2
    if defect=='shape_keys':m['shape_keys']=80
    if defect=='body_changed':a['protected_body_boot_max_world_change_m']=.001
    r=review_measurements(m,a,c,s);assert not r['evidence_valid'] and not r['numeric_regression_passed']


@pytest.mark.parametrize('channel',['accessory_body','accessory_cloth','accessory_hair'])
def test_each_accessory_channel_independently_blocks_regression(channel):
    m,a,c,s=evidence();a['after']['per_frame'][1][channel]+=3
    a['after'][channel+'_crossing_pairs']+=3;a['after'][channel+'_max_pairs']+=3
    r=review_measurements(m,a,c,s)
    assert r['evidence_valid'] and not r['numeric_regression_passed']
    peak=r['channels'][channel]['largest_increases'][0]
    assert peak['frame']==1.5 and peak['increase']==3 and peak['clip_seconds']==pytest.approx(.5/30)


def test_summary_improvement_does_not_hide_local_worsening_in_report():
    m,a,c,s=evidence();frames=a['after']['per_frame']
    for f,n in zip(frames,[1,12,1]):f['body_cloth']=n
    a['after']['body_cloth_crossing_pairs']=14;a['after']['body_cloth_max_pairs']=12
    r=review_measurements(m,a,c,s)
    assert r['numeric_regression_passed'] and r['channels']['body_cloth']['worsened_samples']==1
    assert not r['publication_accepted']


@pytest.mark.parametrize('defect',['false_warm_label','short_window','relaxed_shape','hidden_shape_error'])
def test_old_labels_or_relaxed_limits_do_not_authorize_pass(defect):
    m,a,c,s=evidence()
    if defect=='false_warm_label':c['prewarm_convergence']['maximum_vertex_speed_m_s']=.85
    if defect=='short_window':c['prewarm_convergence']['trailing_frames']=1
    if defect=='relaxed_shape':s['garment_deformation_gate']['relative_limit']=.5
    if defect=='hidden_shape_error':s['garment_deformation_gate']['frames_over_budget']=1
    assert not review_measurements(m,a,c,s)['numeric_regression_passed']


def test_metadata_flags_cannot_enable_missing_native_adapter():
    p={'surface_segmentation':{'required_for_surface_dynamics':True,'review_gate':{'accepted':True}},
       'secondary_contact_rules':{'simulation_ready':True,'blockers':[]}}
    r=regression_readiness(p)
    assert not r['native_execution_allowed'] and not r['short_run_started'] and not r['long_run_started']


def test_read_only_loader_refuses_evidence_outside_project(tmp_path):
    with pytest.raises(ValueError):review_archived_case(tmp_path,tmp_path/'outside',tmp_path/'outside/result')


def test_markdown_keeps_archived_scope_explicit():
    r=review_measurements(*evidence());text=markdown_report({'readiness':regression_readiness({}),'historical_cases':{'fixture':r}})
    assert '当前未执行新版' in text and '不等于原始长视频时间' in text and '短片失败不得启动长片' in text


def test_workbench_catalog_exposes_readiness_without_changing_hair(work,monkeypatch):
    from motionbridge.secondary_cache import read_json,atomic_json
    import motionbridge.secondary_plan as compiler
    manager,client,scene,pool=work
    path=manager.root/'profiles/secondary_motion/rin.json';p=read_json(path)
    p['surface_segmentation']={'required_for_surface_dynamics':True};atomic_json(path,p)
    monkeypatch.setattr(compiler,'compile_profile',lambda *args:{'surface_segmentation':{'required_for_surface_dynamics':True,
        'review_gate':{'accepted':False,'unclassified_faces':2,'unreviewed_overlap_groups':1}}})
    result=client.get('/api/secondary/catalog').json()['profiles'][0]
    assert result['hair'] and not result['surface']
    assert result['regression_preflight']['native_execution_allowed'] is False
