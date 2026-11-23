import copy,json
from pathlib import Path
import pytest
from motionbridge.secondary_guard import attachment_repair_gate,prewarm_stop_gate
from motionbridge.secondary_surface_config import surface_settings
from motionbridge.secondary_cache import atomic_json
from test_secondary_workbench import work,start

def audit():
 row=dict(samples=9,body_cloth_crossing_pairs=100,body_hair_crossing_pairs=10,cloth_hair_crossing_pairs=10,
  accessory_body_crossing_pairs=10,accessory_cloth_crossing_pairs=10,accessory_hair_crossing_pairs=10,
  dynamic_floor_frames_over_1mm=0,dynamic_floor_vertex_samples_over_1mm=0,dynamic_floor_max_depth_m=0.)
 out=dict(protection_passed=True,before=row,after=copy.deepcopy(row));out['after']['body_cloth_crossing_pairs']=90;return out

def warm(passed=True):
 return {'prewarm_convergence':dict(passed=passed,trailing_frames=30,maximum_frame_p95_m_s=.01,maximum_vertex_speed_m_s=.02 if passed else .85)}

@pytest.mark.parametrize('channel',['body','cloth','hair'])
def test_improved_cloth_cannot_pay_for_worse_accessory_contact(channel):
 a=audit();key='accessory_'+channel+'_crossing_pairs';a['after'][key]+=1
 gate=attachment_repair_gate(a);assert not gate['accepted'] and key in gate['rejection_reasons']

def test_equal_cloth_is_not_an_improvement_and_baseline_is_before():
 a=audit();a['after']['body_cloth_crossing_pairs']=100
 gate=attachment_repair_gate(a);assert not gate['accepted'] and gate['reference']=='captured_input_before'
 assert 'body_cloth_not_strictly_improved' in gate['rejection_reasons']

def test_joint_nonworsening_and_strict_cloth_improvement_can_pass():
 assert attachment_repair_gate(audit())['accepted']

@pytest.mark.parametrize('channel',['body','cloth','hair'])
def test_missing_historical_attachment_channel_is_not_pass(channel):
 a=audit();key='accessory_'+channel+'_crossing_pairs'
 del a['before'][key];del a['after'][key]
 assert not attachment_repair_gate(a)['accepted']

@pytest.mark.parametrize('report',[{},warm(False),{'prewarm_convergence':{'passed':True}},{'prewarm_convergence':dict(passed=True,trailing_frames=30,maximum_frame_p95_m_s=.01,maximum_vertex_speed_m_s=.85)}])
def test_missing_unsettled_or_false_label_warmup_stops(report):
 assert not prewarm_stop_gate(report,surface_settings({}))['accepted']

def test_verified_warmup_may_continue():
 assert prewarm_stop_gate(warm(),surface_settings({}))['accepted']

@pytest.mark.parametrize('case',['warm_failed','contact_failed','cloth_equal','passed'])
def test_pipeline_stops_without_creating_automatic_fallbacks(tmp_path,monkeypatch,case):
 from motionbridge import secondary_pipeline as p
 root=tmp_path/'project';root.mkdir();source=root/'source.blend';source.write_bytes(b'input');blender=root/'blender.exe';blender.write_bytes(b'fake')
 atomic_json(root/'profiles/secondary_motion/rin.json',{'hair_dynamics':{}});atomic_json(root/'config.json',{'blender_path':str(blender)})
 output=root/'output/secondary_motion/test';calls=[];report=audit()
 if case=='contact_failed':report['after']['accessory_cloth_crossing_pairs']=11
 if case=='cloth_equal':report['after']['body_cloth_crossing_pairs']=100
 class Session:
  def __init__(self,*args,**kwargs):self.revision_root=output/'revision';self.revision_root.mkdir(parents=True);self.revision='test';self.state={}
  def __enter__(self):return self
  def __exit__(self,*args):return False
  def save(self):pass
  def utility(self,command):atomic_json(command[command.index('--output')+1],{})
  def stage(self,name,key,parent,producer,files,**kwargs):
   calls.append(name);dest=Path(parent)/name;dest.mkdir()
   if name=='cloth':atomic_json(dest/'cloth-validation.json',warm(case!='warm_failed'))
   if name=='bundle-coupled':
    atomic_json(dest/'surface-audit.json',report);atomic_json(dest/'surface-validation.json',{'garment_deformation_gate':{'passed':True}})
   return dest
 monkeypatch.setattr(p,'RunSession',Session);monkeypatch.setattr(p,'code_hashes',lambda root:{})
 monkeypatch.setattr(p,'pipeline_keys',lambda *args:{k:k for k in ('capture','cloth','hair','bundle-coupled')})
 monkeypatch.setattr(p,'canonical_prepare',lambda *args:None)
 result=p.run_pipeline(root,source,output,'rin',use_cache=False)
 assert calls==(['capture','cloth'] if case=='warm_failed' else ['capture','cloth','hair','bundle-coupled'])
 assert result['physics_result_published']==(case=='passed')
 latest=json.loads((output/'latest.json').read_text(encoding='utf-8'))
 assert latest['quality']['automatic_retry'] is False
 if case=='warm_failed':assert latest['quality']['last_experimental_candidate'] is None

def test_workbench_prewarm_rejection_is_terminal_without_preview_or_export(work,monkeypatch):
 w,c,scene,pool=work;job=start(work,mode='surface').json();jid=job['id'];calls=[]
 def reject(jid,command,managed):
  calls.append(command);atomic_json(Path(job['output'])/'latest.json',{'case':job['output'],'quality':{'selected':None,'last_experimental_candidate':None,'stopped_stage':'cloth','stop_reason':'预热未稳定，停止'}});return 3
 monkeypatch.setattr(w,'_call',reject);w.execute(jid,False);row=w.get(jid)
 assert len(calls)==1 and row['status']=='quality_rejected' and row['result'] is None
 assert row['protection_passed'] is None and row['physics_quality']=='rejected' and not row.get('preview_dir')
 assert c.post(f'/api/secondary/runs/{jid}/refresh-preview').status_code==409
 assert c.post(f'/api/secondary/runs/{jid}/export',json={'formats':['blender'],'experimental':True}).status_code==409
