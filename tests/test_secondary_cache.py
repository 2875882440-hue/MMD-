import copy,json,subprocess,sys,threading,time
from pathlib import Path
import pytest
from motionbridge.secondary_cache import ArtifactCache,BusyError,file_sha,key_for,lease,stage_keys
from motionbridge.secondary_tasks import Cancelled,RunSession

def test_artifact_cache_copy_isolation_and_corruption_rejection(tmp_path):
 source=tmp_path/'source';source.mkdir();(source/'data.bin').write_bytes(b'completed computation')
 cache=ArtifactCache(tmp_path/'cache');key=key_for({'source':'fixed'})
 cache.publish('cloth',key,source,['data.bin'],provenance={'physics_quality':False})
 target=tmp_path/'restored';assert cache.restore('cloth',key,target)
 (target/'data.bin').write_bytes(b'working edit')
 assert cache.verify('cloth',key)
 (cache.entry('cloth',key)/'payload/data.bin').write_bytes(b'corrupt')
 assert cache.verify('cloth',key) is None
 assert cache.restore('cloth',key,tmp_path/'not-created') is None
 assert not (tmp_path/'not-created').exists()
 cache.publish('cloth',key,source,['data.bin'])
 assert cache.verify('cloth',key) and len(list((cache.root/'quarantine').iterdir()))==1
 assert (source/'data.bin').read_bytes()==b'completed computation'

@pytest.mark.parametrize('name',['../outside','C:/outside','/absolute','a\\..\\outside'])
def test_cache_rejects_escaping_payload_names(tmp_path,name):
 cache=ArtifactCache(tmp_path/'cache');source=tmp_path/'source';source.mkdir()
 with pytest.raises(ValueError):cache.publish('cloth',key_for(1),source,[name])

def test_failed_cache_prepare_never_becomes_reusable(tmp_path):
 source=tmp_path/'source';source.mkdir();(source/'a').write_text('a');cache=ArtifactCache(tmp_path/'cache');key=key_for(2)
 def fail(*_):raise RuntimeError('injected interrupted normalization')
 with pytest.raises(RuntimeError):cache.publish('hair',key,source,['a'],prepare=fail)
 assert cache.verify('hair',key) is None and list((cache.root/'pending').rglob('failed.json'))

def test_live_lease_rejects_duplicate_work_and_releases_only_its_lock(tmp_path):
 path=tmp_path/'lock.json'
 with lease(path):
  with pytest.raises(BusyError):
   with lease(path):pass
 assert not path.exists()

def test_stage_key_parameter_invalidation_is_selective():
 profile={'model':{'sha256':'same'},'hair_dynamics':{'drag':30},'surface_dynamics':{'mass':.05,'floor_z_m':0}}
 code={k:'fixed' for k in ['capture','hair','cloth']};base=stage_keys({'source':'same'},profile,code)
 changed=copy.deepcopy(profile);changed['hair_dynamics']['drag']=31;x=stage_keys({'source':'same'},changed,code)
 assert x['capture']==base['capture'] and x['cloth']==base['cloth'] and x['hair']!=base['hair']
 changed=copy.deepcopy(profile);changed['surface_dynamics']['mass']=.06;x=stage_keys({'source':'same'},changed,code)
 assert x['capture']==base['capture'] and x['hair']==base['hair'] and x['cloth']!=base['cloth']
 changed=copy.deepcopy(profile);changed['surface_dynamics']['floor_z_m']=1;x=stage_keys({'source':'same'},changed,code)
 assert x['hair']!=base['hair'] and x['cloth']!=base['cloth']
 changed=copy.deepcopy(code);changed['hair']='solver update';x=stage_keys({'source':'same'},profile,changed)
 assert x['capture']==base['capture'] and x['cloth']==base['cloth'] and x['hair']!=base['hair']
 assert stage_keys({'source':'changed'},profile,code)['capture']!=base['capture']

def test_resume_preserves_previous_revision_and_reuses_identical_stage(tmp_path):
 root=tmp_path/'run';cache=ArtifactCache(tmp_path/'cache');calls=[]
 def produce(path,_):path.mkdir(parents=True);(path/'a').write_text('correct');calls.append(path)
 with RunSession(root,{'params':1},cache) as run:first=run.stage('cloth',key_for('cloth1'),run.revision_root,produce,['a'])
 with RunSession(root,{'params':1},cache,resume=True) as run:
  assert run.stage('cloth',key_for('cloth1'),run.revision_root,produce,['a'])==first
 with RunSession(root,{'params':2},cache,resume=True) as run:second=run.stage('cloth',key_for('cloth1'),run.revision_root,produce,['a'])
 assert len(calls)==1 and first!=second and first.is_dir() and second.is_dir()
 assert json.loads((root/'run.json').read_text())['status']=='completed'

def test_modified_working_artifact_restores_cache_without_overwriting_old_attempt(tmp_path):
 root=tmp_path/'run';cache=ArtifactCache(tmp_path/'cache')
 def produce(path,_):path.mkdir(parents=True);(path/'a').write_text('correct')
 with RunSession(root,{},cache) as run:old=run.stage('hair',key_for('h'),run.revision_root,produce,['a'])
 (old/'a').write_text('manual edit')
 with RunSession(root,{},cache,resume=True) as run:new=run.stage('hair',key_for('h'),run.revision_root,produce,['a'])
 assert (old/'a').read_text()=='manual edit' and (new/'a').read_text()=='correct' and old!=new

def test_cooperative_cancel_terminates_only_the_owned_child_and_retains_partial(tmp_path):
 root=tmp_path/'run';sibling=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'])
 stop=threading.Event()
 def cancel():
  while not stop.is_set():
   if list(root.rglob('partial.txt')):RunSession.cancel(root);return
   time.sleep(.03)
 worker=threading.Thread(target=cancel);worker.start()
 try:
  with pytest.raises(Cancelled):
   with RunSession(root,{}) as run:
    def produce(path,execute):
     path.mkdir(parents=True)
     execute([sys.executable,'-c',"from pathlib import Path; import sys,time; Path(sys.argv[1]).write_text('partial'); time.sleep(120)",str(path/'partial.txt')])
    run.stage('cancel-test',key_for('cancel'),run.revision_root,produce,['partial.txt'])
  assert list(root.rglob('partial.txt')) and sibling.poll() is None
  state=json.loads((root/'run.json').read_text());assert state['status']=='cancelled'
  assert all(r.get('child_finished') for r in state['stages'].values())
 finally:
  stop.set();worker.join(timeout=5);sibling.terminate();sibling.wait(timeout=5)

def test_resume_after_interrupted_stage_reuses_completed_stage_only(tmp_path):
 root=tmp_path/'run';cache=ArtifactCache(tmp_path/'cache');calls=[];failed=[]
 def a(path,_):path.mkdir(parents=True);(path/'a').write_text('complete');calls.append('a')
 def b_fail(path,_):
  path.mkdir(parents=True);(path/'partial').write_text('retained');failed.append(path);raise RuntimeError('interrupted')
 with pytest.raises(RuntimeError):
  with RunSession(root,{},cache) as run:
   run.stage('a',key_for('a'),run.revision_root,a,['a']);run.stage('b',key_for('b'),run.revision_root,b_fail,['done'])
 def b_ok(path,_):path.mkdir(parents=True);(path/'done').write_text('complete');calls.append('b')
 with RunSession(root,{},cache,resume=True) as run:
  run.stage('a',key_for('a'),run.revision_root,a,['a']);new=run.stage('b',key_for('b'),run.revision_root,b_ok,['done'])
 assert calls==['a','b'] and (failed[0]/'partial').read_text()=='retained' and new!=failed[0]

def test_stage_output_cannot_escape_managed_run(tmp_path):
 with RunSession(tmp_path/'run',{}) as run:
  with pytest.raises(ValueError):run.stage('safe',key_for(1),tmp_path/'outside',lambda *_:None,[])
  with pytest.raises(ValueError):run.stage('../outside',key_for(1),run.revision_root,lambda *_:None,[])
