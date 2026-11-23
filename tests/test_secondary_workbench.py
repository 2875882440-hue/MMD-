import io,json,zipfile
from pathlib import Path
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from motionbridge.secondary_cache import atomic_json,file_sha
from motionbridge.secondary_workbench import Workbench,StartRequest,ExportRequest,router,package_entries,stream_zip,valid_preview

class Pool:
 def __init__(self):self.calls=[]
 def submit(self,*args):self.calls.append(args)

@pytest.fixture
def work(tmp_path):
 root=tmp_path/'project';root.mkdir();data=root/'data/physics-jobs';pool=Pool()
 model=root/'assets/rin.pmx';model.parent.mkdir();model.write_bytes(b'verified model')
 scene=root/'output/secondary_motion/source.blend';scene.parent.mkdir(parents=True);scene.write_bytes(b'original motion')
 atomic_json(root/'profiles/secondary_motion/rin.json',{'id':'rin','model':{'path':'assets/rin.pmx','sha256':file_sha(model)},'hair_dynamics':{'fixed':1},'surface_dynamics':{'fixed':2}})
 atomic_json(root/'profiles/secondary_sources.json',{'sources':[{'profile':'rin','label':'source','scene':str(scene)}]})
 manager=Workbench(root,data,pool);app=FastAPI();app.include_router(router(lambda:manager))
 with TestClient(app) as client:yield manager,client,scene,pool

def start(work,**override):
 w,c,_,_=work;return c.post('/api/secondary/runs',json={'source_id':w.sources()[0]['id'],'enabled':True,**override})

def test_default_off_does_not_create_task_or_write_source(work):
 w,c,scene,pool=work;before=file_sha(scene)
 assert c.get('/api/secondary/catalog').json()['default_enabled'] is False
 assert start(work,enabled=False).status_code==409
 assert not w.runs and not pool.calls and file_sha(scene)==before

@pytest.mark.parametrize('change',[{'source_id':'../../outside'},{'profile':'nahida'},{'mode':'unknown'},{'source_id':'missing'}])
def test_wrong_source_profile_or_mode_rejected(work,change):
 assert start(work,**change).status_code==409 and not work[0].runs

def test_strict_boolean_and_unknown_fields_rejected(work):
 assert start(work,enabled='true').status_code==422
 assert start(work,source_path='C:/secret.blend').status_code==422

def test_model_fingerprint_and_source_path_escapes_not_silently_accepted(work):
 w,c,scene,pool=work;(w.root/'assets/rin.pmx').write_bytes(b'different model')
 assert start(work).status_code==409 and not pool.calls
 atomic_json(w.root/'profiles/secondary_sources.json',{'sources':[{'profile':'rin','label':'escape','scene':'../outside.blend'}]})
 assert w.sources()==[]

def test_queued_cancel_resume_preserves_same_job_and_source(work):
 w,c,scene,pool=work;job=start(work).json();jid=job['id'];before=file_sha(scene)
 assert start(work).status_code==409
 assert c.post(f'/api/secondary/runs/{jid}/cancel').json()['status']=='cancelled'
 assert c.post(f'/api/secondary/runs/{jid}/resume').json()['status']=='queued'
 assert len(w.runs)==1 and len(pool.calls)==2 and file_sha(scene)==before

def test_changed_source_cannot_resume_old_run(work):
 w,c,scene,pool=work;job=start(work).json();c.post(f"/api/secondary/runs/{job['id']}/cancel")
 scene.write_bytes(b'new animation')
 assert c.post(f"/api/secondary/runs/{job['id']}/resume").status_code==409

def test_interrupted_recovery_is_lazy_and_does_not_start_child(work):
 w,c,scene,pool=work;job=start(work).json()
 recovered=Workbench(w.root,w.store.parent,Pool())
 assert recovered.get(job['id'])['status']=='interrupted' and not recovered.pool.calls

def test_quality_rejected_export_needs_explicit_format_only_ack(work):
 w,c,scene,pool=work;job=start(work).json();jid=job['id']
 w._update(jid,status='quality_rejected',physics_quality='rejected',result={'pair':str(scene.parent),'case':str(scene.parent)})
 assert c.post(f'/api/secondary/runs/{jid}/export',json={}).status_code==409
 exported=c.post(f'/api/secondary/runs/{jid}/export',json={'experimental':True,'formats':['fbx-character','skeleton','blender']})
 assert exported.status_code==202
 result=exported.json();assert result['physics_quality']=='rejected' and result['format_validated'] is False
 assert result['formats']==['blender','skeleton','fbx-character']
 assert w.get(jid)['physics_quality']=='rejected'

def test_preview_failure_resume_does_not_overwrite_partial_attempt(work):
 w,c,scene,pool=work;job=start(work).json();out=Path(job['output']);partial=out/'preview-old';partial.mkdir(parents=True)
 (partial/'partial.png').write_bytes(b'partial')
 w._update(job['id'],status='failed')
 assert c.post(f"/api/secondary/runs/{job['id']}/resume").status_code==202
 assert (partial/'partial.png').read_bytes()==b'partial'

def test_download_is_allowlisted_and_does_not_serve_unrelated_files(work):
 w,c,scene,pool=work;job=start(work).json();out=Path(job['output']);out.mkdir(parents=True)
 (out/'after.blend').write_bytes(b'new independent scene');(out/'private.key').write_bytes(b'secret')
 row=c.get(f"/api/secondary/runs/{job['id']}").json();assert len(row['files'])==1
 file=row['files'][0];assert c.get(file['url']).content==b'new independent scene'
 assert c.get(f"/api/secondary/runs/{job['id']}/files/private.key").status_code==404

def test_streamed_archive_roundtrip_and_package_escape_rejection(work):
 w,c,scene,pool=work;job=start(work).json();out=Path(job['output']);delivery=out/'delivery';delivery.mkdir(parents=True)
 payload=delivery/'scene.blend';payload.write_bytes(b'actual package bytes')
 atomic_json(delivery/'manifest.json',{'files':{'scene.blend':{'bytes':payload.stat().st_size,'sha256':file_sha(payload)}}})
 w._update(job['id'],format_validated=True,delivery=str(delivery),status='completed')
 response=c.get(f"/api/secondary/runs/{job['id']}/download");assert response.status_code==200
 with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
  assert archive.testzip() is None and archive.read('scene.blend')==payload.read_bytes()
 atomic_json(delivery/'manifest.json',{'files':{'../../outside':{'bytes':1}}})
 with pytest.raises(ValueError):package_entries(w.get(job['id']))

def test_completed_rin_character_is_discovered_without_changing_old_job(work):
 w,c,scene,pool=work;folder=w.store.parent/('a'*32);path=folder/'outputs/Rin_MotionBridge_Dance.blend';path.parent.mkdir(parents=True);path.write_bytes(b'new result')
 atomic_json(folder/'job.json',{'status':'completed','name':'new source'})
 before=file_sha(folder/'job.json');assert len(w.sources())==2 and file_sha(folder/'job.json')==before

def test_refresh_preview_reuses_managed_run_and_keeps_rejected_quality(work):
 w,c,scene,pool=work;job=start(work).json();jid=job['id'];out=Path(job['output'])
 atomic_json(out/'run.json',{'status':'completed'})
 w._update(jid,status='quality_rejected',physics_quality='rejected',result={'case':str(out),'pair':str(out)})
 response=c.post(f'/api/secondary/runs/{jid}/refresh-preview');assert response.status_code==202
 row=response.json();assert row['id']==jid and row['physics_quality']=='rejected'
 assert pool.calls[-1][-1] is True and '--resume' in w.command(row,True)

def test_download_rejects_same_size_changed_file(work):
 w,c,scene,pool=work;job=start(work).json();delivery=Path(job['output'])/'delivery';delivery.mkdir(parents=True)
 payload=delivery/'scene.blend';payload.write_bytes(b'original')
 atomic_json(delivery/'manifest.json',{'files':{'scene.blend':{'bytes':8,'sha256':file_sha(payload)}}})
 w._update(job['id'],format_validated=True,delivery=str(delivery),status='completed')
 payload.write_bytes(b'modified')
 assert c.get(f"/api/secondary/runs/{job['id']}/download").status_code==409

@pytest.mark.parametrize('change',[{'passed':False},{'renderer_sha256':'old-renderer'},{'source_sha256':{'before':'a','after':'edited-result'}}])
def test_partial_or_stale_preview_cannot_be_shown_as_current_result(tmp_path,change):
 source={'before':'a','after':'b'};proof={'passed':True,'renderer_sha256':'current','source_sha256':source}
 atomic_json(tmp_path/'preview.json',proof);assert valid_preview(tmp_path,'current',source)
 proof.update(change);atomic_json(tmp_path/'preview.json',proof);assert not valid_preview(tmp_path,'current',source)

def test_task_log_readback_is_scoped_and_available_on_failure(work):
 w,c,scene,pool=work;job=start(work).json();log=w.store/job['id']/'process.log';log.write_text('native failure details',encoding='utf-8')
 w._update(job['id'],status='failed',process_log=str(log))
 response=c.get(f"/api/secondary/runs/{job['id']}/log");assert response.status_code==200 and 'native failure details' in response.text

def test_task_log_cannot_read_arbitrary_external_path(work):
 w,c,scene,pool=work;job=start(work).json();w._update(job['id'],process_log=str(scene))
 assert c.get(f"/api/secondary/runs/{job['id']}/log").status_code==403

def test_selected_nonworsening_candidate_does_not_approve_complete_collision(work,monkeypatch):
 w,c,scene,pool=work;job=start(work,mode='surface',preview=False).json();out=Path(job['output']);pair=out/'candidate';pair.mkdir(parents=True)
 atomic_json(pair/'surface-audit.json',{'protection_passed':True,'collision_quality_passed':False})
 atomic_json(out/'latest.json',{'case':str(out),'quality':{'selected':str(pair),'last_experimental_candidate':str(pair)}})
 monkeypatch.setattr(w,'_call',lambda *_:0);w.execute(job['id'],False)
 actual=w.get(job['id']);assert actual['status']=='completed' and actual['selection_gate_passed'] is True
 assert actual['physics_quality']=='rejected' and actual['protection_passed'] is True
 assert c.post(f"/api/secondary/runs/{job['id']}/export",json={}).status_code==409
