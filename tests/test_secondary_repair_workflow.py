from pathlib import Path
import pytest
from test_secondary_workbench import work,start
from motionbridge.secondary_cache import atomic_json,file_sha
from motionbridge.secondary_contact_constraints import contact_active_faces,prewarm_convergence,simulation_quality_gate,garment_edge_deformation,mesh_edges,FloorContactGraph
import numpy as np

def test_pinned_torso_is_excluded_without_floor_conflict():
 faces=np.array([[0,1,2],[2,3,4],[5,6,7]])
 pins=np.ones(8);pins[7]=.5
 assert contact_active_faces(pins,faces,np.zeros(8,bool)).tolist()==[False,False,True]

def test_contact_selection_adds_pinned_floor_region_without_protected_or_hard_faces():
 faces=np.array([[0,1,2],[2,3,4],[5,6,7],[8,9,10]])
 unsafe=np.zeros(11,bool);unsafe[[0,5,8]]=True;protected=np.zeros(11,bool);protected[5:8]=True
 assert contact_active_faces(np.ones(11),faces,unsafe,protected,[False,False,False,True],rings=2).tolist()==[True,True,False,False]

def completed(work,status='quality_rejected'):
 w,c,scene,pool=work;job=start(work).json();jid=job['id'];out=Path(job['output']);pair=out/'candidate';pair.mkdir(parents=True)
 for v in ('before','after'):(pair/(v+'.blend')).write_bytes(v.encode())
 scripts=w.root/'scripts';scripts.mkdir();(scripts/'render_secondary_workbench_preview.py').write_text('renderer',encoding='utf-8')
 atomic_json(w.root/'config.json',{'blender_path':'native-blender'})
 w._update(jid,status=status,physics_quality='rejected',protection_passed=True,result={'pair':str(pair),'case':str(out)})
 return jid,pair

@pytest.mark.parametrize('status',['completed','quality_rejected'])
def test_preview_refresh_never_runs_physics_even_without_managed_cache(work,monkeypatch,status):
 w,c,scene,pool=work;jid,pair=completed(work,status);before={v:file_sha(pair/(v+'.blend')) for v in ('before','after')};calls=[]
 def render(jid,command,managed):
  calls.append(command);assert managed is None and 'simulate' not in ' '.join(command)
  preview=Path(command[command.index('--output-dir')+1]);atomic_json(preview/'preview.json',{'passed':True,'frames':[1,2],
   'renderer_sha256':file_sha(w.root/'scripts/render_secondary_workbench_preview.py'),'source_sha256':before});return 0
 monkeypatch.setattr(w,'_call',render)
 assert c.post(f'/api/secondary/runs/{jid}/refresh-preview').status_code==202
 w.execute(jid,True);row=w.get(jid)
 assert len(calls)==1 and row['status']==status and row['physics_quality']=='rejected' and row['protection_passed'] is True
 assert before=={v:file_sha(pair/(v+'.blend')) for v in before}

def test_cancelled_preview_resume_does_not_turn_into_physics_job(work,monkeypatch):
 w,c,scene,pool=work;jid,pair=completed(work)
 c.post(f'/api/secondary/runs/{jid}/refresh-preview');c.post(f'/api/secondary/runs/{jid}/cancel')
 assert c.post(f'/api/secondary/runs/{jid}/resume').status_code==202
 calls=[]
 def fail(jid,command,managed):calls.append(command);assert managed is None;return 23
 monkeypatch.setattr(w,'_call',fail);w.execute(jid,True)
 assert len(calls)==1 and w.get(jid)['status']=='failed' and w.get(jid)['operation']=='preview_only'
 c.post(f'/api/secondary/runs/{jid}/resume');w.execute(jid,True)
 assert len(calls)==2 and all('render_secondary_workbench_preview.py' in ' '.join(cmd) for cmd in calls)
 assert w.get(jid)['physics_quality']=='rejected'

def test_invalid_preview_proof_is_not_published(work,monkeypatch):
 w,c,scene,pool=work;jid,pair=completed(work)
 def invalid(jid,command,managed):
  atomic_json(Path(command[command.index('--output-dir')+1])/'preview.json',{'passed':False});return 0
 monkeypatch.setattr(w,'_call',invalid);c.post(f'/api/secondary/runs/{jid}/refresh-preview');w.execute(jid,True)
 assert w.get(jid)['status']=='failed' and not w.get(jid).get('preview_dir')

def test_preview_requires_an_existing_physics_result(work):
 w,c,scene,pool=work;job=start(work).json();w._update(job['id'],status='completed')
 assert c.post(f"/api/secondary/runs/{job['id']}/refresh-preview").status_code==409
 assert len(pool.calls)==1

def test_warmup_must_be_stable_for_entire_trailing_interval():
 speeds=np.zeros((30,100));speeds[0,20]=.2
 assert prewarm_convergence(speeds)['passed'] is False
 assert prewarm_convergence(speeds[1:])['passed'] is True
 speeds[:,:]=.025
 assert prewarm_convergence(speeds)['passed'] is False

@pytest.mark.parametrize('speeds',[[],[[float('nan')]],[[float('inf')]],[[float('-1')]]])
def test_unmeasured_or_invalid_warmup_cannot_be_approved(speeds):
 with pytest.raises(ValueError):prewarm_convergence(speeds)

def test_geometric_nonworsening_does_not_approve_unsettled_cloth():
 surface={'accepted':True,'rejection_reasons':[]}
 assert not simulation_quality_gate(surface,{})['accepted']
 assert not simulation_quality_gate(surface,{'prewarm_convergence':{'passed':False}})['accepted']
 assert not simulation_quality_gate(surface,{'prewarm_convergence':{'passed':True}})['accepted']
 assert simulation_quality_gate(surface,{'prewarm_convergence':{'passed':True}},{'garment_deformation_gate':{'passed':True}})['accepted']
 assert surface=={'accepted':True,'rejection_reasons':[]}

def test_edge_deformation_preserves_rigid_panel_motion_but_rejects_flattening():
 p=np.array([[0,0,0],[1,0,0],[0,0,1]],float);e=mesh_edges([[0,1,2]])
 r=np.array([[0,0,1],[0,1,0],[-1,0,0]],float)
 assert garment_edge_deformation(p,p@r.T+[2,3,4],e)['passed']
 collapsed=p.copy();collapsed[:,2]=0
 assert not garment_edge_deformation(p,collapsed,e)['passed']

def test_small_contact_roundoff_is_not_deformation_but_large_stretch_is():
 p=np.array([[0,0,0],[.001,0,0],[0,.001,0]],float);e=mesh_edges([[0,1,2]])
 assert garment_edge_deformation(p,p*1.001,e)['passed']
 assert not garment_edge_deformation(p,p*2,e)['passed']

def test_clean_crossing_counts_and_settled_warmup_cannot_mask_excessive_deformation():
 gate=simulation_quality_gate({'accepted':True,'rejection_reasons':[]},{'prewarm_convergence':{'passed':True}},{'garment_deformation_gate':{'passed':False}})
 assert not gate['accepted'] and gate['rejection_reasons']==['garment_deformation_excessive_or_unverified']

def test_shape_constraints_repair_stretch_even_above_ground_without_changing_anchor():
 source=np.array([[0,0,1],[.01,0,1],[0,.01,1]],float);candidate=source.copy();candidate[1,0]=.1
 final,report=FloorContactGraph(source,[[0,1,2]],[True,False,False]).solve(candidate,0,iterations=128,reference_positions=source)
 assert np.array_equal(final[0],candidate[0]) and report['contact_vertices']==0
 assert garment_edge_deformation(source,final,mesh_edges([[0,1,2]]))['passed']

def test_valid_rigid_panel_shape_remains_bitwise_unchanged_with_explicit_reference():
 source=np.array([[0,0,1],[.1,0,1],[0,.1,1]],np.float32);candidate=source+[1,2,3]
 final,_=FloorContactGraph(source,[[0,1,2]]).solve(candidate,0,reference_positions=source)
 assert np.array_equal(final,candidate)
