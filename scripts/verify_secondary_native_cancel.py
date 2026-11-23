"""Real Blender cancellation/resume fixture; original scene remains read-only."""
from pathlib import Path
import json,subprocess,sys,threading,time
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT))
from motionbridge.secondary_cache import file_sha,key_for,atomic_json,ArtifactCache
from motionbridge.secondary_tasks import RunSession,Cancelled
from motionbridge.secondary_pipeline import commands
root=ROOT/'output/secondary_motion/dev5/native-cancel';source=ROOT/'output/secondary_motion/dev3/rin-dance-contact/comparison/after.blend'
if root.exists():raise ValueError('Do not repeat an existing validation fixture')
blender=Path(json.loads((ROOT/'config.json').read_text(encoding='utf-8'))['blender_path']);digest=file_sha(source);cache=ArtifactCache(ROOT/'data/secondary-cache');cancel_sent=[]
def request():
 deadline=time.monotonic()+20
 while time.monotonic()<deadline:
  if list(root.rglob('native-started.json')):
   subprocess.run([sys.executable,'-B',ROOT/'scripts/secondary_run_control.py','--run-dir',root,'--cancel'],check=True,stdout=subprocess.DEVNULL);cancel_sent.append(True);return
  time.sleep(.05)
worker=threading.Thread(target=request);worker.start();captured=[]
def completed(path,_):path.mkdir(parents=True);atomic_json(path/'input.json',{'source_sha256':digest});captured.append(path)
try:
 with RunSession(root,{'source_sha256':digest,'fixture':'native_cancel'},cache) as run:
  ready=run.stage('input-probe',key_for({'fixture':'native_cancel_input','sha':digest}),run.revision_root,completed,['input.json'])
  def native(path,execute):execute(commands(ROOT,blender,'native_secondary_cancel_probe.py',['--scene',source,'--output-dir',path]))
  run.stage('native-probe',key_for({'fixture':'native_cancel_read_loop','sha':digest}),run.revision_root,native,['complete.json'])
 raise AssertionError('Cancellation did not interrupt the Blender process')
except Cancelled:pass
finally:worker.join(timeout=25)
state=json.loads((root/'run.json').read_text(encoding='utf-8'));assert state['status']=='cancelled' and cancel_sent and len(captured)==1
partials=list(root.rglob('native-started.json'));assert len(partials)==1 and not (partials[0].parent/'complete.json').exists()
assert cache.verify('native-probe',key_for({'fixture':'native_cancel_read_loop','sha':digest})) is None
with RunSession(root,{'source_sha256':digest,'fixture':'native_cancel'},cache,resume=True) as run:
 reused=run.stage('input-probe',key_for({'fixture':'native_cancel_input','sha':digest}),run.revision_root,completed,['input.json']);assert reused==ready
 run.stage('native-probe',key_for({'fixture':'native_cancel_read_loop','sha':digest}),run.revision_root,native,['complete.json'])
assert len(captured)==1 and file_sha(source)==digest
atomic_json(ROOT/'validation/secondary-motion-dev5/native-cancel-verification.json',{'passed':True,'real_blender_child_cancelled':True,'partial_retained':True,'partial_not_cached':True,'completed_stage_reused_after_resume':True,'same_incomplete_stage_restarted_in_new_attempt':True,'source_file_unchanged':True,'fixture':'read-only scene/frame loop; not OS crash recovery or a long physical solve'})
print('NATIVE_CANCEL_RESUME_PASSED',flush=True)
