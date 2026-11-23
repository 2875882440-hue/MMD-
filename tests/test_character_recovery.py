import hashlib
import json
import pytest
from motionbridge import retarget


def checkpoint(tmp_path):
    source=tmp_path/'source.blend';source.write_bytes(b'original scene')
    motion=tmp_path/'motion.json';motion.write_text('{"frames":[{}]}')
    output=tmp_path/'outputs';output.mkdir()
    target=output/'Nahida_MotionBridge_Dance.blend';target.write_bytes(b'keep this project'*100)
    report={'output':str(target),'motion':str(motion),'original_project':str(source),
        'motion_sha256':hashlib.sha256(motion.read_bytes()).hexdigest(),
        'original_sha256_before':hashlib.sha256(source.read_bytes()).hexdigest(),
        'surface_clearance':{'enabled':True},'frames':1}
    (output/'retarget_report.json').write_text(json.dumps(report))
    (output/'saved_project_audit.json').write_text('{"passed":false}')
    (output/'character_audit.log').write_text('previous failure')
    (output/'character_preview.png').write_bytes(b'preview'*200)
    return {'scene_path':str(source),'surface_clearance':True},motion,output


def test_recovery_runs_only_saved_audit_and_preserves_failure_evidence(tmp_path,monkeypatch):
    profile,motion,output=checkpoint(tmp_path);calls=[];cancelled=lambda:False
    def audit(blender,script,args,log,cancel,timeout):
        calls.append((script,args,cancel));(output/'saved_project_audit.json').write_text('{"passed":true}')
    monkeypatch.setattr(retarget,'run_script',audit)
    result=retarget.audit_existing_character(tmp_path,'blender',profile,motion,output,cancelled,lambda *a:None)
    assert result['passed'] and len(calls)==1
    assert calls[0]==(tmp_path/'scripts/audit_retarget_review.py',[output],cancelled)
    history=list((output/'validation_history').iterdir());assert len(history)==1
    assert json.loads((history[0]/'saved_project_audit.json').read_text())=={'passed':False}
    assert (history[0]/'character_audit.log').read_text()=='previous failure'
    assert (output/'Nahida_MotionBridge_Dance.blend').read_bytes()==b'keep this project'*100


def test_failed_reaudit_is_never_promoted(tmp_path,monkeypatch):
    profile,motion,output=checkpoint(tmp_path)
    monkeypatch.setattr(retarget,'run_script',lambda *a,**kw:None)
    with pytest.raises(RuntimeError,match='验证仍未通过'):
        retarget.audit_existing_character(tmp_path,'blender',profile,motion,output,lambda:False,lambda *a:None)
