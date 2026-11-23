from pathlib import Path
import io
import zipfile
import hashlib
import json

import pytest
from fastapi.testclient import TestClient
import motionbridge.app as server


def test_module_import_store_is_isolated_before_any_client_fixture():
    project=Path(__file__).resolve().parents[1]
    assert server.DATA.resolve().is_relative_to((project/'.test-runtime').resolve())
    assert server.DATA.resolve()!=(project/'data/jobs').resolve()
    assert server.jobs=={}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA", tmp_path / "jobs")
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "jobs", {})
    monkeypatch.setattr(server, "cancel_events", {})
    monkeypatch.setattr(server.pool, "submit", lambda *args, **kwargs: None)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "pose_landmarker_full.task").write_bytes(b"test-model")
    with TestClient(server.app) as api:
        yield api


def upload(client, **data):
    return client.post("/api/jobs", files={"file": ("movement.mp4", b"fake-video-bytes", "video/mp4")}, data=data)


def test_upload_is_persisted_and_processing_is_queued(client):
    response = upload(client, smoothing=.4, sample_fps=24, exports_3d="false")
    assert response.status_code == 202
    job = response.json()
    assert job["settings"]["exports_3d"] is False
    assert job["settings"]["refine_motion"] is True
    assert job["software_version"] == server.__version__
    assert (server.DATA / job["id"] / "input" / "source.mp4").read_bytes() == b"fake-video-bytes"
    assert (server.DATA / job["id"] / "job.json").is_file()


def test_invalid_parameters_do_not_create_jobs(client):
    assert upload(client, sample_fps=999).status_code == 422
    assert upload(client, smoothing="nan").status_code == 422
    assert server.jobs == {}


def test_experimental_engine_requires_assets_and_rejects_unknown_engine(client):
    assert upload(client, engine='imaginary').status_code == 422
    assert upload(client, engine='gvhmr_experimental').status_code == 409
    assert server.jobs == {}


def test_world_correction_controls_survive_upload_and_reprocess(client,monkeypatch):
    from motionbridge import gvhmr_pipeline
    monkeypatch.setattr(gvhmr_pipeline,'engine_status',lambda *a:{'installed':True})
    assert upload(client,world_arm_correction='invalid').status_code==422
    assert upload(client,world_body_collision='invalid').status_code==422
    assert upload(client,world_head_collision='invalid').status_code==422
    original=upload(client,engine='gvhmr_experimental',world_arm_correction='false',world_foot_contacts='true').json()
    assert original['settings']['keypoint_identity']=='temporal'
    assert original['settings']['world_body_collision'] is True
    assert original['settings']['world_head_collision'] is True
    server.update(original['id'],status='completed')
    assert client.post(f"/api/jobs/{original['id']}/reprocess",data={'world_body_collision':'invalid'}).status_code==422
    assert client.post(f"/api/jobs/{original['id']}/reprocess",data={'world_head_collision':'invalid'}).status_code==422
    new=client.post(f"/api/jobs/{original['id']}/reprocess",data={'world_arm_correction':'true','world_foot_contacts':'false','world_body_collision':'false','world_head_collision':'false'}).json()
    assert new['settings']['keypoint_identity']=='temporal_local'
    assert new['settings']['world_foot_contacts'] is False
    assert new['settings']['world_body_collision'] is False
    assert new['settings']['world_head_collision'] is False
    assert server.get_job(original['id'])['settings']['world_arm_correction'] is False


def test_engine_reprocess_uses_new_job_and_preserves_original(client, monkeypatch):
    from motionbridge import gvhmr_pipeline
    monkeypatch.setattr(gvhmr_pipeline, 'engine_status', lambda *a: {'installed': True})
    original = upload(client).json()
    server.update(original['id'], status='completed')
    response = client.post(f"/api/jobs/{original['id']}/reprocess", data={'engine': 'gvhmr_experimental'})
    assert response.status_code == 202
    job = response.json()
    assert job['settings']['engine'] == 'gvhmr_experimental'
    assert job['settings']['sample_fps'] == 30
    assert job['settings']['refine_motion'] is False
    assert server.jobs[original['id']]['settings']['engine'] == 'mediapipe'
    server.update(job['id'], status='completed')
    from motionbridge import retarget
    monkeypatch.setattr(retarget, 'character_profiles', lambda *a: [{'id':'nahida','ready':True}])
    (server.DATA / job['id'] / 'outputs/motion.json').write_text('{}')
    target = client.post(f"/api/jobs/{job['id']}/retarget")
    assert target.status_code == 202
    assert target.json()['settings']['root_mode'] == 'estimated_world_translation'
    assert target.json()['settings']['engine'] == 'gvhmr_experimental'


def test_refinement_can_be_disabled_for_comparison(client):
    response = upload(client, refine_motion="false")
    assert response.status_code == 202
    assert response.json()["settings"]["refine_motion"] is False
    assert client.get("/api/health").json()["features"]["motion_refinement"] is True


def test_reprocess_preserves_original_and_enables_current_correction(client):
    original = upload(client, refine_motion="false", smoothing=.4, sample_fps=30).json()
    old_id = original["id"]
    output = server.DATA / old_id / "outputs" / "motion.json"
    output.write_text("original animation", encoding="utf-8")
    server.update(old_id, status="completed")
    response = client.post(f"/api/jobs/{old_id}/reprocess")
    assert response.status_code == 202
    new = response.json()
    assert new["id"] != old_id and new["reprocessed_from"] == old_id
    assert new["software_version"] == server.__version__
    assert new["settings"]["refine_motion"] is True
    assert new["settings"]["sample_fps"] == 30
    assert new["settings"]["smoothing"] == .4
    assert (server.DATA / new["id"] / "input" / "source.mp4").read_bytes() == b"fake-video-bytes"
    assert output.read_text(encoding="utf-8") == "original animation"
    assert server.get_job(old_id)["settings"]["refine_motion"] is False
    assert server.get_job(old_id)["status"] == "completed"


def test_reprocess_rejects_running_demo_and_missing_original(client):
    job = upload(client).json()
    assert client.post(f"/api/jobs/{job['id']}/reprocess").status_code == 409
    server.update(job["id"], status="completed")
    (server.DATA / job["id"] / "input" / "source.mp4").unlink()
    assert client.post(f"/api/jobs/{job['id']}/reprocess").status_code == 409
    demo_id = client.post("/api/demo").json()["id"]
    server.update(demo_id, status="completed")
    assert client.post(f"/api/jobs/{demo_id}/reprocess").status_code == 409


def test_empty_and_unsupported_files(client):
    assert client.post("/api/jobs", files={"file": ("payload.exe", b"abc")}).status_code == 415
    response = client.post("/api/jobs", files={"file": ("empty.mp4", b"")})
    assert response.status_code == 400
    assert list(server.jobs.values())[0]["status"] == "failed"


def test_queue_has_bounded_capacity(client):
    for _ in range(server.MAX_QUEUE):
        assert upload(client).status_code == 202
    assert upload(client).status_code == 429


def test_cancel_before_worker_prevents_inference(client):
    job_id = upload(client).json()["id"]
    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 200
    server.run_job(job_id)
    assert server.get_job(job_id)["status"] == "cancelled"
    assert not (server.DATA / job_id / "outputs" / "motion.json").exists()


def test_remote_origins_and_hosts_are_rejected(client):
    assert client.post("/api/demo", headers={"Origin": "https://untrusted.example"}).status_code == 403
    assert client.get("/api/health", headers={"Host": "untrusted.example"}).status_code == 403
    assert client.get("/api/health", headers={"Host": "127.0.0.1:8765"}).status_code == 200


def test_file_whitelist_and_video_ranges(client):
    job_id = upload(client).json()["id"]
    output = server.DATA / job_id / "outputs"
    (output / "pose.mp4").write_bytes(b"0123456789")
    (output / "secret.txt").write_text("private")
    assert client.get(f"/api/jobs/{job_id}/files/secret.txt").status_code == 404
    response = client.get(f"/api/jobs/{job_id}/files/pose.mp4", headers={"Range": "bytes=2-4"})
    assert response.status_code == 206
    assert response.content == b"234"


def test_zip_excludes_source_and_includes_outputs(client):
    job_id = upload(client).json()["id"]
    output = server.DATA / job_id / "outputs"
    (output / "source.mp4").write_bytes(b"private original")
    (output / "motion.bvh").write_text("HIERARCHY")
    server.update(job_id, status="completed")
    response = client.get(f"/api/jobs/{job_id}/download")
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert "motion.bvh" in archive.namelist()
        assert "source.mp4" not in archive.namelist()


def test_demo_is_explicitly_marked(client):
    response = client.post("/api/demo")
    assert response.status_code == 202
    assert response.json()["is_demo"] is True


def test_missing_motion_returns_helpful_error(client):
    job_id = upload(client).json()["id"]
    assert client.get(f"/api/jobs/{job_id}/motion").status_code == 404
    assert client.get("/api/jobs/nonexistent").status_code == 404


def test_stability_option_and_reprocessing_defaults(client):
    assert upload(client, stabilize_rotations='invalid').status_code == 422
    old = upload(client, stabilize_rotations='false').json()
    assert not old['settings']['stabilize_rotations']
    server.update(old['id'], status='completed')
    new = client.post(f"/api/jobs/{old['id']}/reprocess").json()
    assert new['settings']['stabilize_rotations']


def configured_character(client, monkeypatch):
    from motionbridge import retarget
    monkeypatch.setattr(server, 'find_blender', lambda: 'fake-blender')
    monkeypatch.setattr(retarget, 'character_profiles', lambda *args: [{'id':'nahida', 'label':'Nahida',
        'ready':True, 'reason':'', 'scene_path':'source.blend', 'weights_scene_path':'weights.blend'}])
    original = upload(client).json()
    output = server.DATA / original['id'] / 'outputs'
    (output / 'motion.json').write_text('{"source":"unchanged"}', encoding='utf-8')
    (output / 'motion.fbx').write_bytes(b'source-animation')
    server.update(original['id'], status='completed')
    return original, output


def test_character_uses_independent_job_and_rejects_arbitrary_profiles(client, monkeypatch):
    original, output = configured_character(client, monkeypatch)
    assert client.post(f"/api/jobs/{original['id']}/retarget",data={'surface_clearance':'invalid'}).status_code==422
    assert client.post(f"/api/jobs/{original['id']}/retarget", data={'character':'../../other'}).status_code == 422
    assert len(server.jobs) == 1
    response = client.post(f"/api/jobs/{original['id']}/retarget")
    assert response.status_code == 202
    child = response.json()
    assert child['settings']['surface_clearance'] is True
    off=client.post(f"/api/jobs/{original['id']}/retarget",data={'surface_clearance':'false'}).json()
    assert off['settings']['surface_clearance'] is False
    assert child['source_job_id'] == original['id'] and child['kind'] == 'retarget'
    assert child['id'] != original['id']
    assert (output / 'motion.json').read_text() == '{"source":"unchanged"}'
    assert server.get_job(original['id'])['status'] == 'completed'
    assert client.post(f"/api/jobs/{child['id']}/reprocess").status_code == 409
    assert client.post(f"/api/jobs/{child['id']}/export-blender").status_code == 409


def test_enhancement_option_and_model_readiness(client):
    assert upload(client, enhance_motion='invalid').status_code == 422
    assert not server.jobs
    assert upload(client, enhance_motion='false').json()['settings']['enhance_motion'] is False
    original=upload(client).json()
    assert original['settings']['enhance_motion'] is True
    server.update(original['id'],status='completed')
    assert client.post(f"/api/jobs/{original['id']}/reprocess").json()['settings']['enhance_motion'] is True
    assert client.get('/api/health').json()['features']['hand_assistance'] is False
    (server.ROOT/'models/hand_landmarker.task').write_bytes(b'model')
    assert client.get('/api/health').json()['features']['hand_assistance'] is True


def test_character_cancel_before_worker_does_not_copy_or_launch_blender(client, monkeypatch):
    original, _ = configured_character(client, monkeypatch)
    child = client.post(f"/api/jobs/{original['id']}/retarget").json()
    client.post(f"/api/jobs/{child['id']}/cancel")
    server.run_character_job(child['id'], original['id'], {}, 'not-used')
    assert server.get_job(child['id'])['status'] == 'cancelled'
    assert not (server.DATA / child['id'] / 'input').exists()


def test_character_failure_preserves_source_and_reports_error(client, monkeypatch):
    from motionbridge import retarget
    original, output = configured_character(client, monkeypatch)
    def fail(*args):
        raise RuntimeError('rig mismatch')
    monkeypatch.setattr(retarget, 'export_character', fail)
    child = client.post(f"/api/jobs/{original['id']}/retarget").json()
    server.run_character_job(child['id'], original['id'], {}, 'not-used')
    assert server.get_job(child['id'])['status'] == 'failed'
    assert 'rig mismatch' in server.get_job(child['id'])['error']
    assert (output / 'motion.fbx').read_bytes() == b'source-animation'


def saved_character_checkpoint(client,monkeypatch):
    from motionbridge import retarget
    original,_=configured_character(client,monkeypatch)
    source=server.ROOT/'source.blend';source.write_bytes(b'original scene')
    monkeypatch.setattr(retarget,'character_profiles',lambda *a:[{'id':'nahida','ready':True,'scene_path':str(source),'weights_scene_path':'weights.blend'}])
    child=client.post(f"/api/jobs/{original['id']}/retarget").json()
    folder=server.DATA/child['id'];(folder/'input').mkdir()
    motion=folder/'input/motion.json';motion.write_text('{"frames":[{}]}')
    output=folder/'outputs';target=output/'Nahida_MotionBridge_Dance.blend';target.write_bytes(b'unchanged blend'*100)
    report={'output':str(target),'motion':str(motion),'original_project':str(source),
        'motion_sha256':hashlib.sha256(motion.read_bytes()).hexdigest(),
        'original_sha256_before':hashlib.sha256(source.read_bytes()).hexdigest(),
        'surface_clearance':{'enabled':True},'frames':1}
    (output/'retarget_report.json').write_text(json.dumps(report))
    server.update(child['id'],status='failed',error='old audit failed')
    return child,folder,report


def test_retry_validation_reuses_job_and_checkpoint_without_rebake(client,monkeypatch):
    child,folder,_=saved_character_checkpoint(client,monkeypatch);calls=[]
    monkeypatch.setattr(server.pool,'submit',lambda *a,**kw:calls.append((a,kw)))
    response=client.post(f"/api/jobs/{child['id']}/retarget/retry-validation")
    assert response.status_code==202
    new=response.json();assert new['id']==child['id'] and new['status']=='queued'
    assert new['validation_history'][0]['previous_error']=='old audit failed'
    assert len(calls)==1 and calls[0][1]=={'validate_only':True}
    assert (folder/'outputs/Nahida_MotionBridge_Dance.blend').read_bytes()==b'unchanged blend'*100
    assert client.post(f"/api/jobs/{child['id']}/retarget/retry-validation").status_code==409


def test_retry_rejects_changed_motion_and_missing_source_record(client,monkeypatch):
    child,folder,_=saved_character_checkpoint(client,monkeypatch)
    (folder/'input/motion.json').write_text('{"frames":[{},{}]}')
    assert client.post(f"/api/jobs/{child['id']}/retarget/retry-validation").status_code==409
    assert server.get_job(child['id'])['status']=='failed'
    del server.jobs[child['source_job_id']]
    assert client.post(f"/api/jobs/{child['id']}/retarget/retry-validation").status_code==409


def test_retry_rejects_other_project_and_zip_keeps_geometry_reference(client,monkeypatch):
    child,folder,report=saved_character_checkpoint(client,monkeypatch)
    report['output']=str(folder/'outputs/other.blend')
    (folder/'outputs/retarget_report.json').write_text(json.dumps(report))
    assert client.post(f"/api/jobs/{child['id']}/retarget/retry-validation").status_code==409
    (folder/'outputs/surface_solver_reference.npz').write_bytes(b'geometry reference')
    server.update(child['id'],status='completed')
    response=client.get(f"/api/jobs/{child['id']}/download")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert archive.read('surface_solver_reference.npz')==b'geometry reference'
