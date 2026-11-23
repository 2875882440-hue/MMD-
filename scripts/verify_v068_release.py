"""Read-only release evidence: saved geometry, unchanged animation and full video."""
from pathlib import Path
import json, hashlib, urllib.request
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'research/v068'
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
records = read(DEST / 'character-jobs.json')
assert len(records) == 2 and all(r['status'] == 'completed' for r in records)
scope = read(DEST / 'code-scope.json')
for job, files in scope['source_artifacts'].items():
    for name, digest in files.items():
        assert hashlib.sha256((ROOT / 'data/jobs' / job / 'outputs' / name).read_bytes()).hexdigest() == digest
results = []
unchanged_keys = ['original_sha256_before', 'original_sha256_after', 'motion_sha256', 'frames', 'fps',
                  'world_root_transfer', 'target_ground_fit', 'target_body_collision', 'target_head_collision',
                  'target_accessory_collision', 'target_mesh_floor_collision', 'expected_root_positions',
                  'target_endpoint_ik', 'world_contact_ik']
for record in records:
    folder = 'nahida' if record['name'] == 'dance' else 'tennis'
    old = read(ROOT / 'research/v067' / folder / 'retarget_report.json')
    new = read(DEST / folder / 'retarget_report.json')
    audit = read(DEST / folder / 'saved_project_audit.json')
    rig = read(DEST / ('rig-unchanged.json' if folder == 'nahida' else 'tennis-rig-unchanged.json'))
    assert audit['passed'] and audit['surface_clearance']['passed'] and rig['passed']
    assert new['original_unchanged'] and new['original_sha256_before'] == new['original_sha256_after']
    unchanged = {key: old[key] == new[key] for key in unchanged_keys}
    assert all(unchanged.values()), unchanged
    surface = audit['surface_clearance']['objects'][0]
    assert surface['temporal_consistent'] and surface['displacement_consistent']
    result = {'name': record['name'], 'job_id': record['job_id'], 'rig_unchanged': rig,
              'prior_constraints_unchanged': unchanged, 'saved_surface': surface,
              'coherent_contact': new['surface_clearance']['objects'][0]['coherent_contact']}
    results.append(result)
old_render = read(ROOT / 'research/v067/nahida/preview_render_report.json')
new_render = read(DEST / 'nahida/preview_render_report.json')
assert old_render['reference_camera'] == new_render['reference_camera']
assert old_render['frames'] == new_render['frames'], 'Rendered source joints changed'
video = read(DEST / 'nahida/comparison_v068_verification.json')
assert video['frames'] == 901 and video['same_camera'] and video['full_av_decode_passed']
tennis_review = read(DEST / 'tennis/samples_render_report.json')
assert tennis_review['diagnostic_stage'] and not tennis_review['scene_saved']
assert len(tennis_review['frames']) == 4
with urllib.request.urlopen('http://127.0.0.1:8765/api/health', timeout=5) as response:
    health = json.load(response)
assert health['version'] == '0.6.8'
manifest = {'version': '0.6.8', 'characters': results, 'code_scope': scope,
            'video': video, 'full_render_joints_unchanged': True,
            'tennis_review_scene_saved': False, 'running_version': health['version'],
            'note': 'Geometry/export evidence, not motion accuracy or certification of all triangle contacts.'}
(DEST / 'release-verification.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print('V068_RELEASE_VERIFIED', flush=True)
