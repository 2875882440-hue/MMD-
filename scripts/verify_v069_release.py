"""Read-only source, real export, saved geometry and video release evidence."""
from pathlib import Path
import hashlib,json,zipfile,urllib.request,subprocess
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'research/v069'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
records=read(DEST/'character-jobs.json')
assert len(records)==2 and all(r['status']=='completed' and r['version']=='0.6.9' for r in records)
scope=read(ROOT/'research/v068/code-scope.json')
for job,files in scope['source_artifacts'].items():
    for name,digest in files.items():assert sha(ROOT/'data/jobs'/job/'outputs'/name)==digest
changed={'motionbridge/__init__.py','motionbridge/blender_surfaces.py','motionbridge/retarget.py','motionbridge/app.py'}
new={'motionbridge/triangle_contacts.py','motionbridge/blender_triangle_contacts.py'}
unchanged=[]
with zipfile.ZipFile(DEST/'v068-source.zip') as archive:
    entries=set(archive.namelist())
    for path in (ROOT/'motionbridge').glob('*.py'):
        name=path.relative_to(ROOT).as_posix()
        if name in changed|new:continue
        assert name in entries and hashlib.sha256(archive.read(name)).hexdigest()==sha(path),name
        unchanged.append(name)
    assert archive.read('scripts/blender_export.py')==(ROOT/'scripts/blender_export.py').read_bytes()
    old=archive.read('scripts/retarget_nahida.py').decode('utf-8').replace('\r\n','\n')
    current=(ROOT/'scripts/retarget_nahida.py').read_text(encoding='utf-8')
    expected=old.replace('Embedded bounded hair/garment corrective shapes against body capsules; no cloth dynamics or self-collision.',
        'Embedded bounded hair/garment shapes against body capsules and actual triangles/layer order; residuals and fallback reported. No cloth dynamics, same-component self-collision or continuous collision guarantee.')
    expected=expected.replace('bake_surface_clearance(scene,rig,garment_objects)',"bake_surface_clearance(scene,rig,garment_objects,reference_path=output/'surface_solver_reference.npz')")
    assert expected==current
unchanged_keys=['original_sha256_before','original_sha256_after','motion_sha256','frames','fps','world_root_transfer',
    'target_ground_fit','target_body_collision','target_head_collision','target_accessory_collision',
    'target_mesh_floor_collision','expected_root_positions','target_endpoint_ik','world_contact_ik']
results=[]
for record in records:
    folder='nahida' if record['name']=='dance' else 'tennis'
    old=read(ROOT/'research/v068'/folder/'retarget_report.json');report=read(DEST/folder/'retarget_report.json')
    audit=read(DEST/folder/'saved_project_audit.json')
    rig=read(DEST/('rig-unchanged.json' if folder=='nahida' else 'tennis-rig-unchanged.json'))
    assert audit['passed'] and audit['surface_clearance']['passed'] and rig['passed']
    same={k:old[k]==report[k] for k in unchanged_keys};assert all(same.values()),same
    assert all(o['triangle_consistent'] and o['triangle_contacts']['saved_geometry_consistent'] for o in audit['surface_clearance']['objects'])
    for obj in audit['surface_clearance']['objects']:
        reference=obj['triangle_contacts']['geometry_reference']
        assert reference['passed'] and reference['max_error_m']<=reference['tolerance_m']==1e-5
        assert reference['point_frames']==report['frames']*next(x['vertices'] for x in report['surface_clearance']['objects'] if x['name']==obj['name'])
        if folder=='tennis':
            assert reference['mode']=='pre_embedding_solver_vertices' and reference['original_solver_track_available']
            snapshot=DEST/folder/'surface_solver_reference.npz'
            assert sha(snapshot)==report['surface_clearance']['solver_reference']['sha256']
        else:
            assert reference['mode']=='legacy_saved_absolute_key_lbs' and not reference['original_solver_track_available']
    original_source=ROOT/'data/jobs'/record['job_id']/'outputs'
    for filename,digest in record['downloads'].items():
        assert sha(original_source/filename)==digest and sha(DEST/folder/filename)==digest
    fbx=read(DEST/('source-dance' if folder=='nahida' else 'source-tennis')/'blender_verification.json')
    assert fbx['passed'] and fbx['motionbridge_version']=='0.6.9' and fbx['foot_floor_collision']['passed']
    assert fbx['frames']==report['frames'] and fbx['fbx_roundtrip']['animation_present']
    results.append({'name':record['name'],'job_id':record['job_id'],'rig':rig,'prior_constraints_unchanged':same,
                    'surface':report['surface_clearance'],'saved_audit':audit['surface_clearance'],'source_fbx':fbx})
old_render=read(ROOT/'research/v068/nahida/preview_render_report.json')
new_render=read(DEST/'nahida/preview_render_report.json')
assert old_render['reference_camera']==new_render['reference_camera'] and old_render['frames']==new_render['frames']
video=read(DEST/'nahida/comparison_v069_verification.json')
assert video['frames']==901 and video['same_camera'] and video['full_av_decode_passed']
tennis=read(DEST/'tennis/samples_render_report.json');assert tennis['diagnostic_stage'] and not tennis['scene_saved']
with urllib.request.urlopen('http://127.0.0.1:8765/api/health',timeout=5) as response:assert json.load(response)['version']=='0.6.9'
assert read(DEST/'solver-blender-tests.json')['passed']
assert read(DEST/'reference-pipeline/verification.json')['passed']
assert read(DEST/'download-recovery-verification.json')['passed']
assert '174 passed' in (DEST/'tests-final.log').read_text(encoding='utf-8-sig')
subprocess.run(['node','--check','motionbridge/static/app.js'],cwd=ROOT,check=True)
visual_path=DEST/'visual-review.json'
visual=read(visual_path) if visual_path.exists() else {}
result={'version':'0.6.9','characters':results,'source_solver_files_unchanged':unchanged,
        'source_artifacts':scope['source_artifacts'],'video':video,'rendered_joints_unchanged':True,
        'python_tests_passed':174,'node_syntax_passed':True,'visual_review_required':not visual.get('completed',False),
        'visual_review_completed':visual.get('completed',False),
        'note':'Real export and saved geometry evidence, not capture accuracy, all-pixel visual improvement or collision-free certification.'}
(DEST/'release-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('V069_RELEASE_TECHNICAL_VERIFIED',flush=True)
