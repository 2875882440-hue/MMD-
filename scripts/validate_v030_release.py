"""Reproducible local v0.3.0 evidence; does not alter existing input projects."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/release-v030'
BLENDER = r'D:\STEAM\steamapps\common\Blender\blender.exe'
JOBS = {'dance': '0b1e96ce61304e9b84d29cee9d025245',
        'tennis': 'f8585fecb46c42b5b171b5328c6d64ce',
        'character': '5e0bed0c3d8f412d8083e5d759d1b0a8'}

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def run_blender(script, folder, phase, log):
    with log.open('w', encoding='utf-8') as stream:
        subprocess.run([BLENDER, '--background', '--factory-startup', '--disable-autoexec',
                        '--threads', '4', '--python-exit-code', '23', '--python',
                        str(ROOT / 'scripts' / script), '--', str(folder), phase],
                       stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=600)

def main():
    baseline = json.loads((DEST / 'preexisting_hashes.json').read_text(encoding='utf-8-sig'))
    checks = {path: sha(path) == expected for path, expected in baseline.items()}
    report = {'preserved_files': checks, 'all_preserved': all(checks.values()), 'jobs': {}}
    for name, ident in JOBS.items():
        jobdir = ROOT / 'data/jobs' / ident
        job = json.loads((jobdir / 'job.json').read_text(encoding='utf-8'))
        (DEST / f'{name}_job.json').write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding='utf-8')
        report['jobs'][name] = {'id': ident, 'status': job['status']}
        assert job['status'] == 'completed', (name, job['status'])
    target = ROOT / 'data/jobs' / JOBS['character'] / 'outputs'
    report['download_matches'] = sha(ROOT / 'output/playwright/v030-nahida.blend') == sha(target / 'Nahida_MotionBridge_Dance.blend')
    (DEST / 'preservation_and_download.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    assert report['all_preserved'] and report['download_matches']
    if '--render' in sys.argv:
        for name, times in [('dance', [1, 4, 7, 10, 13, 16, 19, 23, 28]), ('tennis', [1, 4, 7])]:
            jobdir = ROOT / 'data/jobs' / JOBS[name]
            motion = json.loads((jobdir / 'outputs/motion.json').read_text(encoding='utf-8'))
            folder = DEST / f'{name}-review'
            folder.mkdir(exist_ok=True)
            assessment = {'outputs': str(jobdir / 'outputs'), 'sample_times': times,
                          'source_filename': str(jobdir / 'input/source.mp4'),
                          'source_duration': motion['num_frames'] / motion['fps']}
            (folder / 'assessment.json').write_text(json.dumps(assessment, indent=2), encoding='utf-8')
            for phase in ['probe', 'samples']:
                run_blender('render_dance_review.py', folder, phase, folder / f'{phase}.log')
            subprocess.run([sys.executable, str(ROOT / 'scripts/compose_dance_review.py'), str(folder),
                            str(jobdir / 'input/source.mp4'), '--sheets-only'], check=True)
            print(name, 'rendered', flush=True)
        run_blender('render_retarget_review.py', target, 'samples', DEST / 'character_samples.log')
        print('character rendered', flush=True)
    print('Preserved', len(checks), 'files; browser download hash matches.', flush=True)

if __name__ == '__main__':
    main()
