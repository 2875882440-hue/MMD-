"""Resume verified PNGs in bounded CPU Blender processes, then verify the full AV output."""
from pathlib import Path
import json
import subprocess
import sys
import argparse
from PIL import Image
import imageio_ffmpeg

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from motionbridge.retarget import run_script

parser=argparse.ArgumentParser()
parser.add_argument('--release-dir',type=Path,default=root/'data/release-v050')
parser.add_argument('--frame-step',type=int,choices=[1,2],default=2)
args=parser.parse_args()
release = args.release_dir.resolve()
ids = json.loads((release / 'final_jobs.json').read_text(encoding='utf-8'))
target = root / 'data/jobs' / ids['character'] / 'outputs'
motion = json.loads((root / 'data/jobs' / ids['dance'] / 'outputs/motion.json').read_text(encoding='utf-8'))
frames = list(range(1, motion['num_frames'] + 1, args.frame_step))
reports = []
reused = 0
for start in range(0, len(frames), 80):
    end = min(start + 80, len(frames))
    skip = []
    for frame in frames[start:end]:
        path = target / 'preview_frames' / f'{frame:04d}.png'
        try:
            with Image.open(path) as im:
                im.load()
                if im.size == (360, 480):
                    skip.append(frame)
        except (OSError, ValueError):
            pass
    reused += len(skip)
    run_script(r'D:\STEAM\steamapps\common\Blender\blender.exe', root / 'scripts/render_retarget_review.py',
               [target, 'preview', '--start-index', start, '--end-index', end,
                '--skip-frames', ','.join(map(str, skip)), '--frame-step',args.frame_step],
               release / f'preview-resume-{start}-{end}.log', lambda: False, timeout=300)
    reports.append(json.loads((target / f'preview_render_report_{start}_{end}.json').read_text(encoding='utf-8')))
    print(f'BATCH_COMPLETE {end}/{len(frames)} reused={len(skip)}', flush=True)
report = {'phase': 'preview', 'frames': [f for r in reports for f in r['frames']],
          'saved_blend_reopened': True, 'device': 'CPU', 'threads': 4,
          'elapsed_s': sum(r['elapsed_s'] for r in reports), 'reused_verified_pngs': reused,
          'batch_size': 80, 'frame_step':args.frame_step}
assert [f['frame'] for f in report['frames']] == frames
(target / 'preview_render_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
subprocess.run([sys.executable, str(root / 'scripts/compose_retarget_review.py'), str(target),
                '--source', str(root / 'data/jobs' / ids['dance'] / 'input/source.mp4'),'--frame-step',str(args.frame_step)], check=True)
r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i', str(target / 'Nahida_Dance_Comparison.mp4'),
                    '-map', '0:v:0', '-map', '0:a:0', '-f', 'null', '-'], capture_output=True, text=True, timeout=120)
check = {'passed': r.returncode == 0 and not r.stderr.strip(), 'returncode': r.returncode, 'stderr': r.stderr}
(target / 'av_decode_verification.json').write_text(json.dumps(check), encoding='utf-8')
assert check['passed'], check
print('VISUAL_AND_AV_COMPLETE', flush=True)
