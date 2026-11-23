"""Fully decode the final comparison AV without changing the deliverable."""
from pathlib import Path
import json
import subprocess
import imageio_ffmpeg

root = Path(__file__).resolve().parents[1]
ids = json.loads((root / 'data/release-v040/final_jobs.json').read_text())
folder = root / 'data/jobs' / ids['character'] / 'outputs'
video = folder / 'Nahida_Dance_Comparison.mp4'
result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i', str(video),
    '-map', '0:v:0', '-map', '0:a:0', '-f', 'null', '-'], capture_output=True, text=True,
    encoding='utf-8', errors='replace', timeout=120)
report = {'file': str(video), 'required_streams': ['video', 'audio'],
          'returncode': result.returncode, 'stderr': result.stderr,
          'passed': result.returncode == 0 and not result.stderr.strip()}
(folder / 'av_decode_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report))
assert report['passed']
