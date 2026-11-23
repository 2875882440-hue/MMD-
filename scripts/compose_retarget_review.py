"""Compare the uploaded video against actual rendered target character frames."""
import argparse
from pathlib import Path
import json
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

parser = argparse.ArgumentParser()
parser.add_argument('folder', type=Path)
parser.add_argument('--source', type=Path, help='Explicit original video for isolated character jobs')
parser.add_argument('--sheets-only', action='store_true')
parser.add_argument('--frame-step', type=int, choices=[1,2], default=2)
parser.add_argument('--sample-directory', choices=['samples','preview_frames'], default='samples')
args = parser.parse_args()
folder = args.folder.resolve()
report = json.loads((folder / 'retarget_report.json').read_text(encoding='utf-8'))
motion = json.loads(Path(report['motion']).read_text(encoding='utf-8'))
source_path = args.source or (Path(report['motion']).parent.parent / 'input' / 'source.mp4')
if not source_path.is_file():
    raise RuntimeError('Source video not found; pass --source with the original input video')
source = cv2.VideoCapture(str(source_path))
source_fps = source.get(cv2.CAP_PROP_FPS)
if source_fps <= 0:
    raise RuntimeError('Cannot determine source video frame rate')
fps = motion['fps']
font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 19)
small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 15)


def comparison(frame, directory):
    source.set(cv2.CAP_PROP_POS_FRAMES, round((frame - 1) / fps * source_fps))
    ok, data = source.read()
    if not ok:
        raise RuntimeError(f'Cannot decode source at frame {frame}')
    original = Image.fromarray(cv2.cvtColor(data, cv2.COLOR_BGR2RGB)).resize((270, 480))
    canvas = Image.new('RGB', (630, 548), '#101923')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 9), '上传的原视频', font=font, fill='white')
    draw.text((282, 9), '纳西妲 · 动作替换实测', font=font, fill='white')
    canvas.paste(original, (0, 38))
    with Image.open(folder / directory / f'{frame:04d}.png') as rendered:
        canvas.paste(rendered.convert('RGB').resize((360, 480)), (270, 38))
    note = '实验世界动作 · 仍有肢体跳变和离地漂移' if motion.get('root_motion') else '身体动捕 · 无精细手指 · 接地为估算'
    draw.text((12, 526), f'{(frame-1)/fps:05.2f}s  |  {note}', font=small, fill='#bdcbdc')
    return canvas


for group in range(3):
    sheet = Image.new('RGB', (630, 548 * 3))
    for row, t in enumerate([1, 4, 7, 10, 13, 16, 19, 23, 28][group*3:group*3+3]):
        sheet.paste(comparison(round(t*fps) + 1, args.sample_directory), (0, row*548))
    sheet.save(folder / f'comparison_sheet_{group+1}.jpg', quality=94)
if not args.sheets_only:
    frames = list(range(1, motion['num_frames'] + 1, args.frame_step))
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    silent = folder / 'comparison_silent.mp4'
    with (folder / 'encode.log').open('wb') as log:
        proc = subprocess.Popen([ffmpeg, '-y', '-v', 'warning', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', '630x548', '-r', str(fps/args.frame_step), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
            '-threads', '4', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            str(silent)], stdin=subprocess.PIPE, stderr=log)
        try:
            for frame in frames:
                proc.stdin.write(np.asarray(comparison(frame, 'preview_frames')).tobytes())
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError('Encoding failed')
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        destination = folder / 'Nahida_Dance_Comparison.mp4'
        subprocess.run([ffmpeg, '-y', '-v', 'warning', '-i', str(silent), '-i', str(source_path),
            '-map', '0:v:0', '-map', '1:a:0?', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '160k',
            '-t', str(motion['num_frames']/fps), '-movflags', '+faststart', str(destination)], check=True, stdout=log, stderr=log)
    cap = cv2.VideoCapture(str(destination))
    check = {'file': str(destination), 'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 'fps': cap.get(cv2.CAP_PROP_FPS),
        'expected_frames': len(frames), 'decoded_frames': 0}
    means = []
    while True:
        ok, data = cap.read()
        if not ok:
            break
        check['decoded_frames'] += 1
        means.append(float(data[:, 270:].mean()))
    cap.release()
    check['minimum_target_half_mean'] = min(means)
    subprocess.run([ffmpeg,'-v','error','-xerror','-i',str(destination),'-f','null','-'],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=120)
    check['full_audio_video_decode_passed'] = True
    check['passed'] = check['decoded_frames'] == len(frames) and check['fps'] == fps/args.frame_step and min(means) > 10
    (folder / 'preview_verification.json').write_text(json.dumps(check, indent=2), encoding='utf-8')
    if not check['passed']:
        raise RuntimeError(check)
    print(json.dumps(check), flush=True)
source.release()
