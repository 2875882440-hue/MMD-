"""Build timestamp-aligned evidence from actual Blender renders and input video."""
from pathlib import Path
import argparse
import json
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

parser = argparse.ArgumentParser()
parser.add_argument("folder", type=Path)
parser.add_argument("source", type=Path)
parser.add_argument("--sheets-only", action="store_true")
args = parser.parse_args()
folder = args.folder.resolve()
report = json.loads((folder / "assessment.json").read_text(encoding="utf-8"))
outputs = Path(report["outputs"])
font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 22)
small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 17)
source = cv2.VideoCapture(str(args.source))
overlay = cv2.VideoCapture(str(outputs / "overlay.mp4"))
motion = json.loads((outputs / "motion.json").read_text(encoding="utf-8"))
fps = motion["fps"]
count = motion["num_frames"]
source_fps = source.get(cv2.CAP_PROP_FPS)
if source_fps <= 0:
    raise RuntimeError("Cannot determine original video frame rate")


def frame_at(cap, frame):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, array = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot decode source frame {frame}")
    return Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB)).resize((360, 640))


def comparison(idx):
    frame = idx * 2
    canvas = Image.new("RGB", (1080, 736), "#101923")
    draw = ImageDraw.Draw(canvas)
    for x, label in [(0, "原视频"), (360, "识别叠加"), (720, "Blender 实际模型")]:
        draw.text((x + 16, 12), label, font=font, fill="#e7edf5")
    # Inputs can have different frame rates: align by timestamp, not frame index.
    canvas.paste(frame_at(source, round(frame / fps * source_fps)), (0, 48))
    canvas.paste(frame_at(overlay, frame), (360, 48))
    with Image.open(folder / "blender_frames" / f"{idx:05d}.png") as rendered:
        if rendered.size != (360, 640):
            raise RuntimeError("Unexpected Blender render dimensions")
        canvas.paste(rendered.convert("RGB"), (720, 48))
    draw.text((16, 699), f"{frame / fps:05.2f}s  |  同步预览 {fps/2:g} FPS · 工程 {fps:g} FPS · 原地骨骼 · 未手工修动作", font=small, fill="#c2cfdd")
    return canvas


for group in range(3):
    times = report["sample_times"][group * 3:group * 3 + 3]
    if not times:
        continue
    indices = [round(t * fps / 2) for t in times]
    if not all((folder / "blender_frames" / f"{i:05d}.png").is_file() for i in indices):
        continue
    sheet = Image.new("RGB", (1080, 736 * len(times)))
    for row, idx in enumerate(indices):
        sheet.paste(comparison(idx), (0, 736 * row))
    path = folder / f"comparison_sheet_{group + 1}.jpg"
    sheet.save(path, quality=94)
    print(f"SHEET {path}", flush=True)

if not args.sheets_only:
    expected = (count + 1) // 2
    missing = [i for i in range(expected) if not (folder / "blender_frames" / f"{i:05d}.png").is_file()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} render frames")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    silent = folder / "comparison_silent.mp4"
    log_path = folder / "comparison_encode.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen([ffmpeg, "-y", "-v", "warning", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1080x736",
            "-r", str(fps / 2), "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "4", "-preset", "fast", "-crf", "19",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(silent)], stdin=subprocess.PIPE, stderr=log)
        try:
            for idx in range(expected):
                process.stdin.write(np.asarray(comparison(idx)).tobytes())
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("Video encoding failed; see comparison_encode.log")
        except BaseException:
            process.kill()
            process.wait()
            raise
        destination = folder / "dance_comparison.mp4"
        subprocess.run([ffmpeg, "-y", "-v", "warning", "-i", str(silent), "-i", str(args.source), "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-t", str(report["source_duration"]), "-movflags", "+faststart", str(destination)],
            check=True, stdout=log, stderr=log)
    check = cv2.VideoCapture(str(destination))
    metadata = {"file": str(destination), "width": int(check.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(check.get(cv2.CAP_PROP_FRAME_HEIGHT)), "fps": check.get(cv2.CAP_PROP_FPS),
                "frames": int(check.get(cv2.CAP_PROP_FRAME_COUNT))}
    decoded = 0
    while True:
        ok, frame = check.read()
        if not ok:
            break
        decoded += 1
    check.release()
    metadata["decoded_frames"] = decoded
    metadata["passed"] = decoded == expected and metadata["fps"] == fps / 2
    (folder / "comparison_verification.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if not metadata["passed"]:
        raise RuntimeError(f"Preview verification failed: {metadata}")
    print(json.dumps(metadata), flush=True)
source.release()
overlay.release()
