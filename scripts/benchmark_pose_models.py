"""Read-only model comparison on sampled frames; never edits a captured motion."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from motionbridge.pipeline import _body18, _draw_body18

parser = argparse.ArgumentParser()
parser.add_argument("video", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--models", nargs="+", default=["full", "heavy"])
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(str(args.video))
times = [1, 4, 7, 10, 13, 16, 19, 23, 28]
images = []
for time in times:
    cap.set(cv2.CAP_PROP_POS_MSEC, time * 1000)
    ok, frame = cap.read()
    if not ok:
        break
    images.append(frame)
cap.release()
report = {}
for variant in args.models:
    path = ROOT / "models" / f"pose_landmarker_{variant}.task"
    if not path.exists():
        continue
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE)
    report[variant] = []
    sheet = Image.new("RGB", (720, 3 * 452), "#131920")
    draw = ImageDraw.Draw(sheet)
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as model:
        for index, frame in enumerate(images):
            result = model.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            if result.pose_landmarks:
                points = np.array([[p.x, p.y, p.z] for p in result.pose_landmarks[0]])
                confidence = np.array([min(p.visibility, p.presence) for p in result.pose_landmarks[0]])
                body, scores = _body18(points, confidence)
                overlay = _draw_body18(frame, body, scores)
                report[variant].append({"time": times[index], "image": points.tolist(), "confidence": confidence.tolist()})
            else:
                overlay = frame
                report[variant].append({"time": times[index], "missing": True})
            x, y = index % 3 * 240, index // 3 * 452
            sheet.paste(Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)).resize((240, 424)), (x, y + 24))
            draw.text((x + 5, y + 5), f"{variant} IMAGE {times[index]}s", fill="white")
    sheet.save(args.output / f"{variant}_image.jpg", quality=95)
    print(f"COMPLETE {variant}", flush=True)
(args.output / "model_probe.json").write_text(json.dumps(report), encoding="utf-8")
