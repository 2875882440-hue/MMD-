"""Release check: a valid video with no person must not produce fake motion."""
from pathlib import Path
import json
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from motionbridge.pipeline import process_video

folder = ROOT / "data" / "negative-validation"
folder.mkdir(parents=True, exist_ok=True)
source = folder / "no-person.mp4"
writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 12, (320, 240))
if not writer.isOpened():
    raise RuntimeError("Cannot create validation input")
for _ in range(12):
    writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
writer.release()
try:
    process_video(source, folder / "outputs", {"model": "lite", "sample_fps": 12, "smoothing": .55},
                  lambda *args: None, lambda: False)
except RuntimeError as exc:
    message = str(exc)
    passed = any(text in message for text in ["人物", "人体", "姿态", "身体"]) and not (folder / "outputs" / "motion.json").exists()
    report = {"passed": passed, "case": "real no-person video rejected", "error": message,
              "motion_created": (folder / "outputs" / "motion.json").exists()}
else:
    report = {"passed": False, "case": "real no-person video incorrectly accepted"}
(folder / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False))
raise SystemExit(0 if report["passed"] else 1)
