"""Submit an explicitly supplied user clip and record this assessment separately."""
from pathlib import Path
import argparse
import json
import time
import cv2
import httpx
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("video", type=Path)
parser.add_argument("--folder", default="dance-20261022")
args = parser.parse_args()
folder = ROOT / "data" / "user-tests" / args.folder
folder.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(str(args.video))
fps, frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frames / fps
times = [min(duration - .1, t) for t in [1, 4, 7, 10, 13, 16, 19, 23, 28]]
sheet = Image.new("RGB", (720, 3 * 452), "#14181d")
draw = ImageDraw.Draw(sheet)
for index, sec in enumerate(times):
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    good, frame = cap.read()
    if good:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img.thumbnail((240, 424))
        x, y = index % 3 * 240, index // 3 * 452
        sheet.paste(img, (x + (240 - img.width) // 2, y + 24))
        draw.text((x + 8, y + 7), f"SOURCE  {sec:05.2f}s", fill="white")
cap.release()
sheet.save(folder / "source_contact_sheet.jpg", quality=92)
settings = {"model": "full", "sample_fps": 30, "smoothing": .35, "exports_3d": "true"}
report = {"source_filename": args.video.name, "source_fps": fps, "source_frames": frames,
          "source_duration": duration, "settings": settings, "sample_times": times}
(folder / "assessment.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print("PREPARED " + str(folder), flush=True)
with httpx.Client(base_url="http://127.0.0.1:8765", timeout=90, trust_env=False) as client:
    with args.video.open("rb") as stream:
        response = client.post("/api/jobs", files={"file": ("user-dance-30s.mp4", stream, "video/mp4")}, data=settings)
    response.raise_for_status()
    job = response.json()
    report["job_id"] = job["id"]
    started = time.monotonic()
    while job["status"] in {"queued", "processing"}:
        print(json.dumps({"status": job["status"], "stage": job["stage"], "progress": job["progress"]}, ensure_ascii=False), flush=True)
        time.sleep(3)
        if time.monotonic() - started > 900:
            raise RuntimeError("Assessment timed out; inspect the job in the app")
        job = client.get(f"/api/jobs/{job['id']}").json()
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    report["job"] = job
    report["outputs"] = str(ROOT / "data" / "jobs" / job["id"] / "outputs")
    (folder / "assessment.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if job["status"] != "completed":
        raise RuntimeError(job.get("error") or job["status"])
