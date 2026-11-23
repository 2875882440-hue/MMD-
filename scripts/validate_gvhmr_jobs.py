"""Run the real application queue against preserved dance/tennis inputs in isolated data."""
from pathlib import Path
import os
import sys
import json
import argparse
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=ROOT/'research/global-motion-regression')
args=parser.parse_args()
destination=args.output.resolve()
destination.mkdir(parents=True,exist_ok=True)
os.environ["MOTIONBRIDGE_DATA"] = str(destination / "jobs")
from fastapi.testclient import TestClient
from motionbridge import app as server

results = []
with TestClient(server.app) as client:
    for name, original in [("dance", "1a650f45f8a9431c86646b2757ff0247"),
                           ("tennis", "72a0ddfbac1040e5b3a3ddf7f05ae1e7")]:
        with (ROOT / "data/jobs" / original / "input/source.mp4").open("rb") as source:
            response = client.post("/api/jobs", files={"file": (name + ".mp4", source, "video/mp4")},
                                   data={"engine": "gvhmr_experimental", "exports_3d": "true"})
        response.raise_for_status()
        job_id = response.json()["id"]
        print(f"START {name} {job_id}", flush=True)
        import time
        last = None
        while True:
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["stage"] != last:
                print(f"{name}: {job['status']} {job['stage']}", flush=True)
                last = job["stage"]
            if job["status"] not in {"queued", "processing"}:
                break
            time.sleep(2)
        results.append({"name": name, "source_job": original, "job_id": job_id,
                        "status": job["status"], "error": job["error"], "meta": job["meta"]})
        (destination / "results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        if job["status"] != "completed":
            raise RuntimeError(job["error"] or job["status"])
print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
