"""Verify real jobs without changing their artifacts; export auditable evidence."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from motionbridge.exporters import solve_rotations, validate_motion

parser = argparse.ArgumentParser()
parser.add_argument("jobs", nargs="+")
parser.add_argument("--original", required=True)
parser.add_argument("--output", type=Path, default=ROOT / "data" / "release-v020")
args = parser.parse_args()
args.output.mkdir(exist_ok=True, parents=True)
reports = []
for job_id in args.jobs:
    folder = ROOT / "data" / "jobs" / job_id
    job = json.loads((folder / "job.json").read_text(encoding="utf-8"))
    outputs = folder / "outputs"
    motion = json.loads((outputs / "motion.json").read_text(encoding="utf-8"))
    verification = json.loads((outputs / "blender_verification.json").read_text(encoding="utf-8"))
    names, parents, rest, poses, fps = validate_motion(motion)
    local, _, fk = solve_rotations(motion)
    preview_error = float(np.linalg.norm(fk - poses, axis=-1).max())
    bone_error = max(float(np.abs(np.linalg.norm(poses[:, c] - poses[:, p], axis=-1) - np.linalg.norm(rest[c] - rest[p])).max())
                     for c, p in enumerate(parents) if p >= 0)
    with np.load(outputs / "observations.npz", allow_pickle=False) as observations:
        cache_valid = all(key in observations for key in ("world", "image", "confidence"))
        cache_valid &= observations["world"].shape == (len(poses), 33, 3)
    passed = bool(job["status"] == "completed" and verification["passed"] and cache_valid
                  and preview_error < 1e-6 and bone_error < 1e-6
                  and np.allclose(np.linalg.det(local), 1, atol=1e-6))
    reports.append({"job_id": job_id, "name": job["name"], "frames": len(poses), "fps": fps,
                    "solver": motion.get("solver"), "passed": passed, "all_frame_preview_fk_max_error_m": preview_error,
                    "all_frame_bone_length_max_error_m": bone_error, "refinement": motion.get("refinement"),
                    "blender": verification, "reprocessed_from": job.get("reprocessed_from")})
original = ROOT / "data" / "jobs" / args.original
previous = json.loads((args.output / "original-job-hashes.json").read_text(encoding="utf-8"))
current = {str(p.relative_to(original)): hashlib.sha256(p.read_bytes()).hexdigest()
           for p in original.rglob("*") if p.is_file()}
result = {"jobs": reports, "original_job": args.original, "original_files": len(previous),
          "original_unchanged": previous == current, "passed": all(r["passed"] for r in reports) and previous == current,
          "metric_note": "Image/endpoint residuals measure internal fit to estimates. They are not ground-truth capture accuracy."}
(args.output / "release_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"passed": result["passed"], "original_unchanged": result["original_unchanged"],
                  "jobs": [{k: r[k] for k in ["job_id", "frames", "fps", "passed", "all_frame_preview_fk_max_error_m"]} for r in reports]}, indent=2))
if not result["passed"]:
    raise SystemExit(1)
