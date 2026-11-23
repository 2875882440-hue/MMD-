"""Convert validated experimental world-motion output to MotionBridge JSON/BVH."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from motionbridge.global_motion import gvhmr_to_motion
from motionbridge.exporters import export_bvh

parser = argparse.ArgumentParser()
parser.add_argument("input", type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
motion = gvhmr_to_motion(args.input)
args.output.mkdir(parents=True, exist_ok=True)
(args.output / "motion.json").write_text(json.dumps(motion, ensure_ascii=False), encoding="utf-8")
export_bvh(motion, args.output / "motion.bvh")
print(json.dumps(motion["global_motion"], ensure_ascii=False, indent=2))
