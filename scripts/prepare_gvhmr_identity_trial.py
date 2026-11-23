"""Create a new experimental cache; preserve every original model output."""
from pathlib import Path
import argparse
import json
import shutil
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from motionbridge.body_identity import stabilize_coco_identity

parser=argparse.ArgumentParser()
parser.add_argument('source',type=Path)
parser.add_argument('target',type=Path)
args=parser.parse_args()
args.target.mkdir(parents=True,exist_ok=False)
for name in ['bbx.pt','slam_results.pt','vit_features.pt','track_telemetry.json','camera_telemetry.json','features_telemetry.json']:
    shutil.copy2(args.source/name,args.target/name)
points=torch.load(args.source/'vitpose.pt',map_location='cpu')
boxes=torch.load(args.source/'bbx.pt',map_location='cpu')['bbx_xys']
corrected, report=stabilize_coco_identity(points.numpy(),boxes.numpy())
torch.save(points,args.target/'vitpose_raw.pt')
torch.save(torch.from_numpy(corrected).float(),args.target/'vitpose.pt')
(args.target/'identity_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
telemetry=json.loads((args.source/'pose_telemetry.json').read_text())
telemetry['identity_correction']=report
(args.target/'pose_telemetry.json').write_text(json.dumps(telemetry,indent=2),encoding='utf-8')
config=json.loads((args.source/'run_config.json').read_text())
config['keypoint_identity']='temporal'
(args.target/'run_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
