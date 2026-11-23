"""Read-only prerequisite inventory. Does not install, load pickles, or run inference."""
from pathlib import Path
import json, platform, sys
root=Path(__file__).resolve().parents[1]
repo=root/'research/GVHMR'
paths=['body_models/smplx/SMPLX_NEUTRAL.npz','body_models/smpl/SMPL_NEUTRAL.pkl',
       'gvhmr/gvhmr_siga24_release.ckpt','hmr2/epoch=10-step=25000.ckpt',
       'vitpose/vitpose-h-multi-coco.pth','yolo/yolov8x.pt']
report={'source_downloaded':(repo/'tools/demo/demo.py').exists(),
        'source_commit':(repo/'.git/refs/heads/main').read_text().strip() if (repo/'.git/refs/heads/main').exists() else None,
        'platform':platform.system(),'current_python':sys.version.split()[0],
        'official_environment':'Python 3.10, torch 2.3.0+cu121; requirements include a Linux-only PyTorch3D wheel',
        'files':{p:(repo/'inputs/checkpoints'/p).is_file() for p in paths},
        'licensed_drop_folder':str(root/'models/licensed'),
        'inference_tested':None,'integrated':None,
        'optional_official_renderer_asset':'body_models/smpl/SMPL_NEUTRAL.pkl',
        'validation_record':'docs/VALIDATION_0.6.0.md',
        'note':'Presence inventory only, not a runtime or integration test. SMPL PKL is optional for the MotionBridge Blender path. No weights loaded.'}
print(json.dumps(report,ensure_ascii=False,indent=2))
