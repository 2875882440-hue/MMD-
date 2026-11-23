"""Re-infer on preserved image features with refined 2D observations, never edit old jobs."""
from pathlib import Path
import json, shutil, subprocess, sys
import torch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from motionbridge.body_identity import stabilize_arm_observations
for name,ident in [('dance','f2bdc1313b3343c39f14a55dcb8a35d5'),('tennis','21c5391351334cb5bcbf13cfdcfbf963')]:
    source=ROOT/'data/jobs'/ident/'outputs'
    destination=ROOT/'research/v061'/name
    destination.mkdir(exist_ok=True)
    if (destination/'motion_world.npz').exists():
        raise RuntimeError('Existing trial: do not overwrite '+str(destination))
    for filename in ['bbx.pt','vit_features.pt','slam_results.pt']:
        shutil.copy2(source/'gvhmr'/filename,destination/filename)
    kp=np.load(source/'gvhmr/keypoints_2d.npz')['keypoints']
    boxes=torch.load(destination/'bbx.pt',map_location='cpu')['bbx_xys'].numpy()
    corrected,report=stabilize_arm_observations(kp,boxes)
    torch.save(torch.from_numpy(corrected).float(),destination/'vitpose.pt')
    np.savez_compressed(destination/'keypoints_2d.npz',keypoints=corrected)
    (destination/'arm_report.json').write_text(json.dumps(report,indent=2))
    config=json.loads((source/'gvhmr/run_config.json').read_text())
    config['keypoint_identity']='temporal_local'
    (destination/'run_config.json').write_text(json.dumps(config,indent=2))
    (destination/'cache_provenance.json').write_text(json.dumps({'source_job':ident,'reused':['bbx','image_features','camera'],'modified':'arm observations only'},indent=2))
    with (destination/'infer.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts/run_gvhmr_probe.py'),'--video',str(source/'source.mp4'),
                        '--output',str(destination),'--stage','infer','--feature-precision','fp16',
                        '--keypoint-identity','temporal_local'],stdout=log,stderr=subprocess.STDOUT,check=True)
    print('INFERENCE_DONE',name,flush=True)
