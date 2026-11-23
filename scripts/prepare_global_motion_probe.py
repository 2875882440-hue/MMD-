"""Prepare a bounded regression window without changing the uploaded video."""
from pathlib import Path
import json,subprocess
import numpy as np
import cv2
import imageio_ffmpeg
from PIL import Image,ImageDraw
root=Path(__file__).resolve().parents[1]
folder=root/'research/global-motion-probe';folder.mkdir(exist_ok=True)
source=root/'data/jobs/1a650f45f8a9431c86646b2757ff0247'
m=json.loads((source/'outputs/motion.json').read_text(encoding='utf-8'))
p=np.asarray([f['positions'][0] for f in m['frames']])
(folder/'current_root.json').write_text(json.dumps({'motion':str(source/'outputs/motion.json'),
    'root_mode':m['root_mode'],'axis_range_m':np.ptp(p,axis=0).tolist(),
    'note':'Y-up. Horizontal X/Z translation is discarded by the current engine.'},indent=2),encoding='utf-8')
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-ss','12','-i',str(source/'input/source.mp4'),
    '-t','6','-c:v','libx264','-threads','2','-crf','18','-c:a','aac',str(folder/'dance_12_18s.mp4')],check=True)
cap=cv2.VideoCapture(str(source/'input/source.mp4'));sheet=Image.new('RGB',(240*6,450),'#151515');draw=ImageDraw.Draw(sheet)
for k,t in enumerate([12,13,14,15,16,17]):
    cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,frame=cap.read();assert ok
    im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));im.thumbnail((240,426))
    sheet.paste(im,(k*240,24));draw.text((k*240+8,5),f'{t}s source',fill='white')
cap.release();sheet.save(folder/'source_12_17s.jpg')
print('Prepared',folder)
