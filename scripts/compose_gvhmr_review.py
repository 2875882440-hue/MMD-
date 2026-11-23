"""Compose timestamp-aligned original/Blender evidence at the full input FPS."""
from pathlib import Path
import argparse
import json
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

parser=argparse.ArgumentParser()
parser.add_argument('folder',type=Path)
parser.add_argument('source',type=Path)
parser.add_argument('--source-start',type=float,default=0.)
args=parser.parse_args()
folder=args.folder.resolve()
motion=json.loads((folder/'motion.json').read_text(encoding='utf-8'))
fps=motion['fps']; count=motion['num_frames']
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
cap=cv2.VideoCapture(str(args.source))
temp=folder/'comparison_working.mp4'
writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),fps,(720,716))
if not writer.isOpened(): raise RuntimeError('Cannot create comparison video')
samples=[]
try:
    for f in range(count):
        ok,frame=cap.read()
        if not ok: raise RuntimeError(f'Source ended at frame {f}')
        canvas=Image.new('RGB',(720,716),'#101923')
        draw=ImageDraw.Draw(canvas)
        draw.text((12,8),'原视频',font=font,fill='white')
        draw.text((372,8),'GVHMR + 左右连续性纠错',font=font,fill='white')
        source=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((360,640))
        with Image.open(folder/'world_frames_initial_view'/f'{f+1:05d}.png') as rendered:
            canvas.paste(rendered.convert('RGB'),(360,38))
        canvas.paste(source,(0,38))
        draw.text((12,685),f'原片 {args.source_start+f/fps:.2f}s · {fps:g} FPS · 世界位移为模型估计，尚未标定',font=small,fill='#c2cfdd')
        writer.write(cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR))
        if f in {0,60,90}: samples.append(canvas.copy())
finally:
    cap.release();writer.release()
dest=folder/'GVHMR_转身与移动对照.mp4'
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run([ffmpeg,'-y','-v','error','-i',str(temp),'-i',str(args.source),'-map','0:v','-map','1:a?',
                '-c:v','libx264','-threads','2','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-shortest',
                '-movflags','+faststart',str(dest)],check=True)
subprocess.run([ffmpeg,'-v','error','-i',str(dest),'-f','null','-'],check=True)
sheet=Image.new('RGB',(720,716*len(samples)))
for i,im in enumerate(samples): sheet.paste(im,(0,716*i))
sheet.save(folder/'comparison_contact_sheet.jpg')
capture=cv2.VideoCapture(str(dest))
actual=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)); capture.release()
if actual!=count: raise RuntimeError(f'Comparison frame count {actual} != {count}')
(folder/'comparison_verification.json').write_text(json.dumps({'frames':actual,'fps':fps,'full_decode_passed':True,
    'source_start_s':args.source_start,'pose_edits_for_render':False,'file':str(dest)},indent=2),encoding='utf-8')
print(dest)
