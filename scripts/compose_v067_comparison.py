"""Full-rate source / previous target / new target comparison with shared camera."""
from pathlib import Path
import json,subprocess
import cv2,imageio_ffmpeg
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
old=ROOT/'research/v066/nahida-final'
new=ROOT/'research/v067/nahida'
source_path=ROOT/'data/jobs/1a650f45f8a9431c86646b2757ff0247/input/source.mp4'
old_render=json.loads((old/'preview_render_report.json').read_text(encoding='utf-8'))
new_render=json.loads((new/'preview_render_report.json').read_text(encoding='utf-8'))
old_camera=old_render.get('reference_camera');new_camera=new_render.get('reference_camera')
if not old_camera or not new_camera or Path(old_camera).resolve()!=Path(new_camera).resolve():
    raise ValueError('Comparison requires the same actual render reference camera, not just the saved project camera')
if len(old_render['frames'])!=901 or len(new_render['frames'])!=901:
    raise ValueError('Both full render reports must be complete')
report=json.loads((new/'retarget_report.json').read_text(encoding='utf-8'))
fps=report['fps'];n=report['frames'];cap=cv2.VideoCapture(str(source_path))
if abs(cap.get(cv2.CAP_PROP_FPS)-fps)>.001:raise ValueError('Comparison requires matching source fps')
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();silent=new/'comparison_v067_silent.mp4'
with (new/'comparison_v067_encode.log').open('wb') as log:
    proc=subprocess.Popen([ffmpeg,'-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','990x548','-r',str(fps),
          '-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','18','-pix_fmt','yuv420p',str(silent)],
          stdin=subprocess.PIPE,stderr=log)
    try:
        for i in range(n):
            ok,frame=cap.read()
            if not ok:raise RuntimeError('Source ended early')
            canvas=Image.new('RGB',(990,548),'#111923');draw=ImageDraw.Draw(canvas)
            for x,title in [(10,'上传原片'),(280,'v0.6.6'),(640,'v0.6.7 · 衣物拉扯保护')]:draw.text((x,8),title,font=font,fill='white')
            canvas.paste(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((270,480)),(0,38))
            for folder,x in [(old,270),(new,630)]:
                with Image.open(folder/'preview_frames'/f'{i+1:04d}.png') as im:canvas.paste(im.convert('RGB').resize((360,480)),(x,38))
            draw.text((10,526),f'{i/fps:05.2f}s | 新旧角色使用同一相机 · 接地为估计 · 局部误判仍可能存在',font=small,fill='#ccd6e0')
            proc.stdin.write(np.asarray(canvas).tobytes())
            if i in [36,180,219,263,300,420,450,510,690,880,884]:canvas.save(new/f'comparison_v067_{i:04d}.jpg',quality=94)
        proc.stdin.close()
        if proc.wait()!=0:raise RuntimeError('Encoding failed')
    finally:
        cap.release()
        if proc.poll() is None:proc.kill();proc.wait()
    final=new/'MotionBridge_v067_三列完整对照.mp4'
    subprocess.run([ffmpeg,'-y','-v','error','-i',str(silent),'-i',str(source_path),'-map','0:v:0','-map','1:a:0?',
                    '-c:v','copy','-c:a','aac','-b:a','160k','-t',str(n/fps),'-movflags','+faststart',str(final)],check=True,stderr=log)
    subprocess.run([ffmpeg,'-v','error','-xerror','-i',str(final),'-f','null','-'],check=True,stderr=log)
cap=cv2.VideoCapture(str(final));count=0
while cap.read()[0]:count+=1
actual_fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
assert count==n and abs(actual_fps-fps)<.001
verification={'file':str(final),'frames':count,'fps':actual_fps,'full_av_decode_passed':True,'same_camera':True,'verified_reference_camera':new_camera,'pose_edits_for_render':False}
(new/'comparison_v067_verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(verification,ensure_ascii=False),flush=True)
