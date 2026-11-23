"""Compose full same-camera frames and verify every decoded video frame."""
from pathlib import Path
import argparse,hashlib,json,subprocess
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--pair-dir',type=Path,required=True);parser.add_argument('--input-dir',type=Path,required=True)
args=parser.parse_args();root=args.pair_dir.resolve();source=args.input_dir.resolve()
if not root.is_relative_to(ROOT/'output/secondary_motion'):raise ValueError('非法输出目录')
camera=json.loads((root/'camera.json').read_text(encoding='utf-8'));assert camera['passed']
report=json.loads((root.parent/'surface-validation.json').read_text(encoding='utf-8'));assert report['passed']
render=json.loads((root/'frames/render-complete.json').read_text(encoding='utf-8'));assert render['same_camera']
paths={v:sorted((root/f'frames/{v}').glob('*.png')) for v in ('before','after')}
assert len(paths['before'])==len(paths['after'])==report['frames']
assert [p.name for p in paths['before']]==[p.name for p in paths['after']]
output=root/'Rin_衣发碰撞_同相机前后对照.mp4'
if output.exists():raise ValueError('已有视频，拒绝覆盖')
font=lambda size:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',size)
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();fps=render['fps'];size=render['cameras']['before']['size'];w,h=size
width=w*2;height=h+100;assert width%2==height%2==0
command=[ffmpeg,'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','medium','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
overview=Image.new('RGB',(1600,1350),'#191c22')
with (root/'encode.log').open('x',encoding='utf-8') as log:
 process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=log,stderr=log)
 try:
  for index in range(report['frames']):
   canvas=Image.new('RGB',(width,height),'#191c22');draw=ImageDraw.Draw(canvas)
   draw.text((20,8),'施加本轮物理前 · 第3步验收版',font=font(24),fill='white')
   draw.text((w+20,8),'本轮物理后 · 衣服 / 头发 / 硬饰',font=font(24),fill='white')
   for column,variant in enumerate(('before','after')):
    with Image.open(paths[variant][index]) as frame:
     assert frame.size==tuple(size);canvas.paste(frame.convert('RGB'),(column*w,50))
   draw.line((w,50,w,h+50),fill='#9ca3af',width=2)
   draw.text((20,h+60),f'同相机 | 48骨主动作保持 | {index/fps:05.2f}/{report["frames"]/fps:g} 秒 | 实验输出，残余穿插见报告',font=font(21),fill='#d1d5db')
   process.stdin.write(canvas.tobytes())
   if index==report['frames']//2:canvas.save(root/'对照封面.jpg',quality=96)
   if index%30==0 and index//30<30:overview.paste(canvas.resize((320,225)),((index//30%5)*320,(index//30//5)*225))
  process.stdin.close();code=process.wait(timeout=120)
  if code:raise RuntimeError(f'编码失败{code}')
 except BaseException:process.kill();process.wait();raise
overview.save(root/'逐秒前后对照.jpg',quality=94)
with (root/'decode-verification.log').open('x',encoding='utf-8') as log:
 subprocess.run([ffmpeg,'-v','error','-xerror','-i',str(output),'-f','null','-'],stdout=log,stderr=log,check=True)
reader=imageio_ffmpeg.read_frames(str(output));metadata=next(reader);decoded=sum(1 for _ in reader)
assert decoded==report['frames']
verification={'passed':True,'video':str(output),'frames':decoded,'fps':metadata['fps'],'size':metadata['size'],'duration_seconds':decoded/metadata['fps'],'audio':False,'same_camera':True,'sha256':hashlib.file_digest(output.open('rb'),'sha256').hexdigest(),'full_decode_checked':True,'camera_complete_actor_verified':True}
(root/'video-verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(verification,ensure_ascii=False,indent=2))
