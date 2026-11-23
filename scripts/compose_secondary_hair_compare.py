"""Compose and decode-check a silent, same-camera before/after hair comparison."""
from pathlib import Path
import argparse,hashlib,json,subprocess
import imageio_ffmpeg
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--pair-dir',type=Path,required=True)
args=parser.parse_args();root=args.pair_dir.resolve()
if not root.is_relative_to(ROOT/'output/secondary_motion'):raise ValueError('非法输出目录')
report=json.loads((root/'physics-validation.json').read_text(encoding='utf-8'))
assert report['passed'] and report['simulation_performed'] is True
render=json.loads((root/'frames/render-complete.json').read_text(encoding='utf-8'));assert render['same_camera']
paths={v:sorted((root/f'frames/{v}').glob('*.png')) for v in ('before','after')}
assert len(paths['before'])==len(paths['after'])==report['integer_frames']
assert [p.name for p in paths['before']]==[p.name for p in paths['after']]
output=root/'Rin_头发动力学_同相机前后对照.mp4'
if output.exists():raise ValueError('对照视频已存在，拒绝覆盖')
font=lambda size:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',size)
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();fps=render['fps'];duration=report['integer_frames']/fps
command=[ffmpeg,'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x900','-r',str(fps),'-i','pipe:0',
         '-an','-c:v','libx264','-threads','2','-preset','medium','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
overview=Image.new('RGB',(1600,1350),'#191c22')
with (root/'encode.log').open('x',encoding='utf-8') as log:
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=log,stderr=log)
    try:
        for index in range(len(paths['before'])):
            canvas=Image.new('RGB',(1280,900),'#191c22');draw=ImageDraw.Draw(canvas)
            draw.text((24,8),'施加物理前 · 原动作',font=font(25),fill='white')
            draw.text((664,8),'施加物理后 · 仅头发',font=font(25),fill='white')
            for column,variant in enumerate(('before','after')):
                with Image.open(paths[variant][index]) as frame:
                    assert frame.size==(640,800)
                    canvas.paste(frame.convert('RGB'),(column*640,50))
            draw.line((640,50,640,850),fill='#9ca3af',width=2)
            draw.text((24,862),f'同一相机 | {report["hair_bones"]}根发束 | {report["protected_bones"]}骨主动作保持 | {index/fps:05.2f}秒 / {duration:g}秒 | 实验版，无音频',
                      font=font(21),fill='#d1d5db')
            process.stdin.write(canvas.tobytes())
            if index==len(paths['before'])//2:canvas.save(root/'对照封面.jpg',quality=96)
            if index%30==0 and index//30<30:
                tile=canvas.resize((320,225))
                overview.paste(tile,((index//30%5)*320,(index//30//5)*225))
        process.stdin.close();code=process.wait(timeout=120)
        if code:raise RuntimeError(f'编码失败{code}')
    except BaseException:
        process.kill();process.wait();raise
overview.save(root/'逐秒前后对照.jpg',quality=94)
with (root/'decode-verification.log').open('x',encoding='utf-8') as log:
    subprocess.run([ffmpeg,'-v','error','-xerror','-i',str(output),'-f','null','-'],stdout=log,stderr=log,check=True)
reader=imageio_ffmpeg.read_frames(str(output));metadata=next(reader);decoded=sum(1 for _ in reader)
assert decoded==report['integer_frames']
verification={'passed':True,'video':str(output),'frames':decoded,'fps':metadata['fps'],'size':metadata['size'],
              'duration_seconds':decoded/metadata['fps'],'audio':False,'same_camera':True,
              'sha256':hashlib.file_digest(output.open('rb'),'sha256').hexdigest(),'full_decode_checked':True}
(root/'video-verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(verification,ensure_ascii=False,indent=2))
