"""Extract source-only grid frames for manual visible-wrist review."""
from pathlib import Path
import cv2
from PIL import Image, ImageDraw
root=Path(__file__).resolve().parents[1]
dest=root/'data/release-v050/manual-reference'; dest.mkdir(exist_ok=True)
for name,ident,times in [('dance','b6f9d511e8214257b7df4fab6df2196f',[1,4,7,10,16,28]),('tennis','288450f7d88848609abd9d7c52351f8e',[4,7])]:
    cap=cv2.VideoCapture(str(root/'data/jobs'/ident/'input/source.mp4'))
    fps=cap.get(cv2.CAP_PROP_FPS)
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES,round(t*fps)); ok,frame=cap.read(); assert ok
        im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); draw=ImageDraw.Draw(im)
        for x in range(0,im.width,50):
            draw.line((x,0,x,im.height),fill=(150,150,150),width=1)
            for y in range(0,im.height,200): draw.text((x+2,y+2),str(x),fill='white',stroke_width=1,stroke_fill='black')
        for y in range(0,im.height,50):
            draw.line((0,y,im.width,y),fill=(150,150,150),width=1)
            draw.text((2,y+2),str(y),fill='white',stroke_width=1,stroke_fill='black')
        im.save(dest/f'{name}-{t}.jpg',quality=94)
    cap.release()
