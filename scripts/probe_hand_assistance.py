from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import json
from motionbridge.hands import HandObserver

root=Path(__file__).resolve().parents[1]
job=root/'data/jobs/0b1e96ce61304e9b84d29cee9d025245'
dest=root/'data/release-v040/hand-probe'
dest.mkdir(parents=True,exist_ok=True)
data=np.load(job/'outputs/observations.npz')
cap=cv2.VideoCapture(str(job/'outputs/source.mp4'))
observer=HandObserver(root/'models/hand_landmarker.task')
results=[]
for sec in [0,1,4,7,10,13,16,19,23,28]:
    f=round(sec*30)
    cap.set(cv2.CAP_PROP_POS_FRAMES,f)
    ok,frame=cap.read()
    assert ok
    candidates=observer.observe(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),data['image'][f],data['confidence'][f])
    for p in candidates:
        x,y=(np.array(p['wrist'])*[frame.shape[1],frame.shape[0]]).astype(int)
        cv2.circle(frame,(x,y),12,(0,255,0),3)
    for j in [15,16]:
        x,y=(data['image'][f,j,:2]*[frame.shape[1],frame.shape[0]]).astype(int)
        cv2.circle(frame,(x,y),9,(0,0,255),2)
    cv2.imwrite(str(dest/f'{sec:02d}.jpg'),frame)
    results.append({'time':sec,'candidates':candidates})
observer.close()
cap.release()
(dest/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results))
