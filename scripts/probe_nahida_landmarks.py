import bpy, json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
bpy.ops.wm.open_mainfile(filepath=str(root.parent/'mmd-dance/KamiNoManimani/Nahida_KamiNoManimani_Toon_SmoothMotion30.blend'),load_ui=False)
rig=bpy.data.objects['Nahida_arm']
items={b.name:{'head':list(b.head_local),'tail':list(b.tail_local),'parent':b.parent.name if b.parent else None}
       for b in rig.data.bones if any(k in b.name for k in ['頭','目','肩','首','腕','ひじ','足首'])}
out=root/'data/release-v050'; out.mkdir(parents=True,exist_ok=True)
(out/'nahida_landmarks.json').write_text(json.dumps(items,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(items,ensure_ascii=False))
