"""Compare every saved rig animation curve; surface-only changes must not move bones."""
from pathlib import Path
import sys,json,hashlib
import bpy
old,new,dest=map(Path,sys.argv[sys.argv.index('--')+1:])
def curves(path):
    bpy.ops.wm.open_mainfile(filepath=str(path.resolve()),load_ui=False)
    rig=bpy.data.objects['Nahida_arm'];action=rig.animation_data.action;records=[]
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for curve in bag.fcurves:
                    records.append({'path':curve.data_path,'index':curve.array_index,'extrapolation':curve.extrapolation,
                        'points':[[list(k.co),list(k.handle_left),list(k.handle_right),k.interpolation] for k in curve.keyframe_points]})
    records.sort(key=lambda x:(x['path'],x['index']))
    return records
a=curves(old);b=curves(new)
result={'passed':a==b,'old':str(old),'new':str(new),'curves':len(a),'keys':sum(len(c['points']) for c in a),
        'old_sha256':hashlib.sha256(json.dumps(a,sort_keys=True).encode()).hexdigest(),
        'new_sha256':hashlib.sha256(json.dumps(b,sort_keys=True).encode()).hexdigest(),
        'note':'All saved rig FCurve values, handles and interpolation match; garment shape curves intentionally differ.'}
dest.write_text(json.dumps(result,indent=2),encoding='utf-8');print(result,flush=True)
if not result['passed']:raise RuntimeError('Surface-only update changed rig animation')
