"""Evaluate fixed visible-wrist labels with one frozen baseline camera per clip.

Labels and camera are approximate; report pixel errors and uncertainty, never
claim 3D accuracy from a small 2D screen-space review.
"""
from pathlib import Path
import json, argparse
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
root=Path(__file__).resolve().parents[1]; dest=root/'data/release-v050'
labels=json.loads((dest/'manual-reference/visible_wrists.json').read_text())
ids=json.loads((dest/'final_jobs.json').read_text())
parser=argparse.ArgumentParser(); parser.add_argument('--candidate-prefix'); parser.add_argument('--release-dir',type=Path); parser.add_argument('--version',default='v050'); args=parser.parse_args()
if args.release_dir:
    dest=args.release_dir.resolve()
    if (dest/'final_jobs.json').exists(): ids=json.loads((dest/'final_jobs.json').read_text())
rows=[]
for name,oldid in [('dance','b6f9d511e8214257b7df4fab6df2196f'),('tennis','288450f7d88848609abd9d7c52351f8e')]:
    oldpath=root/'data/jobs'/oldid/'outputs'
    old=json.loads((oldpath/'motion.json').read_text())
    candidate=dest/(args.candidate_prefix+'-'+name)/'motion.json' if args.candidate_prefix else root/'data/jobs'/ids[name]/'outputs/motion.json'
    new=json.loads(candidate.read_text())
    a=np.load(oldpath/'observations.npz'); im=a['image']; c=a['confidence']
    w,h=old['source']['normalized_width'],old['source']['normalized_height']
    op=np.array([f['positions'] for f in old['frames']]); np_=np.array([f['positions'] for f in new['frames']])
    hip=im[:,[23,24],:2].mean(1); xy=(im[:,:,:2]-hip[:,None])*[w/h,-1]
    mp,js=[11,12,23,24],[6,10,13,17]
    weights=np.clip((c[:,mp]-.4)/.4,0,1)
    torso=op[:,js,:2]-op[:,:1,:2]
    den=np.sum(xy[:,mp]**2*weights[...,None],axis=(1,2))
    scale=np.sum(xy[:,mp]*torso*weights[...,None],axis=(1,2))/np.maximum(den,1e-9)
    valid=(np.sum(weights>.5,axis=1)>=3)&(scale>.1)&(scale<20)&(den>1e-5)
    t=np.arange(len(op)); scale=np.interp(t,t[valid],scale[valid])
    scale=gaussian_filter1d(median_filter(scale,size=max(1,round(old['fps']*.15))|1),max(.5,old['fps']*.08))
    for sample in labels['samples']:
        if sample['clip']!=name: continue
        f=round(sample['time']*old['fps']); j=old['joint_names'].index(sample['joint'])
        row=dict(sample)
        for version,poses in [('v040',op),(args.version,np_)]:
            point=((poses[f,j,:2]-poses[f,0,:2])/scale[f]/[w/h,-1]+hip[f])*[w,h]
            row[version+'_projected_xy']=point.round(2).tolist()
            row[version+'_error_pixels']=float(np.linalg.norm(point-sample['xy']))
        row['improvement_pixels']=row['v040_error_pixels']-row[args.version+'_error_pixels']
        rows.append(row)
report={'method':labels['method'],'uncertainty_pixels':labels['uncertainty_pixels'],
        'camera':'Frozen v0.4 shoulder/hip weak perspective; source hip alignment removes root translation, same camera for both versions.',
        'samples':rows,'summary':{}}
for clip in ['dance','tennis']:
    subset=[r for r in rows if r['clip']==clip]
    report['summary'][clip]={'count':len(subset),**{v+'_mean_pixels':float(np.mean([r[v+'_error_pixels'] for r in subset])) for v in ['v040',args.version]},
                           'improved_beyond_15px':sum(r['improvement_pixels']>15 for r in subset),
                           'regressed_beyond_15px':sum(r['improvement_pixels']<-15 for r in subset)}
(dest/(('manual_wrist_'+args.candidate_prefix+'.json') if args.candidate_prefix else 'manual_wrist_comparison.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
