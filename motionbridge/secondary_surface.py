"""Model-owned surface simulation helpers; no body-action or skin mutation."""
from pathlib import Path
import struct
import re
import numpy as np


def new_child_output(root,tag):
    if not isinstance(tag,str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}',tag):
        raise ValueError('输出标签只能含字母、数字、下划线和连字符')
    output=Path(root).resolve()/tag
    if output.exists():raise ValueError('输出目录已存在，拒绝覆盖')
    return output


def cloth_partition(plan):
    parts=[r for r in plan['regions'] if r['role']=='cloth']
    if not parts:raise ValueError('角色未配置衣物分区')
    vertices=sorted({v for r in parts for v in r['vertex_ids']})
    faces=sorted({f for r in parts for f in r['face_ids']})
    protected={v for r in plan['regions'] if r['role'] in ('body','boots') for v in r['vertex_ids']}
    pins={v:0. for v in vertices}
    for r in parts:
        for v,w in zip(r['vertex_ids'],r['pin_weights']):pins[v]=max(pins[v],w)
    for v in protected & set(vertices):pins[v]=1.
    indices=np.asarray(vertices,int)
    triangles=np.searchsorted(indices,np.asarray(plan['geometry']['triangles'],int)[faces])
    if triangles.max(initial=-1)>=len(vertices):raise ValueError('衣物面引用未分配顶点')
    return indices,triangles,np.asarray([pins[v] for v in vertices]),np.asarray(faces,int)


def rigid_fit(rest,target):
    """Proper Kabsch rigid attachment; preserves shape, not a 6-DOF solver."""
    rest=np.asarray(rest,float);target=np.asarray(target,float)
    if rest.shape!=target.shape or rest.ndim!=2 or rest.shape[1]!=3 or len(rest)<3 or not np.isfinite([rest,target]).all():
        raise ValueError('刚性部件需要匹配且有限的三维顶点')
    center=rest.mean(0);goal=target.mean(0);a=rest-center;b=target-goal
    if np.linalg.matrix_rank(a,tol=1e-9)<2:raise ValueError('刚性部件退化为线或点，需要专用绑定')
    u,_,vt=np.linalg.svd(a.T@b);rotation=vt.T@u.T
    if np.linalg.det(rotation)<0:
        vt[-1]*=-1;rotation=vt.T@u.T
    fitted=a@rotation.T+goal
    return fitted,rotation,goal-rotation@center


def rigid_components(plan,bone_tokens,maximum_rigid_extent=.16):
    positions=np.asarray(plan['geometry']['positions_m'])
    hard_faces={f for r in plan['regions'] if r['role']=='accessory' for f in r['face_ids']}
    hard_bones={str(b['id']) for b in plan['bones'] if any(t in b['name'] for t in bone_tokens)}
    skins=plan['geometry']['skin_weights']
    for region in plan['regions']:
        if region['role'] not in ('hair','cloth'):continue
        for f in region['face_ids']:
            if all(sum(w for b,w in skins[v].items() if b in hard_bones)>.95 for v in plan['geometry']['triangles'][f]):
                hard_faces.add(f)
    faces=np.asarray(plan['geometry']['triangles'],int)[sorted(hard_faces)]
    used=np.unique(faces);parent={int(v):int(v) for v in used}
    def find(v):
        while parent[v]!=v:parent[v]=parent[parent[v]];v=parent[v]
        return v
    def join(a,b):
        a,b=find(int(a)),find(int(b))
        if a!=b:parent[b]=a
    for a,b,c in faces:join(a,b);join(b,c)
    # UV seams can split rigid pieces; weld only selected hard corners.
    seen={}
    for v in used:
        key=tuple(np.round(positions[v],6))
        if key in seen:join(v,seen[key])
        else:seen[key]=v
    groups={}
    for v in used:groups.setdefault(find(int(v)),[]).append(int(v))
    protected={v for r in plan['regions'] if r['role'] in ('body','boots') for v in r['vertex_ids']}
    output=[];excluded=[]
    for vertices in groups.values():
        extent=float(np.ptp(positions[vertices],axis=0).max())
        influenced={int(b) for v in vertices for b,w in skins[v].items() if w>.01}
        ribbon_bones={b for b in influenced if any(t in plan['bones'][b]['name'] for t in ('衣帶','裙','夾克'))}
        gem_vertices={v for region in plan['regions'] if region['role']=='accessory' and '寶石' in region['materials']
                      for f in region['face_ids'] if plan['geometry']['material_names'][plan['geometry']['face_material_ids'][f]]=='寶石'
                      for v in plan['geometry']['triangles'][f]}
        flexible=len(ribbon_bones)>1 and not set(vertices)&gem_vertices
        reason=('shares_protected_body_vertices' if set(vertices)&protected else
                'flexible_trim_requires_cloth_follow' if extent>maximum_rigid_extent or flexible else
                'degenerate_component' if len(vertices)<3 or np.linalg.matrix_rank(positions[vertices]-positions[vertices].mean(0),tol=1e-9)<2 else None)
        (excluded if reason else output).append({'vertex_ids':vertices,'rest_extent_m':extent,**({'reason':reason} if reason else {})})
    return output,excluded,sorted(hard_faces)


class MDDWriter:
    """Streaming native MDD, big-endian float32; output is immutable on close."""
    def __init__(self,path,frames,vertices,fps):
        if type(frames) is not int or type(vertices) is not int or frames<1 or vertices<1 or not np.isfinite(fps) or fps<=0:
            raise ValueError('非法MDD缓存尺寸/帧率')
        self.file=Path(path).open('xb');self.frames=frames;self.vertices=vertices;self.written=0
        self.file.write(struct.pack('>2i',frames,vertices))
        self.file.write((np.arange(frames,dtype=float)/fps).astype('>f4').tobytes())
    def append(self,positions):
        p=np.asarray(positions)
        if p.shape!=(self.vertices,3) or not np.isfinite(p).all() or self.written>=self.frames:
            raise ValueError('缓存顶点形状/数据/帧数错误')
        self.file.write(p.astype('>f4').tobytes());self.written+=1
    def close(self):
        self.file.close()
        if self.written!=self.frames:raise ValueError('MDD缓存未完整写入')


def closest_segments(a0,a1,b0,b1):
    """Vectorized finite-segment closest points, including parallel/point cases."""
    arrays=[np.asarray(v,float) for v in (a0,a1,b0,b1)]
    if any(v.shape!=arrays[0].shape for v in arrays) or arrays[0].ndim!=2 or arrays[0].shape[1]!=3 or not np.isfinite(arrays).all():
        raise ValueError('非法线段数据')
    a0,a1,b0,b1=arrays;u=a1-a0;v=b1-b0;w=a0-b0
    aa=np.sum(u*u,1);bb=np.sum(u*v,1);cc=np.sum(v*v,1);dd=np.sum(u*w,1);ee=np.sum(v*w,1)
    denominator=aa*cc-bb*bb
    s=np.divide(bb*ee-cc*dd,denominator,out=np.zeros(len(aa)),where=denominator>1e-18)
    s=np.clip(s,0,1)
    for _ in range(3):
        t=np.clip(np.divide(bb*s+ee,cc,out=np.zeros(len(aa)),where=cc>1e-18),0,1)
        s=np.clip(np.divide(bb*t-dd,aa,out=np.zeros(len(aa)),where=aa>1e-18),0,1)
    return a0+s[:,None]*u,b0+t[:,None]*v,s,t


def floor_depth(points,radius,height=0.):
    p=np.asarray(points,float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all() or not np.isfinite([radius,height]).all() or radius<0:
        raise ValueError('非法地面接触数据')
    return np.maximum(height+radius-p[:,2],0)


def weld_driver_surface(rest,faces,pins,skin_keys):
    """Merge coincident UV seams only when their skin signature also agrees.

Faces remain the actual authored triangles; this is not a low-poly proxy.
Animated driver positions must independently verify every proposed weld.
"""
    rest=np.asarray(rest,float);faces=np.asarray(faces,int);pins=np.asarray(pins,float)
    if len(rest)!=len(pins) or len(rest)!=len(skin_keys):raise ValueError('焊接驱动数组长度不匹配')
    lookup={};representatives=[];inverse=[];weights=[]
    for i,point in enumerate(rest):
        key=(tuple(np.round(point,6)),skin_keys[i])
        if key not in lookup:
            lookup[key]=len(representatives);representatives.append(i);weights.append(float(pins[i]))
        group=lookup[key];weights[group]=max(weights[group],float(pins[i]));inverse.append(group)
    inverse=np.asarray(inverse,int);mapped=inverse[faces]
    nondegenerate=(mapped[:,0]!=mapped[:,1])&(mapped[:,1]!=mapped[:,2])&(mapped[:,0]!=mapped[:,2])
    candidate=mapped[nondegenerate]
    _,first=np.unique(np.sort(candidate,axis=1),axis=0,return_index=True)
    return np.asarray(representatives,int),candidate[np.sort(first)],np.asarray(weights),inverse


def anchor_unattached_components(rest,faces,pins,band=.015):
    """Rin model policy: fix the highest neutral-rest band of unanchored cloth.

Not inferred from a video pose. Existing anchors are preserved; tiny separate
components become kinematic instead of falling off the character.
"""
    rest=np.asarray(rest,float);faces=np.asarray(faces,int);result=np.asarray(pins,float).copy()
    parent=np.arange(len(rest))
    def find(v):
        while parent[v]!=v:parent[v]=parent[parent[v]];v=parent[v]
        return v
    for a,b,c in faces:
        root=find(a);parent[find(b)]=root;parent[find(c)]=root
    labels=np.array([find(i) for i in range(len(rest))]);records=[]
    for label in np.unique(labels):
        ids=np.flatnonzero(labels==label)
        if np.sum(result[ids]>=.999999)>=3:continue
        selected=ids[rest[ids,2]>=rest[ids,2].max()-band]
        if len(selected)<3:selected=ids[np.argsort(rest[ids,2])[-min(3,len(ids)):]]
        result[selected]=1
        records.append({'vertices':ids.tolist(),'new_fixed_vertices':selected.tolist()})
    return result,records
