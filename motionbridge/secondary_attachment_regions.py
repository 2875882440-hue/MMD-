"""Explicit experimental attachment masks; no automatic anatomical pinning."""
import numpy as np

def compile_regions(config, source_ids, faces):
    ids=list(map(int,source_ids));known=set(ids)
    if len(known)!=len(ids) or not config.get('evidence'):
        raise ValueError('Unique source IDs and attachment evidence required')
    groups=[config.get(k,[]) for k in ('fixed','driven','free')]
    if any(not g or len(g)!=len(set(g)) for g in groups):raise ValueError('Three nonempty unique regions required')
    sets=list(map(set,groups))
    if set.union(*sets)!=known or any(sets[i]&sets[j] for i,j in ((0,1),(0,2),(1,2))):raise ValueError('Regions must partition the patch exactly')
    faces=np.asarray(faces,int)
    if faces.ndim!=2 or faces.shape[1]!=3 or faces.min()<0 or faces.max()>=len(ids):raise ValueError('Invalid patch triangles')
    adjacency=[set() for _ in ids]
    for a,b,c in faces:
        for u,v in ((a,b),(b,c),(c,a)):adjacency[u].add(v);adjacency[v].add(u)
    unseen=set(range(len(ids)))
    while unseen:
        stack=[unseen.pop()];component=set(stack)
        while stack:
            for v in adjacency[stack.pop()] & unseen:unseen.remove(v);component.add(v);stack.append(v)
        if not {ids[v] for v in component}&sets[0]:raise ValueError('Unanchored component')
    weights=config.get('pin_weights',{})
    if weights.get('fixed')!=1 or weights.get('free')!=0 or not 0<weights.get('driven',0)<1:raise ValueError('Invalid region strengths')
    return np.array([weights[next(k for k,g in zip(('fixed','driven','free'),sets) if v in g)] for v in ids],float)
