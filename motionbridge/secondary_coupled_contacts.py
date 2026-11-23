"""Discrete true-triangle proximity with explicit bonded point equivalence.

No solid inside/outside inference. Existing proper intersections or ambiguous
coincident features fail closed; no silent category exclusion or source-reset
of collision counts. A bounded broad phase must enumerate all AABB overlaps.
"""
import numpy as np
from .triangle_contacts import closest_triangle, proper_triangle_crossings
from .secondary_joint_contacts import segment_closest


def aabb_pairs(points, triangles, gap, budget=100000):
    """Small-fixture broad phase, no quadratic allocation. Blender uses BVH."""
    t = points[triangles]; lo = t.min(1)-gap; hi = t.max(1)+gap
    out=[]
    for a in range(len(t)):
        match=np.flatnonzero(np.all(hi[a]>=lo[a+1:],axis=1)&np.all(hi[a+1:]>=lo[a],axis=1))+a+1
        if len(out)+len(match)>budget: raise ValueError('Contact broad-phase budget exceeded; no sampling')
        out.extend((a,int(b)) for b in match)
    return out


class SourceTriangleContacts:
    def __init__(self, mapping, mounts, connections, *, gap, floor_z, body_gap=None, pair_provider=aabb_pairs, pair_budget=100000):
        if not np.isfinite([gap,floor_z]).all() or not 0 < gap <= .05:
            raise ValueError('Invalid explicit contact distance/floor')
        if type(pair_budget) is not int or not 1 <= pair_budget <= 200000:
            raise ValueError('Invalid triangle pair budget')
        self.mapping=mapping; self.gap=float(gap); self.floor=float(floor_z)
        self.body_gap=float(gap if body_gap is None else body_gap)
        if not np.isfinite(self.body_gap) or not 0 < self.body_gap <= .05:
            raise ValueError('Invalid body clearance')
        self.provider=pair_provider; self.budget=pair_budget
        self.faces=np.asarray(mapping.physical_triangles,int)
        self.source_face_ids=np.asarray(mapping.physical_face_ids,int)
        owner={f:i for i,p in enumerate(mapping.parts) for f in p['face_ids']}
        self.parts=np.array([owner[int(f)] for f in self.source_face_ids])
        self.categories=[mapping.parts[i]['category'] for i in self.parts]
        # Only actual declared attachment/stitch equalities, no near-neighbour welding.
        parent=np.arange(len(mapping.rest))
        def find(v):
            while parent[v]!=v: v=int(parent[v])
            return v
        for a,b in connections['stitch_vertex_pairs']+[p for m in mounts['mounts'] for p in m['anchors']]:
            a=find(int(mapping.canonical_vertex[a])); b=find(int(mapping.canonical_vertex[b]))
            parent[max(a,b)]=min(a,b)
        self.bond=np.array([find(int(v)) for v in mapping.canonical_vertex])

    def _bonded(self, ids, coefficients):
        balance={}
        for v,w in zip(ids,coefficients):
            k=int(self.bond[v]); balance[k]=balance.get(k,0.)+float(w)
        return all(abs(w)<1e-7 for w in balance.values())

    def scan(self, points):
        self.mapping.check_positions(points); points=np.asarray(points)
        pairs=self.provider(points,self.faces,max(self.gap,self.body_gap),self.budget)
        if len(pairs)>self.budget: raise ValueError('Contact pair provider exceeded budget')
        pairs=np.asarray(pairs,int).reshape(-1,2)
        if len(pairs) and (pairs.min()<0 or pairs.max()>=len(self.faces) or np.any(pairs[:,0]>=pairs[:,1]) or len(np.unique(pairs,axis=0))!=len(pairs)):
            raise ValueError('Invalid or duplicate broad-phase pairs')
        crossings={}; examples=[]
        for start in range(0,len(pairs),8192):
            batch=pairs[start:start+8192]
            hit=proper_triangle_crossings(points[self.faces[batch[:,0]]],points[self.faces[batch[:,1]]],tolerance=1e-9)
            for a,b in batch[hit]:
                channel='/'.join(sorted((self.categories[a],self.categories[b])))
                crossings[channel]=crossings.get(channel,0)+1
                if len(examples)<24:
                    examples.append({'source_faces':[int(self.source_face_ids[a]),int(self.source_face_ids[b])],
                        'parts':[self.mapping.parts[self.parts[a]]['id'],self.mapping.parts[self.parts[b]]['id']], 'channel':channel})
        audit={'physical_triangle_count':len(self.faces),'candidate_pairs':len(pairs),
               'proper_crossings_by_channel':crossings,'proper_crossings':sum(crossings.values()),
               'crossing_examples':examples,'maximum_deficit_m':0.,'bonded_features':0,
               'ambiguous_features':0,'local_adjacent_pairs':0,'continuous_collision_checked':False,
               'raw_render_faces_audited':False}
        if crossings:
            return {'audit':audit,'constraints':[], 'blocking_reason':'intersecting_source_or_candidate_requires_authored_exit_side'}
        constraints=[]
        def feature(ids, coefficients, gap):
            q=sum(float(w)*points[int(v)] for v,w in zip(ids,coefficients)); distance=float(np.linalg.norm(q))
            if distance>=gap: return
            if self._bonded(ids,coefficients):
                audit['bonded_features']+=1; return
            if distance<=1e-10:
                audit['ambiguous_features']+=1; return
            normal=q/distance; value=distance-gap
            constraints.append({'ids':list(map(int,ids)), 'gradients':[float(w)*normal for w in coefficients], 'value':value, 'offset':gap})
            audit['maximum_deficit_m']=max(audit['maximum_deficit_m'],-value)
        for a,b in pairs:
            ia,ib=self.faces[a],self.faces[b]
            gap=self.body_gap if 'protected_surface' in (self.categories[a],self.categories[b]) else self.gap
            # Immediate neighbours within one indexed cloth patch are not self-contact.
            # Proper intersection was still tested above, before this exception.
            if self.parts[a]==self.parts[b] and set(ia)&set(ib):
                audit['local_adjacent_pairs']+=1; continue
            ta,tb=points[ia],points[ib]
            for ids,tri,other,othertri in ((ia,ta,ib,tb),(ib,tb,ia,ta)):
                nearest=closest_triangle(tri,np.repeat(othertri[None],3,axis=0))
                axes=np.column_stack((othertri[1]-othertri[0],othertri[2]-othertri[0]))
                if np.linalg.matrix_rank(axes,tol=1e-12)<2:
                    audit['ambiguous_features']+=1; continue
                for v,q in zip(ids,nearest):
                    uv=np.linalg.lstsq(axes,q-othertri[0],rcond=None)[0]
                    feature([v,*other],[1.,-(1-uv.sum()),-uv[0],-uv[1]],gap)
            for ea in ((0,1),(1,2),(2,0)):
                for eb in ((0,1),(1,2),(2,0)):
                    s,t,_,_=segment_closest(*ta[list(ea)],*tb[list(eb)])
                    feature([*ia[list(ea)],*ib[list(eb)]],[1-s,s,-(1-t),-t],gap)
        for v in np.unique(self.mapping.canonical_vertex):
            value=float(points[v,2]-self.floor)
            if value<0:
                constraints.append({'ids':[int(v)],'gradients':[np.array([0.,0.,1.])],'value':value,'offset':self.floor})
                audit['maximum_deficit_m']=max(audit['maximum_deficit_m'],-value)
        audit['constraint_count']=len(constraints)
        return {'audit':audit, 'constraints':constraints,
                'blocking_reason':'ambiguous_zero_distance_contact_requires_review' if audit['ambiguous_features'] else None}


def sweep_aabb_pairs(points, triangles, gap, budget):
    """Complete padded AABB sweep, bounded before pair allocation grows.

    Blender BVHTree.overlap rejects separated parallel faces even with epsilon;
    it is NOT a complete proximity broad phase. No Blender dependency here.
    """
    tri=points[triangles];lo=tri.min(1)-gap*.5;hi=tri.max(1)+gap*.5
    # Choose the axis with largest scene span, never omit an axis in filtering.
    axis=int(np.argmax(hi.max(0)-lo.min(0))) if len(tri) else 0
    other=[i for i in range(3) if i!=axis];order=np.argsort(lo[:,axis],kind='stable')
    active=np.empty(0,int);pairs=[]
    for a in order:
        active=active[hi[active,axis]>=lo[a,axis]]
        mask=np.all(hi[active][:,other]>=lo[a,other],axis=1)&np.all(hi[a,other]>=lo[active][:,other],axis=1)
        matched=active[mask]
        if len(pairs)+len(matched)>budget:raise ValueError('Full-mesh AABB pair budget exceeded; no sampling')
        pairs.extend((min(int(a),int(b)),max(int(a),int(b))) for b in matched)
        active=np.append(active,a)
    return sorted(pairs)
