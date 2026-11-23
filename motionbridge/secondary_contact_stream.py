"""Bounded complete triangle scan and explicit initial-intersection policy.

Blocks own disjoint pairs, not independent physical worlds. Only a final scan
of the whole corrected state can accept a candidate. No source motion writes.
"""
import hashlib
import numpy as np
from .secondary_coupled_contacts import SourceTriangleContacts
from .triangle_contacts import closest_triangle, proper_triangle_crossings


def geometry_key(mapping):
    h=hashlib.sha256()
    for a in (np.asarray(mapping.rest,dtype='<f8'),np.asarray(mapping.triangles,dtype='<i8')):
        h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def iter_aabb_blocks(points,triangles,gap,block_size=4096):
    """Every padded AABB pair exactly once, including across yield boundaries.

    Sweep ownership belongs to the later face in the stable sorted order.
    Active intervals survive yields; no accumulated all-pairs list or set.
    """
    if type(block_size) is not int or not 1<=block_size<=4096:
        raise ValueError('Block size must be 1..4096; feature budget stays below 100000')
    if not np.isfinite(gap) or not 0<gap<=.05:raise ValueError('Invalid gap')
    points=np.asarray(points,float);triangles=np.asarray(triangles,int)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all() or triangles.ndim!=2 or triangles.shape[1]!=3:
        raise ValueError('Invalid triangle arrays')
    if not len(triangles):return
    if triangles.min()<0 or triangles.max()>=len(points):raise ValueError('Invalid triangle index')
    tri=points[triangles];lo=tri.min(1)-gap*.5;hi=tri.max(1)+gap*.5
    axis=int(np.argmax(hi.max(0)-lo.min(0)));other=[i for i in range(3) if i!=axis]
    order=np.argsort(lo[:,axis],kind='stable');active=np.empty(0,int)
    block=np.empty((block_size,2),np.int64);used=0
    for a in order:
        active=active[hi[active,axis]>=lo[a,axis]]
        # Subtile even the active set: no face can allocate all matching pairs.
        for start in range(0,len(active),block_size):
            ids=active[start:start+block_size]
            mask=np.all(hi[ids][:,other]>=lo[a,other],axis=1)&np.all(hi[a,other]>=lo[ids][:,other],axis=1)
            for b in ids[mask]:
                block[used]=min(int(a),int(b)),max(int(a),int(b));used+=1
                if used==block_size:
                    yield block.copy();used=0
        active=np.append(active,a)
    if used:yield block[:used].copy()


def segment_weights(a,b,c,d):
    """Batched exact closest segment features (four boundaries + interior)."""
    u=b-a;v=d-c;w=a-c
    aa=np.sum(u*u,1);bb=np.sum(u*v,1);cc=np.sum(v*v,1)
    dd=np.sum(u*w,1);ee=np.sum(v*w,1);den=aa*cc-bb*bb
    safea=np.maximum(aa,1e-30);safec=np.maximum(cc,1e-30)
    ss=np.stack((np.zeros(len(a)),np.ones(len(a)),np.clip(-dd/safea,0,1),np.clip((bb-dd)/safea,0,1),
        np.divide(bb*ee-cc*dd,den,out=np.zeros_like(den),where=den>1e-12*aa*cc)),1)
    tt=np.stack((np.clip(ee/safec,0,1),np.clip((ee+bb)/safec,0,1),np.zeros(len(a)),np.ones(len(a)),
        np.divide(aa*ee-bb*dd,den,out=np.zeros_like(den),where=den>1e-12*aa*cc)),1)
    dist=np.sum((w[:,None]+ss[:,:,None]*u[:,None]-tt[:,:,None]*v[:,None])**2,2)
    valid=(den>1e-12*aa*cc)&(ss[:,4]>=0)&(ss[:,4]<=1)&(tt[:,4]>=0)&(tt[:,4]<=1)
    dist[~valid,4]=np.inf;which=np.argmin(dist,1);idx=np.arange(len(a))
    return ss[idx,which],tt[idx,which],(aa<=1e-20)|(cc<=1e-20)


class StreamingSourceContacts(SourceTriangleContacts):
    def __init__(self,frame,*,gap,floor_z,body_gap=None,block_size=4096,exit_policy=None):
        mounts={'mounts':frame.mounts}
        connections={'stitch_vertex_pairs':frame.stitches}
        super().__init__(frame.mapping,mounts,connections,gap=gap,floor_z=floor_z,body_gap=body_gap,pair_budget=100000)
        if type(block_size) is not int or not 1<=block_size<=4096:raise ValueError('Invalid bounded block size')
        self.frame=frame;self.block_size=block_size;self.source=frame.source.copy()
        self.free=np.array([frame.weights[int(frame.mapping.canonical_vertex[v])]>0 or
            (int(frame.mapping.canonical_vertex[v]) in frame.owner and np.any(frame.metrics[frame.owner[int(frame.mapping.canonical_vertex[v])]]>0)) for v in range(len(self.source))])
        self.rules={};policy=exit_policy or {'schema_version':1,'geometry_sha256':geometry_key(frame.mapping),'rules':[]}
        if set(policy)!={'schema_version','geometry_sha256','rules'} or policy['schema_version']!=1 or policy['geometry_sha256']!=geometry_key(frame.mapping):
            raise ValueError('Exit policy geometry fingerprint/schema mismatch')
        source_to_physical={int(f):i for i,f in enumerate(self.source_face_ids)}
        for rule in policy['rules']:
            if set(rule)!={'moving_face','target_face','side','max_depth_m','evidence'}:raise ValueError('Explicit exit fields required')
            a,b=rule['moving_face'],rule['target_face']
            if type(a) is not int or type(b) is not int or a==b or a not in source_to_physical or b not in source_to_physical:
                raise ValueError('Exit faces must be distinct physical source faces')
            key=tuple(sorted((source_to_physical[a],source_to_physical[b])))
            if key in self.rules or type(rule['side']) is not int or rule['side'] not in (-1,1):raise ValueError('Duplicate or ambiguous exit side')
            if not isinstance(rule['evidence'],str) or not rule['evidence'].strip():raise ValueError('Exit side needs authored evidence')
            depth=rule['max_depth_m']
            if type(depth) not in (int,float) or not np.isfinite(depth) or not 0<depth<=.002:raise ValueError('Exit depth must be explicit and no more than 2mm')
            move=self.faces[source_to_physical[a]];target=self.faces[source_to_physical[b]]
            if self.free[target].any() or not self.free[move].any():raise ValueError('Exit requires an immutable target and declared moving DOFs')
            self.rules[key]={**rule,'moving_ids':move,'target_ids':target}

    def _exit(self,a,b,x):
        ids=np.r_[self.faces[a],self.faces[b]]
        if not self.free[ids].any():
            return ('retain_immutable_baseline',[]) if np.array_equal(x[ids],self.source[ids]) else ('stop_fixed_source_changed',[])
        rule=self.rules.get((int(a),int(b)))
        if rule is None:return 'stop_missing_authored_exit_side',[]
        moving=rule['moving_ids'];target=rule['target_ids']
        if not np.array_equal(x[target],self.source[target]):return 'stop_fixed_source_changed',[]
        tri=self.source[target];normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);length=np.linalg.norm(normal)
        if length<=1e-14:return 'stop_degenerate_exit_target',[]
        normal=rule['side']*normal/length
        gap=self.body_gap if 'protected_surface' in (self.categories[a],self.categories[b]) else self.gap
        signed_distance=(x[moving]-tri[0])@normal
        # Admission measures actual penetration. Clearance is a target, not depth.
        penetration=max(0.,-float(signed_distance.min(initial=0)))
        if penetration>rule['max_depth_m']:return 'stop_authored_exit_depth_exceeded',[]
        values=signed_distance-gap
        if np.any((values < -2e-6)&~self.free[moving]):return 'stop_fixed_vertex_blocks_exit',[]
        constraints=[{'ids':[int(v)],'gradients':[normal],'offset':float(tri[0]@normal+gap),'value':float(value)}
                     for v,value in zip(moving,values) if value<0]
        return 'push_authored_side',constraints

    @staticmethod
    def _add(counts,key,amount=1):counts[key]=counts.get(key,0)+int(amount)

    def blocks(self,points):
        self.mapping.check_positions(points);x=np.asarray(points,float)
        if x.shape!=self.source.shape:raise ValueError('Source shape mismatch')
        for index,pairs in enumerate(iter_aabb_blocks(x,self.faces,max(self.gap,self.body_gap),self.block_size)):
            a,b=pairs.T;ia,ib=self.faces[a],self.faces[b];ta,tb=x[ia],x[ib]
            hit=proper_triangle_crossings(ta,tb,tolerance=1e-9)
            stats={'block':index,'pairs':len(pairs),'proper_crossings':int(hit.sum()),'crossings_by_channel':{},
                'exit_decisions':{},'maximum_deficit_m':0.,'proximity_features':0,'ambiguous_features':0,
                'bonded_features':0,'local_adjacent_pairs':0,'fixed_proximity_features':0,'examples':[]}
            constraints=[];reasons=set()
            for k in np.flatnonzero(hit):
                aa,bb=int(a[k]),int(b[k]);channel='/'.join(sorted((self.categories[aa],self.categories[bb])))
                self._add(stats['crossings_by_channel'],channel)
                decision,generated=self._exit(aa,bb,x);self._add(stats['exit_decisions'],decision)
                if decision.startswith('stop_'):reasons.add(decision)
                if generated:stats['maximum_deficit_m']=max(stats['maximum_deficit_m'],max(-v['value'] for v in generated))
                if len(stats['examples'])<24:stats['examples'].append({'source_faces':[int(self.source_face_ids[aa]),int(self.source_face_ids[bb])],'decision':decision,'channel':channel})
            adjacent=(self.parts[a]==self.parts[b])&np.any(ia[:,:,None]==ib[:,None,:],axis=(1,2))
            stats['local_adjacent_pairs']=int(adjacent.sum())
            # A geometric crossing remains in raw counts even at a local seam.
            gaps=np.where(np.array([self.categories[i]=='protected_surface' for i in a])|
                          np.array([self.categories[i]=='protected_surface' for i in b]),self.body_gap,self.gap)
            fixed=~(np.any(self.free[ia],1)|np.any(self.free[ib],1))
            if np.any(fixed & np.any(x[np.c_[ia,ib]]!=self.source[np.c_[ia,ib]],axis=(1,2))):reasons.add('stop_fixed_source_changed')
            authored=np.array([tuple(pair) in self.rules for pair in pairs])
            eligible=~adjacent
            def feature(ids,coeff,degenerate=None):
                delta=np.sum(x[ids]*coeff[:,:,None],axis=1);distance=np.linalg.norm(delta,axis=1)
                near=eligible&(distance<gaps)
                stats['proximity_features']+=int(near.sum())
                bonds=self.bond[ids];balanced=np.ones(len(ids),bool)
                for col in range(ids.shape[1]):balanced&=np.abs(np.sum(coeff*(bonds==bonds[:,col,None]),axis=1))<1e-7
                bonded=near&balanced;stats['bonded_features']+=int(bonded.sum())
                stats['fixed_proximity_features']+=int(np.sum(near&fixed&~balanced))
                active=near&~balanced&~fixed&~hit&~authored
                ambiguous=active&(distance<=1e-10)
                if degenerate is not None:ambiguous|=active&degenerate
                stats['ambiguous_features']+=int(ambiguous.sum())
                if ambiguous.any():reasons.add('stop_ambiguous_zero_distance_contact')
                selected=np.flatnonzero(active&~ambiguous)
                if len(selected):stats['maximum_deficit_m']=max(stats['maximum_deficit_m'],float((gaps[selected]-distance[selected]).max()))
                for k in selected:
                    normal=delta[k]/distance[k]
                    constraints.append({'ids':ids[k].tolist(),'gradients':coeff[k,:,None]*normal,
                                        'offset':float(gaps[k]),'value':float(distance[k]-gaps[k])})
            for ids,tri,other,othertri in ((ia,ta,ib,tb),(ib,tb,ia,ta)):
                u=othertri[:,1]-othertri[:,0];v=othertri[:,2]-othertri[:,0]
                aa=np.sum(u*u,1);bb=np.sum(u*v,1);cc=np.sum(v*v,1);den=aa*cc-bb*bb
                for col in range(3):
                    nearest=closest_triangle(tri[:,col],othertri);q=nearest-othertri[:,0]
                    du=np.sum(q*u,1);dv=np.sum(q*v,1)
                    s=np.divide(cc*du-bb*dv,den,out=np.zeros_like(den),where=den>1e-24)
                    t=np.divide(aa*dv-bb*du,den,out=np.zeros_like(den),where=den>1e-24)
                    feature(np.c_[ids[:,col],other],np.c_[np.ones(len(ids)),-(1-s-t),-s,-t],den<=1e-24)
            for ea in ((0,1),(1,2),(2,0)):
                for eb in ((0,1),(1,2),(2,0)):
                    s,t,bad=segment_weights(ta[:,ea[0]],ta[:,ea[1]],tb[:,eb[0]],tb[:,eb[1]])
                    feature(np.c_[ia[:,ea],ib[:,eb]],np.c_[1-s,s,-(1-t),-t],bad)
            stats['constraint_count']=len(constraints)
            assert len(constraints)<=18*self.block_size<=100000
            yield {'audit':stats,'constraints':constraints,'blocking_reasons':sorted(reasons),'pairs':pairs}
        # Signed exits remain active after an intersection disappears. They are
        # not proximity pairs and are applied once, even outside broad-phase AABBs.
        for pair in sorted(self.rules):
            decision,constraints=self._exit(*pair,x)
            yield {'audit':{'kind':'authored_exit','pairs':0,'constraint_count':len(constraints),
                'maximum_deficit_m':max((-c['value'] for c in constraints),default=0.)},
                'constraints':constraints,'blocking_reasons':[decision] if decision.startswith('stop_') else [],
                'pairs':np.empty((0,2),int)}
        # Floor work also bounded and settled in its own blocks, no duplicate alias points.
        canonical=np.unique(self.mapping.canonical_vertex)
        for start in range(0,len(canonical),self.block_size):
            ids=canonical[start:start+self.block_size];values=x[ids,2]-self.floor
            moved=(x[ids]!=self.source[ids]).any(1);reason=[]
            if np.any(~self.free[ids]&moved):reason=['stop_fixed_source_changed']
            take=(values<0)&self.free[ids]
            constraints=[{'ids':[int(v)],'gradients':[np.array([0.,0.,1.])],'offset':self.floor,'value':float(value)} for v,value in zip(ids[take],values[take])]
            yield {'audit':{'kind':'floor','pairs':0,'constraint_count':len(constraints),
                'maximum_deficit_m':float(np.max(-values[take],initial=0)),'fixed_floor_vertices':int(np.sum((values<0)&~self.free[ids]))},
                'constraints':constraints,'blocking_reasons':reason,'pairs':np.empty((0,2),int)}

    def _consume(self,points,frame=None,on_block=None):
        audit={'physical_triangle_count':len(self.faces),'candidate_pairs':0,'pair_blocks':0,'maximum_block_pairs':0,
            'maximum_block_constraints':0,'constraint_count':0,'proper_crossings':0,'proper_crossings_by_channel':{},
            'exit_decisions':{},'maximum_deficit_m':0.,'proximity_features':0,'ambiguous_features':0,
            'fixed_proximity_features':0,'bonded_features':0,'local_adjacent_pairs':0,'fixed_floor_vertices':0,
            'scan_complete':False,'raw_render_faces_audited':False,'continuous_collision_checked':False}
        reasons=set();examples=[]
        for block in self.blocks(points):
            s=block['audit'];reasons.update(block['blocking_reasons'])
            audit['candidate_pairs']+=s['pairs'];audit['pair_blocks']+=int(s['pairs']>0)
            audit['maximum_block_pairs']=max(audit['maximum_block_pairs'],s['pairs'])
            audit['maximum_block_constraints']=max(audit['maximum_block_constraints'],s['constraint_count'])
            audit['maximum_deficit_m']=max(audit['maximum_deficit_m'],s['maximum_deficit_m'])
            for k in ('constraint_count','proper_crossings','proximity_features','ambiguous_features','fixed_proximity_features','bonded_features','local_adjacent_pairs','fixed_floor_vertices'):
                audit[k]+=s.get(k,0)
            for source,target in (('crossings_by_channel','proper_crossings_by_channel'),('exit_decisions','exit_decisions')):
                for key,value in s.get(source,{}).items():self._add(audit[target],key,value)
            examples.extend(s.get('examples',[])[:max(0,24-len(examples))])
            if on_block:on_block(s,block['pairs'])
            if frame is not None and not reasons:
                for c in block['constraints']:
                    value=sum(float(frame.point(v)@g) for v,g in zip(c['ids'],c['gradients']))-c['offset']
                    if value<0:frame.project_scalar(c['ids'],c['gradients'],value)
        audit.update(scan_complete=True,crossing_examples=examples)
        return {'audit':audit,'constraints':[],'blocking_reasons':sorted(reasons),
                'blocking_reason':';'.join(sorted(reasons)) if reasons else None}

    def scan(self,points,on_block=None):return self._consume(np.array(points,copy=True),on_block=on_block)

    def project_all(self,frame):
        if frame is not self.frame:raise ValueError('Contact/frame identity mismatch')
        import copy
        saved=(frame.x.copy(),copy.deepcopy(frame.states))
        try:
            result=self._consume(frame.positions(),frame=frame)
            if result['blocking_reason']:frame.x,frame.states=saved
            return result
        except BaseException:
            frame.x,frame.states=saved;raise
