"""Bounded discrete joint projection on explicitly oriented triangle pairs.

Engineering kernel, not a production/continuous cloth simulator. It never
infers inside/outside from nearest neighbours. An asset adapter must supply
reviewed contact sides. Unsupported mappings and an exhausted budget fail
closed. No source animation, weights or shape keys are writable here.
"""
import copy
import itertools
import math
import numpy as np
from .secondary_attachment_driver import AttachmentSystem, exp_rotation, triangle_frame
from .triangle_contacts import closest_triangle, proper_triangle_crossings


def joint_solver_status():
    return {'implementation':'bounded_discrete_joint_projection_v1',
            'discrete_core_implemented':True,'production_ready':False,
            'continuous_collision_implemented':False,'self_collision_implemented':False,
            'native_model_binding_accepted':False,'rollback_on_failure':True}


def _positive(value, name, zero=False):
    if type(value) not in (int,float) or not math.isfinite(value) or (value<0 if zero else value<=0):
        raise ValueError(name+' must be finite and '+('nonnegative' if zero else 'positive'))
    return float(value)


def segment_closest(a,b,c,d):
    """Exact finite segment closest features, including parallel edges."""
    u=b-a;v=d-c;w=a-c;aa=float(u@u);bb=float(u@v);cc=float(v@v)
    if min(aa,cc)<1e-20:raise ValueError('Degenerate triangle edge')
    dd=float(u@w);ee=float(v@w);den=aa*cc-bb*bb
    candidates=[(0.,float(np.clip(ee/cc,0,1))),
                (1.,float(np.clip((ee+bb)/cc,0,1))),
                (float(np.clip(-dd/aa,0,1)),0.),
                (float(np.clip((bb-dd)/aa,0,1)),1.)]
    if den>1e-12*aa*cc:
        s=(bb*ee-cc*dd)/den;t=(aa*ee-bb*dd)/den
        if 0<=s<=1 and 0<=t<=1:candidates.append((s,t))
    s,t=min(candidates,key=lambda st:np.linalg.norm(w+st[0]*u-st[1]*v))
    return s,t,a+s*u,c+t*v


class JointContactSystem:
    """Explicit part-pair sides: A relative to B normal and B relative to A.

    Both parts react according to their actual free DOFs. All distinct physical
    part pairs must be declared, even fixed/fixed pairs. Same-part contact,
    aliases and seam exceptions are deliberately unsupported, never skipped
    under a claim of complete collision coverage.
    """
    def __init__(self,plan,review,bindings,rules,*,floor_z=0.,floor_gap=0.,max_feature_tests=20000):
        self.attachment=AttachmentSystem(plan,review,bindings)
        if bindings['aliases'] or bindings['seams'] or any(p['category']=='display_overlay' for p in review['parts']):
            raise ValueError('Native alias/seam mapping is not implemented by this discrete kernel')
        if type(floor_z) not in (int,float) or not math.isfinite(floor_z):raise ValueError('Invalid floor')
        self.floor=float(floor_z)+_positive(floor_gap,'floor gap',True)
        if type(max_feature_tests) is not int or not 1<=max_feature_tests<=200000:raise ValueError('Invalid feature budget')
        parts={p['id']:p for p in review['parts']};expected={tuple(sorted(x)) for x in itertools.combinations(parts,2)}
        found=set();self.pairs=[]
        for row in rules:
            if set(row)!={'part_a','part_b','side_a','side_b','clearance_m'}:raise ValueError('Explicit sides and clearance required')
            a,b=row['part_a'],row['part_b'];key=tuple(sorted((a,b)))
            if key not in expected or key in found:raise ValueError('Unknown, same-part or duplicate contact pair')
            if any(type(row[k]) is not int or row[k] not in (-1,1) for k in ('side_a','side_b')):raise ValueError('Contact sides must be -1 or 1')
            gap=_positive(row['clearance_m'],'clearance');found.add(key)
            # Bound expansion before allocating quadratic work.
            count=len(parts[a]['face_ids'])*len(parts[b]['face_ids'])
            if (len(self.pairs)+count)*15>max_feature_tests:raise ValueError('Feature budget exceeded; no silent sampling')
            for fa in parts[a]['face_ids']:
                for fb in parts[b]['face_ids']:
                    ia=self.attachment.faces[fa];ib=self.attachment.faces[fb]
                    if set(ia)&set(ib):raise ValueError('Shared contact vertices need an explicit native seam mapping')
                    self.pairs.append({'a':int(fa),'b':int(fb),'sa':row['side_a'],'sb':row['side_b'],
                                       'gap':gap,'channel':'/'.join(sorted((parts[a]['category'],parts[b]['category'])))})
        if found!=expected:raise ValueError('Missing physical part pairs; cannot omit a collision channel')
        self.edges=set()
        for p in parts.values():
            if p['category'] not in ('inner_cloth','outer_cloth','hair','flexible_trim'):continue
            for face in p['face_ids']:
                tri=self.attachment.faces[face]
                self.edges.update(tuple(sorted((int(tri[a]),int(tri[b])))) for a,b in ((0,1),(1,2),(2,0)))
        self.edges=sorted(self.edges)

    def begin_frame(self,*args,**kwargs):return self.attachment.begin_frame(*args,**kwargs)

    def _features(self,frame):
        """Rebuild features after every sweep; actual triangles, no proxies."""
        x=frame.positions();faces=self.attachment.faces;out=[]
        for pair in self.pairs:
            ia,ib=faces[pair['a']],faces[pair['b']];a,b=x[ia],x[ib]
            for ids,points,target_ids,target,side in ((ia,a,ib,b,pair['sa']),(ib,b,ia,a,pair['sb'])):
                n=triangle_frame(target)[:,2];axes=np.column_stack((target[1]-target[0],target[2]-target[0]))
                for vertex,p in zip(ids,points):
                    uv=np.linalg.lstsq(axes,p-target[0],rcond=None)[0]
                    if min(uv)>=-1e-9 and sum(uv)<=1+1e-9:
                        out.append({'kind':'vf','ids':(int(vertex),*map(int,target_ids)),
                                    'side':side,'gap':pair['gap'],'channel':pair['channel']})
            # Edge/edge proximity uses its actual nearest features. Exact
            # intersection has no unique normal and is caught by final audit.
            for ea in ((0,1),(1,2),(2,0)):
                for eb in ((0,1),(1,2),(2,0)):
                    s,t,p,q=segment_closest(*a[list(ea)],*b[list(eb)])
                    delta=p-q;distance=float(np.linalg.norm(delta))
                    # Endpoints are essential: after a small rotation the
                    # nearest feature may leave a face interior and become
                    # vertex/edge or vertex/vertex. Dropping endpoints leaves
                    # a clearance hole although signed face tests look clear.
                    if 1e-10<distance<pair['gap']:
                        out.append({'kind':'ee','ids':tuple(map(int,(*ia[list(ea)],*ib[list(eb)]))),
                                    's':s,'t':t,'normal':delta/distance,'gap':pair['gap'],'channel':pair['channel']})
        out.extend({'kind':'floor','ids':(v,),'gap':self.floor,'channel':'ground'} for v in range(len(x)))
        return out

    @staticmethod
    def _value(frame,c):
        p=np.array([frame.point(v) for v in c['ids']]);kind=c['kind']
        if kind=='floor':return float(p[0,2]-c['gap'])
        if kind=='edge':return float(np.linalg.norm(p[1]-p[0])-c['length'])
        if kind=='vf':return float(c['side']*((p[0]-p[1])@triangle_frame(p[1:])[:,2])-c['gap'])
        return float(((1-c['s'])*p[0]+c['s']*p[1]-(1-c['t'])*p[2]-c['t']*p[3])@c['normal']-c['gap'])

    def _project(self,frame,c,eps=1e-6):
        value=self._value(frame,c)
        if c['kind']!='edge' and value>=0:return
        ids=c['ids'];keys=frame._dofs({'child':ids[0],'target':ids[1:]});blocks=[]
        for key in keys:
            kind,idx=key;count=3 if kind=='vertex' else 6;j=np.zeros(count)
            saved=frame.x[idx].copy() if kind=='vertex' else copy.deepcopy(frame.rigid[idx])
            try:
                for axis in range(count):
                    values=[]
                    for sign in (1,-1):
                        if kind=='vertex':
                            frame.x[idx]=saved.copy();frame.x[idx,axis]+=sign*eps
                        else:
                            frame.rigid[idx]=copy.deepcopy(saved)
                            if axis<3:frame.rigid[idx]['center'][axis]+=sign*eps
                            else:
                                vector=np.zeros(3);vector[axis-3]=sign*eps
                                frame.rigid[idx]['rotation']=exp_rotation(vector)@saved['rotation']
                        values.append(self._value(frame,c))
                    j[axis]=(values[0]-values[1])/(2*eps)
            finally:
                if kind=='vertex':frame.x[idx]=saved
                else:frame.rigid[idx]=saved
            blocks.append((key,j,frame._weight(key)))
        effective=sum(float(j@w@j) for _,j,w in blocks)
        if effective<=1e-14:return
        updates=[(key,-value*(w@j)/effective) for key,j,w in blocks]
        turn=max((float(np.linalg.norm(d[3:])) for key,d in updates if key[0]=='rigid'),default=0.)
        scale=min(1.,.25/turn) if turn else 1.
        for (kind,idx),delta in updates:
            delta=delta*scale
            if kind=='vertex':frame.x[idx]+=delta
            else:
                frame.rigid[idx]['center']+=delta[:3]
                frame.rigid[idx]['rotation']=exp_rotation(delta[3:])@frame.rigid[idx]['rotation']
        frame.enforce_limits()

    def audit(self,frame,edge_constraints):
        features=self._features(frame);x=frame.positions();faces=self.attachment.faces
        deficits={}
        for c in features:deficits[c['channel']]=max(deficits.get(c['channel'],0.),-self._value(frame,c),0.)
        crossings={};clearances={}
        for pair in self.pairs:
            a,b=x[faces[pair['a']]],x[faces[pair['b']]];channel=pair['channel']
            crosses=bool(proper_triangle_crossings(a[None],b[None],tolerance=1e-9)[0])
            crossings[channel]=crossings.get(channel,0)+int(crosses)
            distances=[float(np.linalg.norm(p-q)) for p,q in zip(a,closest_triangle(a,np.repeat(b[None],3,axis=0)))]
            distances.extend(float(np.linalg.norm(p-q)) for p,q in zip(b,closest_triangle(b,np.repeat(a[None],3,axis=0))))
            for ea in ((0,1),(1,2),(2,0)):
                for eb in ((0,1),(1,2),(2,0)):
                    _,_,p,q=segment_closest(*a[list(ea)],*b[list(eb)]);distances.append(float(np.linalg.norm(p-q)))
            deficit=max(0.,pair['gap']-min(distances))
            clearances[channel]=max(clearances.get(channel,0.),deficit)
        attachment=max((float(np.linalg.norm(frame.residual(c))) for c in frame.system.constraints),default=0.)
        edge=max((abs(self._value(frame,c)) for c in edge_constraints),default=0.)
        return {'attachment_residual_m':attachment,'edge_length_error_m':edge,
                'maximum_contact_deficit_m':max([*deficits.values(),*clearances.values()],default=0.),
                'signed_deficits_m':deficits,'clearance_deficits_m':clearances,
                'proper_crossings_by_channel':crossings,'proper_crossings':sum(crossings.values()),
                'protected_vertices_unchanged':bool(np.array_equal(frame.x[frame.system.fixed],frame.source[frame.system.fixed]))}

    def solve(self,frame,*,iterations=96,tolerance_m=1e-5):
        if frame.system is not self.attachment:raise ValueError('Frame belongs to a different attachment system')
        if type(iterations) is not int or not 1<=iterations<=256:raise ValueError('Invalid joint iteration budget')
        tol=_positive(tolerance_m,'tolerance')
        if not np.isfinite(frame.x).all() or not np.array_equal(frame.x[frame.system.fixed],frame.source[frame.system.fixed]):
            raise ValueError('Predictor modified fixed source vertices or contains nonfinite values')
        original=(frame.x.copy(),copy.deepcopy(frame.rigid));used=0
        edges=[{'kind':'edge','ids':(a,b),'length':float(np.linalg.norm(frame.source[a]-frame.source[b]))} for a,b in self.edges]
        try:
            frame.enforce_limits()
            initial=self.audit(frame,edges);report=initial
            def accepted(r):
                return r['protected_vertices_unchanged'] and r['proper_crossings']==0 and max(r['attachment_residual_m'],r['edge_length_error_m'],r['maximum_contact_deficit_m'])<=tol
            for step in range(iterations):
                if accepted(report):break
                used=step+1
                frame.project_attachments(iterations=1,tolerance_m=tol*.25)
                for c in edges:self._project(frame,c)
                for c in self._features(frame):self._project(frame,c)
                # The last contact/ground projection is NOT accepted on trust:
                # recheck attachment, lengths and all pairs on this exact state.
                report=self.audit(frame,edges)
            converged=accepted(report)
            if not np.isfinite(frame.positions()).all():raise ValueError('Nonfinite joint candidate')
            if not converged:frame.x[:]=original[0];frame.rigid=original[1]
            return {'discrete_constraints_converged':converged,'candidate_committed':converged,
                    'rolled_back':not converged,'iterations_used':used,'initial_prediction_audit':initial,
                    'candidate_audit':report,'physics_accepted':False,'continuous_collision_checked':False,
                    'self_collision_checked':False,'original_before_quality_gate_checked':False,
                    'scope':'explicitly_oriented_distinct_part_pairs_and_floor',
                    'reason':None if converged else 'joint_constraints_conflict_or_budget_exhausted'}
        except Exception:
            frame.x[:]=original[0];frame.rigid=original[1]
            raise
