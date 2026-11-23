"""Model surface contact priorities, attachment transport and rigid floor bounds.

No root, bone, skin or shape-key writes. Ground constraints affect configured
garment vertices only. A plane bound is not full body/cloth triangle CCD.
"""
import numpy as np
from .triangle_contacts import closest_triangle

def prewarm_convergence(speeds,p95_limit=.02,max_limit=.1):
 """Require the entire trailing held-pose interval to have settled.

A single quiet final frame is insufficient. Empty/nonfinite measurements cannot
approve a bake. Thresholds are model-owned speeds in meters per second.
"""
 s=np.asarray(speeds,float)
 if s.ndim!=2 or not s.size or not np.isfinite(s).all() or np.any(s<0):raise ValueError('Finite nonempty speed history required')
 if not 0<p95_limit<=max_limit or not np.isfinite([p95_limit,max_limit]).all():raise ValueError('Invalid convergence speed limits')
 p95=float(np.percentile(s,95,axis=1).max());maximum=float(s.max())
 return {'passed':bool(p95<=p95_limit and maximum<=max_limit),'trailing_frames':len(s),
  'maximum_frame_p95_m_s':p95,'maximum_vertex_speed_m_s':maximum,'p95_limit_m_s':p95_limit,'max_limit_m_s':max_limit}

def garment_edge_deformation(reference,candidate,edges,relative_limit=.25,absolute_tolerance=5e-6):
 """Measure extra edge-length change, invariant to whole-panel rigid motion.

Pose skinning is the reference, so this tests added deformation, not the source
model's pre-existing skinning error. Micro edges use a 1mm denominator.
"""
 a=np.asarray(reference,float);b=np.asarray(candidate,float);e=np.asarray(edges,int)
 if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=3 or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Invalid garment geometry')
 if not 0<relative_limit<1 or absolute_tolerance<0:raise ValueError('Invalid deformation limits')
 before=np.linalg.norm(a[e[:,1]]-a[e[:,0]],axis=1);after=np.linalg.norm(b[e[:,1]]-b[e[:,0]],axis=1)
 error=np.maximum(np.abs(after-before)-absolute_tolerance,0)/np.maximum(before,.001)
 return {'passed':bool(np.all(error<=relative_limit)),'maximum_additional_relative_edge_error':float(error.max(initial=0)),
  'edges_over_limit':int(np.sum(error>relative_limit)),'relative_limit':relative_limit,'absolute_tolerance_m':absolute_tolerance}

def simulation_quality_gate(surface_gate,cloth_report,surface_report=None):
 """Keep geometric regression and actual warmup approval separate and strict."""
 gate={**surface_gate,'rejection_reasons':list(surface_gate['rejection_reasons'])}
 warm=cloth_report.get('prewarm_convergence',{})
 if warm.get('passed') is not True:gate['rejection_reasons'].append('cloth_prewarm_unsettled_or_unverified')
 deformation=(surface_report or {}).get('garment_deformation_gate',{})
 if deformation.get('passed') is not True:gate['rejection_reasons'].append('garment_deformation_excessive_or_unverified')
 gate['accepted']=not gate['rejection_reasons'];gate['prewarm_convergence']=warm
 gate['garment_deformation_gate']=deformation
 return gate

def mesh_edges(faces):
 f=np.asarray(faces,int)
 if f.ndim!=2 or f.shape[1]!=3:raise ValueError('Triangle faces required')
 e=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);e=np.unique(np.sort(e,axis=1),axis=0)
 return e[e[:,0]!=e[:,1]]

def contact_pin_release(pins,faces,unsafe,protected=None,rings=4):
 """Release incompatible pose pins consistently for the whole trajectory.

Protected body vertices stay fixed. The contact region and its graph collar
lose pose attraction; native cloth edges still maintain material continuity.
"""
 pins=np.asarray(pins,float);unsafe=np.asarray(unsafe,bool);protected=np.zeros(len(pins),bool) if protected is None else np.asarray(protected,bool)
 if pins.ndim!=1 or unsafe.shape!=pins.shape or protected.shape!=pins.shape or not np.isfinite(pins).all() or np.any((pins<0)|(pins>1)):raise ValueError('Invalid pin/contact arrays')
 if type(rings) is not int or rings<1:raise ValueError('Positive contact collar required')
 edge=mesh_edges(faces);depth=np.full(len(pins),rings+1,int);depth[unsafe&~protected]=0
 for step in range(1,rings+1):
  reached=(depth[edge[:,0]]<step)|(depth[edge[:,1]]<step)
  vertices=np.unique(edge[reached]);depth[vertices]=np.minimum(depth[vertices],step);depth[protected]=rings+1
 scale=np.minimum(depth/rings,1.);result=pins*scale;result[protected]=pins[protected]
 return result,depth

def contact_active_faces(pins,faces,unsafe,protected=None,excluded=None,rings=4):
 """Keep normal movable panels plus a collar around conflicting pinned cloth.

Fully pinned torso panels outside the contact region remain kinematic. Selection
is constant over the entire trajectory, so panels cannot switch topology mid-run.
"""
 pins=np.asarray(pins,float);faces=np.asarray(faces,int)
 protected=np.zeros(len(pins),bool) if protected is None else np.asarray(protected,bool)
 _,depth=contact_pin_release(pins,faces,unsafe,protected,rings)
 active=np.any(pins[faces]<.999999,axis=1)|np.any(depth[faces]<rings,axis=1)
 active&=~np.all(protected[faces],axis=1)
 if excluded is not None:active&=~np.asarray(excluded,bool)
 return active

class FloorContactGraph:
 def __init__(self,rest,faces,protected=None,weld_keys=None):
  raw=np.asarray(rest,float)
  if raw.ndim!=2 or raw.shape[1]!=3 or not np.isfinite(raw).all():raise ValueError('Finite rest vertices required')
  self.source_count=len(raw);self.source_protected=np.zeros(len(raw),bool) if protected is None else np.asarray(protected,bool)
  if self.source_protected.shape!=(len(raw),):raise ValueError('Invalid body protection mask')
  keys=weld_keys if weld_keys is not None else list(range(len(raw)))
  if len(keys)!=len(raw):raise ValueError('Invalid weld signatures')
  table={};representatives=[];inverse=[]
  for i,point in enumerate(raw):
   key=(tuple(np.round(point,6)),keys[i])
   if key not in table:table[key]=len(representatives);representatives.append(i)
   inverse.append(table[key])
  self.inverse=np.asarray(inverse,int);self.rest=raw[representatives]
  self.edges=mesh_edges(self.inverse[np.asarray(faces,int)])
  # Nonincident edges may be projected together without averaging away short
  # edge corrections at high-valence vertices. Deterministic graph coloring
  # gives Gauss-Seidel convergence with vectorized, conflict-free batches.
  used=[set() for _ in self.rest];colors=[]
  for a,b in self.edges:
   occupied=used[a]|used[b];color=0
   while color in occupied:color+=1
   used[a].add(color);used[b].add(color);colors.append(color)
  self.edge_colors=np.asarray(colors,int)
  self.protected=np.zeros(len(self.rest),bool);np.logical_or.at(self.protected,self.inverse,self.source_protected)

 def solve(self,positions,floor,margin=.002,iterations=48,strain=.05,reference_positions=None):
  raw=np.asarray(positions);level=float(floor)+float(margin)
  if raw.shape!=(self.source_count,3) or not np.isfinite(raw).all() or not np.isfinite([level,strain]).all() or margin<0 or not 0<=strain<1 or type(iterations) is not int or iterations<0:raise ValueError('Invalid surface floor solve')
  reference_raw=raw if reference_positions is None else np.asarray(reference_positions)
  if reference_raw.shape!=raw.shape or not np.isfinite(reference_raw).all():raise ValueError('Invalid source edge reference')
  if reference_positions is None and not np.any((raw[:,2]<level)&~self.source_protected):return raw.copy(),{'contact_vertices':0,'maximum_shift_m':0.,'edge_relative_error':0.}
  p=np.zeros_like(self.rest);np.add.at(p,self.inverse,raw);p/=np.bincount(self.inverse)[:,None]
  reference_p=np.zeros_like(self.rest);np.add.at(reference_p,self.inverse,reference_raw);reference_p/=np.bincount(self.inverse)[:,None]
  for i in np.flatnonzero(self.source_protected):p[self.inverse[i]]=raw[i]
  original=p.copy()
  contact=(p[:,2]<level)&~self.protected
  e=self.edges;active=contact.copy()
  if reference_positions is not None:
   reference_length=np.linalg.norm(reference_p[e[:,1]]-reference_p[e[:,0]],axis=1)
   current_length=np.linalg.norm(p[e[:,1]]-p[e[:,0]],axis=1)
   broken=np.abs(current_length-reference_length)>reference_length*strain+1e-6
   active[np.unique(e[broken])]=True
  if not active.any():return raw.copy(),{'contact_vertices':0,'maximum_shift_m':0.,'edge_relative_error':0.}
  for _ in range(6):
   touch=active[e[:,0]]|active[e[:,1]];active[np.unique(e[touch])]=True
  active&=~self.protected
  edge_active=active[e[:,0]]|active[e[:,1]];colors=self.edge_colors[edge_active];e=e[edge_active]
  reference=reference_p[e[:,1]]-reference_p[e[:,0]];length=np.linalg.norm(reference,axis=1)
  hint=self.rest[e[:,1]]-self.rest[e[:,0]]
  # Stable tangential direction for a vertically collapsed contact edge.
  fallback=reference.copy();fallback[:,2]=0
  absent=np.linalg.norm(fallback,axis=1)<1e-10;fallback[absent]=hint[absent];fallback[:,2]=0
  absent=np.linalg.norm(fallback,axis=1)<1e-10;fallback[absent]=[1.,0,0]
  fallback/=np.maximum(np.linalg.norm(fallback,axis=1,keepdims=True),1e-20)
  mobility=active.astype(float);p[contact,2]=level
  batches=[np.flatnonzero(colors==color) for color in np.unique(colors)]
  for _ in range(iterations):
   if reference_positions is None:
    # Keep native driver/floor behavior unchanged. The colored refinement is
    # for explicit source-edge preservation during final surface assembly.
    delta=p[e[:,1]]-p[e[:,0]];distance=np.linalg.norm(delta,axis=1)
    direction=delta/np.maximum(distance[:,None],1e-20);zero=distance<1e-10;direction[zero]=fallback[zero]
    target=np.clip(distance,length*(1-strain),length*(1+strain));error=distance-target
    denominator=mobility[e[:,0]]+mobility[e[:,1]]
    correction=direction*np.divide(error,denominator,out=np.zeros(len(e)),where=denominator>0)[:,None]
    change=np.zeros_like(p);weight=np.zeros(len(p))
    for vertex,sign in [(e[:,0],1),(e[:,1],-1)]:
     np.add.at(change,vertex,sign*correction*mobility[vertex,None]);np.add.at(weight,vertex,(np.abs(error)>1e-9)*mobility[vertex])
    p+=np.divide(change,np.maximum(weight[:,None],1))*.8
    p[~self.protected,2]=np.maximum(p[~self.protected,2],level)
    p[self.protected]=original[self.protected]
    if np.max(np.abs(error),initial=0)<1e-6:break
    continue
   for batch in batches:
    a,b=e[batch].T;delta=p[b]-p[a];distance=np.linalg.norm(delta,axis=1)
    direction=delta/np.maximum(distance[:,None],1e-20);zero=distance<1e-10;direction[zero]=fallback[batch[zero]]
    target=np.clip(distance,length[batch]*(1-strain),length[batch]*(1+strain));error=distance-target
    denominator=mobility[a]+mobility[b]
    correction=direction*np.divide(error,denominator,out=np.zeros(len(batch)),where=denominator>0)[:,None]
    p[a]+=correction*mobility[a,None];p[b]-=correction*mobility[b,None]
    p[active,2]=np.maximum(p[active,2],level)
   p[self.protected]=original[self.protected]
   distance=np.linalg.norm(p[e[:,1]]-p[e[:,0]],axis=1)
   residual=np.abs(distance-np.clip(distance,length*(1-strain),length*(1+strain)))
   if np.max(residual,initial=0)<1e-6:break
  p[~self.protected,2]=np.maximum(p[~self.protected,2],level)
  final=np.linalg.norm(p[e[:,1]]-p[e[:,0]],axis=1)
  output=p[self.inverse].astype(raw.dtype);output[self.source_protected]=raw[self.source_protected]
  return output,{'contact_vertices':int(contact.sum()),'maximum_shift_m':float(np.linalg.norm(output-raw,axis=1).max(initial=0)),
    'edge_relative_error':float(np.max(np.abs(final-length)/np.maximum(length,1e-6),initial=0))}

def triangle_frames(triangles):
 t=np.asarray(triangles,float);x=t[:,1]-t[:,0];x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-20)
 z=np.cross(x,t[:,2]-t[:,0]);z/=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-20);y=np.cross(z,x)
 return np.stack([x,y,z],axis=2)

def regularized_attachment_fit(source,target,rotation_prior=1.):
 """Rigid patch transport with the source pose as an orientation prior.

 Tiny or nearly collinear garment patches cannot determine a reliable twist.
 An unconstrained Kabsch fit can turn micron-scale deformation into a 180 deg
 accessory flip. Minimize patch residual plus a scale-normalized SO(3) prior;
 the original posed orientation is retained as contact information vanishes.
 Translation follows the final patch exactly. This is kinematic, not 6-DOF.
 """
 a=np.asarray(source,float);b=np.asarray(target,float)
 if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=3 or len(a)<3 or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Finite matching attachment patches required')
 if not np.isfinite(rotation_prior) or not 0<rotation_prior<=10:raise ValueError('Invalid attachment rotation prior')
 center=a.mean(0);goal=b.mean(0);x=a-center;y=b-goal;energy=float(np.sum(x*x))
 if energy<1e-24:return np.eye(3),goal-center
 covariance=x.T@y+rotation_prior*energy*np.eye(3)
 u,_,vt=np.linalg.svd(covariance);rotation=vt.T@u.T
 if np.linalg.det(rotation)<0:vt[-1]*=-1;rotation=vt.T@u.T
 return rotation,goal-rotation@center

def bind_surface_points(points,rest,faces,point_skins=None,vertex_skins=None,max_distance=None):
 """Closest authored triangle with barycentric coordinates; no 3cm dropout."""
 points=np.asarray(points,float);rest=np.asarray(rest,float);faces=np.asarray(faces,int);triangles=rest[faces]
 if not len(faces) or points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('Invalid attachment geometry')
 valid=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)>1e-12
 original_ids=np.flatnonzero(valid);triangles=triangles[valid];selected_faces=faces[valid]
 if not len(triangles):raise ValueError('No nondegenerate attachment triangles')
 face_skin={}
 if point_skins is not None and vertex_skins is not None:
  bones={b for skin in point_skins for b in skin}
  face_skin={b:np.asarray([max(float(vertex_skins[int(v)].get(b,0)) for v in f) for f in selected_faces]) for b in bones}
 index=[];bary=[];distances=[];matched=[]
 for i,point in enumerate(points):
  query=np.broadcast_to(point,(len(triangles),3));near=closest_triangle(query,triangles);distance=np.linalg.norm(near-point,axis=1)
  score=distance.copy()
  if point_skins is not None and vertex_skins is not None:
   skin=point_skins[i];overlap=sum(np.minimum(float(w),face_skin[b]) for b,w in skin.items())
   # Bone agreement breaks local ambiguity; never jump across a whole outfit.
   local=distance<=distance.min()+.015
   score+=.01*(1-overlap);score[~local]=np.inf
  selected=int(np.argmin(score));a,b,c=triangles[selected];ab=b-a;ac=c-a;q=near[selected]-a
  gram=np.array([[ab@ab,ab@ac],[ab@ac,ac@ac]])
  uv=np.linalg.lstsq(gram,np.array([q@ab,q@ac]),rcond=None)[0];weights=np.array([1-uv.sum(),uv[0],uv[1]])
  weights=np.maximum(weights,0);weights/=max(weights.sum(),1e-20)
  index.append(int(original_ids[selected]));bary.append(weights);distances.append(float(distance[selected]));matched.append(max_distance is None or distance[selected]<=max_distance)
 return {'triangles':np.asarray(index,int),'barycentric':np.asarray(bary,float),'distance_m':np.asarray(distances),'bound':np.asarray(matched,bool)}

def transport_surface_points(binding,points,source_cloth,target_cloth,faces):
 points=np.asarray(points);source=np.asarray(source_cloth,float)[np.asarray(faces,int)[binding['triangles']]]
 target=np.asarray(target_cloth,float)[np.asarray(faces,int)[binding['triangles']]]
 weight=binding['barycentric'];a=np.sum(source*weight[:,:,None],axis=1);b=np.sum(target*weight[:,:,None],axis=1)
 source_frames=triangle_frames(source);target_frames=triangle_frames(target);rotation=target_frames@source_frames.transpose(0,2,1)
 result=b+np.einsum('nij,nj->ni',rotation,np.asarray(points,float)-a)
 degenerate=(np.linalg.norm(np.cross(source[:,1]-source[:,0],source[:,2]-source[:,0]),axis=1)<1e-10)|(np.linalg.norm(np.cross(target[:,1]-target[:,0],target[:,2]-target[:,0]),axis=1)<1e-10)
 result[degenerate]=np.asarray(points,float)[degenerate]+b[degenerate]-a[degenerate]
 unchanged=np.max(np.abs(target-source),axis=(1,2))<1e-12
 result[unchanged|~binding['bound']]=points[unchanged|~binding['bound']]
 return result.astype(points.dtype)

def rigid_floor_envelope(local_points,rotations,centers,floor,margin=.001):
 """Conservative plane clearance for every quaternion interpolation interval.

Each center is linear between keys. Rotated vertices stay within the endpoint
chord plus the rotation-arc sagitta; interval minima of center and local Z give
a safe lower bound. No rigid scale or protected-body translation is used.
"""
 local=np.asarray(local_points,float);r=np.asarray(rotations,float);c=np.asarray(centers,float).copy()
 if r.shape!=(len(c),3,3) or c.shape[1:]!=(3,) or not np.isfinite([float(floor),float(margin)]).all() or margin<0:raise ValueError('Invalid rigid trajectory')
 if not np.isfinite(local).all() or not np.isfinite(r).all() or not np.isfinite(c).all():raise ValueError('Nonfinite rigid trajectory')
 if not np.allclose(r.transpose(0,2,1)@r,np.eye(3),atol=1e-5) or np.any(np.linalg.det(r)<0):raise ValueError('Rigid rotations must be proper orthogonal matrices')
 z=np.einsum('tij,nj->tni',r,local)[:,:,2];lift=np.maximum(float(floor)+margin-(c[:,2]+z.min(axis=1)),0)
 if len(c)>1:
  relative=r[1:]@r[:-1].transpose(0,2,1);theta=np.arccos(np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1))
  radius=float(np.linalg.norm(local,axis=1).max(initial=0));sag=radius*(1-np.cos(theta/2))
  lower=np.minimum(c[:-1,2],c[1:,2])+np.minimum(z[:-1],z[1:]).min(axis=1)-sag
  interval=np.maximum(float(floor)+margin-lower,0)
  lift[:-1]=np.maximum(lift[:-1],interval);lift[1:]=np.maximum(lift[1:],interval)
 c[:,2]+=lift
 return c,lift
