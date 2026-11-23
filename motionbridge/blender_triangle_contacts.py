"""Blender nearest-triangle backend for configured corrective garment contact."""
import numpy as np


class TriangleSurface:
    def __init__(self, points, faces):
        from mathutils.bvhtree import BVHTree
        self.points=np.asarray(points,float);self.faces=np.asarray(faces,int)
        if (self.points.ndim!=2 or self.points.shape[1]!=3 or self.faces.ndim!=2 or self.faces.shape[1]!=3
            or not len(self.faces) or self.faces.min()<0 or self.faces.max()>=len(self.points)
            or not np.isfinite(self.points).all()):
            raise ValueError('Invalid triangle collider')
        self.triangles=self.points[self.faces]
        self.bounds=(self.triangles.reshape(-1,3).min(0),self.triangles.reshape(-1,3).max(0))
        used=np.unique(self.faces)
        self.tree=BVHTree.FromPolygons(self.points[used].tolist(),np.searchsorted(used,self.faces).tolist(),all_triangles=True)

    def nearest(self, points, radius):
        from mathutils import Vector
        query=np.zeros_like(points);face=np.full(len(points),-1,int)
        lo=self.bounds[0]-radius;hi=self.bounds[1]+radius
        for i in np.flatnonzero(((points>=lo)&(points<=hi)).all(1)):
            p=points[i]
            q,_,f,_=self.tree.find_nearest(Vector(p),radius)
            if q is not None:query[i]=q;face[i]=f
        return query,face

    def crossings(self,other):
        from .triangle_contacts import proper_triangle_crossings
        pairs=np.asarray(self.tree.overlap(other.tree),int)
        if not len(pairs):return set()
        valid=proper_triangle_crossings(self.triangles[pairs[:,0]],other.triangles[pairs[:,1]])
        return set(map(tuple,pairs[valid].tolist()))


def configured_body_surface(scene):
    """Explicit Nahida material profile; never include shared contour copies."""
    obj=scene.objects.get('Nahida_mesh')
    materials={'颜','前髮','头饰','体','短裤','体饰','肌2'}
    if obj is None or not np.allclose(obj.matrix_world,np.eye(4)):
        raise ValueError('Missing configured identity-transform collision mesh')
    obj.data.calc_loop_triangles()
    faces=np.array([p.vertices[:] for p in obj.data.loop_triangles if obj.data.materials[p.material_index].name in materials],int)
    if not len(faces):raise ValueError('Missing configured collision materials')
    rest=np.array([v.co[:] for v in obj.data.vertices])
    return obj,rest,faces


def evaluated_points(obj):
    import bpy
    ev=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh()
    try:
        if len(mesh.vertices)!=len(obj.data.vertices):raise ValueError('Collider topology changed')
        points=np.empty(len(mesh.vertices)*3);mesh.vertices.foreach_get('co',points)
        return points.reshape(-1,3)
    finally:ev.to_mesh_clear()


class MeshContactProfile:
    """Reusable rest geometry and layer order; no clip identities or frames."""
    def __init__(self,rest,edges,faces,body_rest,body_faces):
        from .triangle_contacts import contact_samples,sample_points
        from .surface_directed import connected_surfaces
        self.rest=np.asarray(rest);self.edges=np.asarray(edges);self.faces=np.asarray(faces)
        self.body_rest=np.asarray(body_rest);self.body_faces=np.asarray(body_faces)
        self.ids,self.weights,self.groups=contact_samples(rest,edges,faces)
        self.samples=sample_points(rest,self.ids,self.weights)
        self.labels=connected_surfaces(rest,edges);self.sample_labels=self.labels[self.ids[:,0]]
        self.components=np.unique(self.labels)
        self.component_faces={k:self.faces[self.labels[self.faces[:,0]]==k] for k in self.components}
        self.rest_trees={k:TriangleSurface(rest,self.component_faces[k]) for k in self.components}
        body_tree=TriangleSurface(body_rest,body_faces)
        self.rest_body_pairs=body_tree.crossings(TriangleSurface(rest,faces))
        self.rest_layer_pairs={}
        for i,k in enumerate(self.components):
            for j in self.components[i+1:]:
                self.rest_layer_pairs[k,j]=self.rest_trees[k].crossings(self.rest_trees[j])

    def force(self,points,body):
        from .triangle_contacts import sample_points,surface_contacts
        samples=sample_points(points,self.ids,self.weights)
        depth,normals=surface_contacts(samples,self.samples,self.body_rest[self.body_faces],body.triangles,body.nearest)
        force=normals*depth[:,None]
        trees={k:TriangleSurface(points,self.component_faces[k]) for k in self.components}
        for k in self.components:
            mask=self.sample_labels!=k
            d,n=surface_contacts(samples[mask],self.samples[mask],self.rest[self.component_faces[k]],
                trees[k].triangles,trees[k].nearest,gap=.0015,radius=.015,rest_order=True)
            force[mask]+=n*d[:,None]
        length=np.linalg.norm(force,axis=1)
        return length,force/np.maximum(length[:,None],1e-20)

    def measure_frame(self,points,body):
        from .triangle_contacts import sample_points,surface_contacts
        samples=sample_points(points,self.ids,self.weights)
        depth,_=surface_contacts(samples,self.samples,self.body_rest[self.body_faces],body.triangles,body.nearest)
        cloth=TriangleSurface(points,self.faces)
        body_pairs=body.crossings(cloth)-self.rest_body_pairs
        counts=np.bincount([self.labels[self.faces[b,0]] for _,b in body_pairs],minlength=len(self.components))
        components={}
        for k in self.components:
            d=depth[self.sample_labels==k]
            components[int(k)]={'body_samples_over_1mm':int((d>.001).sum()),'max_body_depth_m':float(d.max(initial=0)),
                                'body_depth_sum_m':float(d.sum()),'body_depth_energy_m2':float(np.sum(d*d)),
                                'new_body_triangle_pairs':int(counts[k])}
        trees={k:TriangleSurface(points,self.component_faces[k]) for k in self.components}
        layers={}
        for (a,b),rest_pairs in self.rest_layer_pairs.items():
            values=[]
            for source,target in [(a,b),(b,a)]:
                mask=self.sample_labels==source
                d,_=surface_contacts(samples[mask],self.samples[mask],self.rest[self.component_faces[target]],
                    trees[target].triangles,trees[target].nearest,gap=.0015,radius=.015,rest_order=True)
                values.append(d)
            d=np.concatenate(values)
            layers[f'{a}:{b}']={'triangle_pairs':len(trees[a].crossings(trees[b])-rest_pairs),
                               'contact_samples_over_1mm':int((d>.001).sum()),
                               'max_depth_m':float(d.max(initial=0)),'depth_sum_m':float(d.sum()),'depth_energy_m2':float(np.sum(d*d))}
        return components,layers


def measure_triangle_tracks(profile,track,body_at,tag='TRACK'):
    components={int(k):{'body_samples_over_1mm':0,'max_body_depth_m':0.,'body_depth_sum_m':0.,'body_depth_energy_m2':0.,
                        'new_body_triangle_pairs':0,'max_frame_body_triangle_pairs':0} for k in profile.components}
    layers={f'{a}:{b}':{'triangle_pairs':0,'max_frame_pairs':0,'frames':0,
                        'contact_samples_over_1mm':0,'max_depth_m':0.,'depth_sum_m':0.,'depth_energy_m2':0.} for a,b in profile.rest_layer_pairs}
    for f,points in enumerate(track):
        current,pairs=profile.measure_frame(points,body_at(f))
        for k,row in current.items():
            total=components[k]
            for name in ['body_samples_over_1mm','body_depth_sum_m','body_depth_energy_m2','new_body_triangle_pairs']:total[name]+=row[name]
            total['max_body_depth_m']=max(total['max_body_depth_m'],row['max_body_depth_m'])
            total['max_frame_body_triangle_pairs']=max(total['max_frame_body_triangle_pairs'],row['new_body_triangle_pairs'])
        for key,row in pairs.items():
            n=row['triangle_pairs'];layers[key]['triangle_pairs']+=n
            layers[key]['max_frame_pairs']=max(layers[key]['max_frame_pairs'],n)
            layers[key]['frames']+=int(n>0)
            layers[key]['contact_samples_over_1mm']+=row['contact_samples_over_1mm']
            layers[key]['max_depth_m']=max(layers[key]['max_depth_m'],row['max_depth_m'])
            layers[key]['depth_sum_m']+=row['depth_sum_m']
            layers[key]['depth_energy_m2']+=row['depth_energy_m2']
        if f%150==0:print('MESH_CONTACT_AUDIT',tag,f+1,len(track),flush=True)
    return {'components':components,'layers':layers,
            'body_samples_over_1mm':sum(c['body_samples_over_1mm'] for c in components.values()),
            'max_body_depth_m':max(c['max_body_depth_m'] for c in components.values()),
            'body_depth_sum_m':sum(c['body_depth_sum_m'] for c in components.values()),
            'body_depth_energy_m2':sum(c['body_depth_energy_m2'] for c in components.values()),
            'new_body_triangle_pairs':sum(c['new_body_triangle_pairs'] for c in components.values()),
            'new_layer_triangle_pairs':sum(c['triangle_pairs'] for c in layers.values()),
            'layer_contact_samples_over_1mm':sum(c['contact_samples_over_1mm'] for c in layers.values()),
            'max_layer_depth_m':max((c['max_depth_m'] for c in layers.values()),default=0.),
            'layer_depth_sum_m':sum(c['depth_sum_m'] for c in layers.values()),
            'layer_depth_energy_m2':sum(c['depth_energy_m2'] for c in layers.values())}


def mesh_contact_tracks(raw,baseline,caps,limits,edges,rest,faces,body_rest,body_faces,body_at,fps):
    """Offline sampled proposals, full-frame guards and whole-component fallback."""
    from .triangle_contacts import scatter_corrections,bound_surface,regularize_contact_delta,support_safe_track
    from .surface_directed import relax_surface,smooth_corrections
    profile=MeshContactProfile(rest,edges,faces,body_rest,body_faces)
    stride=max(1,round(fps/10));grid=np.unique(np.r_[np.arange(0,len(raw),stride),len(raw)-1])
    proposals=[]
    for i,f in enumerate(grid):
        body=body_at(int(f));p=np.array(baseline[f])
        for _ in range(8):
            depth,normal=profile.force(p,body)
            p=bound_surface(raw[f],baseline[f],p+.9*scatter_corrections(depth,normal,profile.ids,profile.weights,profile.groups,edges),extra=.03)
            p=relax_surface(raw[f],p,caps[f],limits,edges,.12,iterations=4)
        proposals.append(bound_surface(raw[f],baseline[f],p,extra=.03)-baseline[f])
        if i%30==0:print('MESH_CONTACT_PROPOSAL',i+1,len(grid),flush=True)
    proposals=np.asarray(proposals)
    if len(grid)==1:candidate=baseline+proposals
    else:
        index=np.clip(np.searchsorted(grid,np.arange(len(raw)),side='right')-1,0,len(grid)-2)
        alpha=(np.arange(len(raw))-grid[index])/(grid[index+1]-grid[index])
        delta=proposals[index]*(1-alpha[:,None,None])+proposals[index+1]*alpha[:,None,None]
        candidate=smooth_corrections(baseline,baseline+delta,fps)
    for f in range(len(raw)):
        candidate[f]=regularize_contact_delta(raw[f],baseline[f],candidate[f],caps[f],limits,edges,profile.groups)
    candidate,support=support_safe_track(baseline,candidate,caps,limits,edges,rest)
    final,report=finish_mesh_candidate(profile,raw,baseline,candidate,caps,limits,edges,rest,body_at,fps,stride)
    report['support_ceiling']=support
    return final,report


def audit_mesh_contacts(scene,obj,track,expected,geometry_reference):
    """Recompute actual saved surfaces; never write the opened project."""
    body,body_rest,body_faces=configured_body_surface(scene)
    rest=np.array([v.co[:] for v in obj.data.vertices]);edges=np.array([e.vertices[:] for e in obj.data.edges])
    obj.data.calc_loop_triangles();faces=np.array([p.vertices[:] for p in obj.data.loop_triangles])
    profile=MeshContactProfile(rest,edges,faces,body_rest,body_faces)
    def body_at(f):
        scene.frame_set(scene.frame_start+f)
        return TriangleSurface(evaluated_points(body),body_faces)
    actual=measure_triangle_tracks(profile,track,body_at,'SAVED')
    reference=expected['after'];differences={};metric_equal=True
    for name in ['body_samples_over_1mm','new_body_triangle_pairs','new_layer_triangle_pairs','layer_contact_samples_over_1mm']:
        diff=actual[name]-reference[name];tolerance=max(6,int(reference[name]*.0005))
        differences[name]={'difference':diff,'tolerance':tolerance}
        metric_equal &= abs(diff)<=tolerance
    for name in ['max_body_depth_m','max_layer_depth_m','body_depth_sum_m','layer_depth_sum_m','body_depth_energy_m2','layer_depth_energy_m2']:
        diff=actual[name]-reference[name]
        tolerance=1e-5 if name.startswith('max_') else (.001 if name.endswith('_m2') else .005)
        differences[name]={'difference':diff,'tolerance':tolerance};metric_equal &= abs(diff)<=tolerance
    return {'enabled':True,'saved_geometry_consistent':bool(geometry_reference['passed']),'metrics':actual,
            'geometry_reference':geometry_reference,'metric_equality_within_tolerance':bool(metric_equal),
            'solver_differences':differences,'metric_differences_are_diagnostic':True,
            'note':'Export fidelity is checked at every vertex/frame against its geometry reference. Threshold contacts and triangle-edge classifications can change under float32 evaluation; actual saved metrics are authoritative and differences remain visible, not a motion-accuracy score. Integer frames only; not collision-free certification.'}


def finish_mesh_candidate(profile,raw,baseline,candidate,caps,limits,edges,rest,body_at,fps,stride):
    from .triangle_contacts import select_contact_components
    from .surface_directed import track_metrics
    final,guard=select_contact_components(raw,baseline,candidate,caps,limits,edges,rest,fps)
    before=measure_triangle_tracks(profile,baseline,body_at,'BEFORE')
    measured=before if np.array_equal(final,baseline) else measure_triangle_tracks(profile,final,body_at,'CANDIDATE')
    rejected={}
    for k in profile.components:
        a=before['components'][int(k)];b=measured['components'][int(k)];reasons=[]
        for name in ['body_samples_over_1mm','new_body_triangle_pairs','max_frame_body_triangle_pairs']:
            if b[name]>a[name]:reasons.append(name)
        if b['max_body_depth_m']>a['max_body_depth_m']+5e-5:reasons.append('max_body_depth_m')
        if b['body_depth_sum_m']>a['body_depth_sum_m']+1e-5:reasons.append('body_depth_sum_m')
        if b['body_depth_energy_m2']>a['body_depth_energy_m2']+1e-7:reasons.append('body_depth_energy_m2')
        if reasons:rejected[int(k)]=reasons;final[:,profile.labels==k]=baseline[:,profile.labels==k]
    # A retained neighbour can change a pair's outcome. Recheck combinations
    # and conservatively retain both participants when a layer count regresses.
    needs_measure=bool(rejected)
    for _ in range(len(profile.components)):
        if needs_measure:measured=measure_triangle_tracks(profile,final,body_at,'COMBINED')
        bad=[]
        for pair,b in measured['layers'].items():
            a=before['layers'][pair]
            if (b['triangle_pairs']>a['triangle_pairs']
                or b['max_depth_m']>a['max_depth_m']+5e-5 or b['depth_sum_m']>a['depth_sum_m']+1e-5
                or b['depth_energy_m2']>a['depth_energy_m2']+1e-7):
                bad.extend(map(int,pair.split(':')))
        changed=False
        for k in set(bad):
            mask=profile.labels==k
            if not np.array_equal(final[:,mask],baseline[:,mask]):
                final[:,mask]=baseline[:,mask];rejected.setdefault(k,[]).append('layer_pair_regression');changed=True
        if not changed:break
        if np.array_equal(final,baseline):measured=before;needs_measure=False
        else:needs_measure=True
    # Repeat the explicit topology budget and original capsule/motion limits
    # after replacements, including combined-percentile effects.
    from .triangle_contacts import contact_track_gate
    combined=contact_track_gate(raw,baseline,final,caps,limits,edges,fps)
    if not combined['accepted']:final=baseline.copy()
    if not combined['accepted']:measured=before
    return final,{'enabled':True,'method':'actual_triangles_rest_order_components_v1',
        'proposal_stride_frames':stride,'extra_shift_budget_m':.03,'total_shift_budget_m':.12,
        'body_query_radius_m':.03,'layer_query_radius_m':.015,'clearance_m':.0015,
        'before':before,'after':measured,'topology_motion_guard':guard,'combined_guard':combined,
        'rejected_components':rejected,'after_motion':track_metrics(raw,final,caps,limits,edges,fps),
        'maximum_extra_shift_m':float(np.linalg.norm(final-baseline,axis=-1).max()),
        'triangle_pair_method':'proper_noncoplanar_crossing_v2',
        'note':'Actual triangles with vertex/edge/face sampling and proper non-coplanar edge-through-face confirmation after BVH broad phase (10um tolerance). Rest crossings excluded from new-pair counts. Integer-frame, local signed nearest surfaces; no watertight volume, continuous/self-collision or cloth dynamics guarantee.'}
