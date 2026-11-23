"""Separate shape-preserving ornament objects; no edits to original skin/keys.

Kinematic attachments, not a free 6-DOF rigid-body simulation. Quaternion object
transforms preserve rigidity between frames, unlike interpolated vertex caches.
"""
import bpy,numpy as np
from mathutils import Matrix


def source_attribute(mesh,name,domain,values):
    attribute=mesh.attributes.get(name) or mesh.attributes.new(name=name,type='INT',domain=domain)
    attribute.data.foreach_set('value',np.asarray(values,dtype=np.int32))


def create_rigid_objects(source,plan,parts,transforms,frames):
    rest=np.asarray(plan['geometry']['positions_m']);all_faces=np.asarray(plan['geometry']['triangles'],int)
    ownership={int(v):index for index,part in enumerate(parts) for v in part}
    face_groups=[[] for _ in parts]
    for face_id,face in enumerate(all_faces):
        owners=[ownership.get(int(v),-1) for v in face]
        if owners[0]>=0 and owners[0]==owners[1]==owners[2]:face_groups[owners[0]].append(face_id)
    collection=bpy.data.collections.new('MB4_RigidAttachments');bpy.context.scene.collection.children.link(collection)
    mask=np.zeros(len(source.data.polygons),dtype=bool);objects=[]
    for index,(vertices,face_ids) in enumerate(zip(parts,face_groups)):
        if not face_ids:continue
        vertices=np.asarray(vertices,int);center=rest[vertices].mean(0)
        local_faces=np.searchsorted(vertices,all_faces[face_ids])
        assert np.array_equal(vertices[local_faces],all_faces[face_ids])
        data=bpy.data.meshes.new(f'MB4_RigidMesh_{index}');data.from_pydata((rest[vertices]-center).tolist(),[],local_faces.tolist());data.update()
        for material in source.data.materials:data.materials.append(material)
        for poly,source_id in zip(data.polygons,face_ids):
            poly.material_index=source.data.polygons[source_id].material_index;poly.use_smooth=source.data.polygons[source_id].use_smooth
        for original_uv in source.data.uv_layers:
            uv=data.uv_layers.new(name=original_uv.name)
            for poly,source_id in zip(data.polygons,face_ids):
                source_poly=source.data.polygons[source_id]
                for target_loop,source_loop in zip(poly.loop_indices,source_poly.loop_indices):
                    uv.data[target_loop].uv=original_uv.data[source_loop].uv
        source_attribute(data,'MB4_SourceVertex','POINT',vertices)
        source_attribute(data,'MB4_SourceFace','FACE',face_ids)
        obj=bpy.data.objects.new(f'MB4_RigidAttachment_{index}',data);collection.objects.link(obj)
        obj.rotation_mode='QUATERNION';obj['motionbridge_attachment_mode']='kinematic_rigid_shape_preserving'
        action=bpy.data.actions.new(f'MB4_RigidMotion_{index}');slot=action.slots.new(id_type='OBJECT',name=obj.name)
        layer=action.layers.new('Model rigid attachment');strip=layer.strips.new(type='KEYFRAME');bag=strip.channelbags.new(slot)
        for path,offset,width in [('location',0,3),('rotation_quaternion',3,4)]:
            for component in range(width):
                curve=bag.fcurves.new(data_path=path,index=component);curve.keyframe_points.add(len(frames))
                curve.keyframe_points.foreach_set('co',np.column_stack((frames,transforms[:,index,offset+component])).ravel())
                for key in curve.keyframe_points:key.interpolation='LINEAR'
                curve.update()
        obj.animation_data_create();obj.animation_data.action=action;obj.animation_data.action_slot=slot
        objects.append(obj);mask[face_ids]=True
    source_attribute(source.data,'MB4_SourceVertex','POINT',np.arange(len(source.data.vertices)))
    source_attribute(source.data,'MB4_SourceFace','FACE',np.arange(len(source.data.polygons)))
    attribute=source.data.attributes.new(name='MB4_RigidFaceMask',type='BOOLEAN',domain='FACE')
    attribute.data.foreach_set('value',mask)
    group=bpy.data.node_groups.new('MB4_KeepOriginalDataHideRigidFaces','GeometryNodeTree')
    group.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry')
    group.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
    incoming=group.nodes.new('NodeGroupInput');outgoing=group.nodes.new('NodeGroupOutput')
    named=group.nodes.new('GeometryNodeInputNamedAttribute');named.data_type='BOOLEAN';named.inputs['Name'].default_value='MB4_RigidFaceMask'
    delete=group.nodes.new('GeometryNodeDeleteGeometry');delete.domain='FACE';delete.mode='ALL'
    group.links.new(incoming.outputs['Geometry'],delete.inputs['Geometry'])
    group.links.new(named.outputs['Attribute'],delete.inputs['Selection'])
    group.links.new(delete.outputs['Geometry'],outgoing.inputs['Geometry'])
    modifier=source.modifiers.new('MB4_RenderRigidPiecesSeparately','NODES');modifier.node_group=group
    return objects,np.flatnonzero(mask)


def evaluated_mesh_records(obj):
    ev=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());data=ev.to_mesh()
    try:
        points=np.empty(len(data.vertices)*3,dtype=np.float32);data.vertices.foreach_get('co',points);points=points.reshape(-1,3)
        transform=np.asarray(obj.matrix_world);points=points@transform[:3,:3].T+transform[:3,3]
        attribute=data.attributes.get('MB4_SourceVertex')
        indices=np.arange(len(points),dtype=np.int32)
        if attribute:attribute.data.foreach_get('value',indices)
        data.calc_loop_triangles();faces=np.asarray([t.vertices[:] for t in data.loop_triangles],int)
        face_attribute=data.attributes.get('MB4_SourceFace')
        source_faces=np.arange(len(data.polygons),dtype=np.int32)
        if face_attribute:face_attribute.data.foreach_get('value',source_faces)
        triangle_source_ids=np.asarray([source_faces[t.polygon_index] for t in data.loop_triangles],int)
        return points,indices,faces,triangle_source_ids
    finally:ev.to_mesh_clear()
