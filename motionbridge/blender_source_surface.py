"""Blender-only read adapter for reviewed real model surfaces.

Does not enable native rigid-body/cloth simulations or modify the source mesh.
Returned geometry is suitable for future joint/contact assembly, which must
still enforce its own capability and quality gates.
"""
import numpy as np

from .secondary_source_map import SourceSurfaceMap
from .blender_secondary_rigid import evaluated_mesh_records


class BlenderSourceSurfaceAdapter:
    def __init__(self, mesh, rig, plan, review, manifest):
        import bpy
        self.mesh, self.rig, self.plan = mesh, rig, plan
        self.mapping = SourceSurfaceMap(plan,review,manifest)
        scene = bpy.context.scene
        if scene.rigidbody_world and scene.rigidbody_world.enabled:
            raise ValueError('源场景已启用刚体，禁止把仿真输出当原始动作')
        if any(m.type != 'ARMATURE' and m.show_viewport for m in mesh.modifiers):
            raise ValueError('源修改器不在只读适配范围，需单独验证源点映射')
        if any(m.type == 'ARMATURE' and m.show_viewport and m.object != rig for m in mesh.modifiers):
            raise ValueError('源网格的骨骼修改器未绑定声明骨架')
        if not any(m.type == 'ARMATURE' and m.show_viewport and m.object == rig for m in mesh.modifiers):
            raise ValueError('源网格缺少启用的声明骨架修改器')
        expected = self.mapping.rest
        if len(mesh.data.vertices) != len(expected):
            raise ValueError('源顶点数量变化')
        local = np.empty_like(expected);mesh.data.vertices.foreach_get('co',local.ravel())
        m = np.array(mesh.matrix_world);world = local @ m[:3,:3].T + m[:3,3]
        if np.max(np.abs(world-expected)) > 2e-6:
            raise ValueError('源静态位置或单位发生变化')
        self._check_topology(mesh.data)
        names = {g.index:g.name for g in mesh.vertex_groups}
        bone_names = {b['id']:b['blender_name'] for b in plan['bones']}
        deform_names = set(bone_names.values())
        if not deform_names <= set(rig.data.bones.keys()):
            raise ValueError('原生源骨骼缺失')
        for bone in plan['bones']:
            actual_head=np.array(rig.matrix_world @ rig.data.bones[bone['blender_name']].head_local)
            if np.linalg.norm(actual_head-np.array(bone['rest_head_m']))>2e-6:
                raise ValueError('源骨架静态位置或单位变化')
        for vertex,weights in zip(mesh.data.vertices,plan['geometry']['skin_weights']):
            actual = {names[g.group]:g.weight for g in vertex.groups if names[g.group] in deform_names and g.weight>0}
            expected_weights = {bone_names[int(b)]:w for b,w in weights.items()}
            if any(abs(actual.get(k,0)-expected_weights.get(k,0))>2e-6 for k in set(actual)|set(expected_weights)):
                raise ValueError('源蒙皮变化，不能沿用已审核绑定')
        shape_errors=[]
        for key in mesh.data.shape_keys.key_blocks if mesh.data.shape_keys else []:
            points=np.empty_like(expected);key.data.foreach_get('co',points.ravel())
            points=points @ m[:3,:3].T + m[:3,3]
            shape_errors.append({'name':key.name,'alias_error_m':self.mapping.check_positions(points)})
        self.validation={'shape_keys_checked':len(shape_errors),'shape_key_alias_errors':shape_errors,
                         'native_source_geometry_verified':True,'simulation_performed':False}

    def _check_topology(self, data):
        triangles = np.array([p.vertices[:] for p in data.polygons],dtype=int)
        if triangles.shape != self.mapping.triangles.shape or not np.array_equal(np.sort(triangles,axis=1),np.sort(self.mapping.triangles,axis=1)):
            raise ValueError('源三角拓扑或面编号变化')
        if [p.material_index for p in data.polygons] != self.plan['geometry']['face_material_ids']:
            raise ValueError('源材质面归属改变')

    def capture(self):
        """Evaluate current source pose; call again for EACH integer/half sample."""
        points,ids,faces,face_ids=evaluated_mesh_records(self.mesh)
        if (not np.array_equal(ids,np.arange(len(self.mapping.rest))) or
                not np.array_equal(face_ids,np.arange(len(self.mapping.triangles))) or
                faces.shape != self.mapping.triangles.shape or
                not np.array_equal(np.sort(faces,axis=1),np.sort(self.mapping.triangles,axis=1))):
            raise ValueError('当前源姿态的拓扑/源编号变化')
        self._check_topology(self.mesh.data)
        error=self.mapping.check_positions(points)
        bones=np.array([self.rig.matrix_world @ self.rig.pose.bones[b['blender_name']].matrix for b in self.plan['bones']])
        if not np.isfinite(bones).all():
            raise ValueError('源骨骼矩阵含非有限值')
        return {'source_positions':points,'physical_face_ids':self.mapping.physical_face_ids.copy(),
                'physical_triangles':self.mapping.canonical_vertex[faces[self.mapping.physical_face_ids]],
                'source_bone_world':bones,'display_alias_error_m':error}
