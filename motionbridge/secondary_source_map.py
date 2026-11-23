"""Source-index surface adapter. No inference from final simulated clothing.

Keeps render topology, UV seams and shape keys intact. Only explicitly reviewed
and natively checked display aliases share physical degrees of freedom. This is
geometry ingestion/return mapping, NOT a collision or joint solver.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .secondary_segmentation import _sha, review_gate


def load_model_document(root, reference):
    if not isinstance(reference, dict) or set(reference) != {'path', 'sha256'}:
        raise ValueError('模型映射文档需要路径和SHA256')
    root = Path(root).resolve()
    path = (root / reference['path']).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('模型映射文档必须位于当前开发副本')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != reference['sha256']:
        raise ValueError('模型映射文档指纹改变，需重新审核')
    return json.loads(data.decode('utf-8'))


class SourceSurfaceMap:
    def __init__(self, plan, review, manifest):
        inv = plan['surface_segmentation']['inventory']
        geometry = plan['geometry']
        if _sha({k:geometry[k] for k in ('positions_m','triangles','face_material_ids','material_names','skin_weights')}) != inv['geometry_sha256']:
            raise ValueError('源几何与已审核指纹不一致')
        if not review_gate(inv, review)['accepted']:
            raise ValueError('真实表面分类不完整')
        if (manifest.get('schema_version') != 1 or
                manifest.get('model_sha256') != plan['model_sha256'] or
                manifest.get('geometry_sha256') != inv['geometry_sha256'] or
                manifest.get('review_sha256') != _sha(review)):
            raise ValueError('源表面映射与当前模型/分类不匹配')
        self.rest = np.array(plan['geometry']['positions_m'], dtype=float)
        self.triangles = np.array(plan['geometry']['triangles'], dtype=int)
        self.parts = review['parts']
        self.face_part = np.full(len(self.triangles), -1, dtype=int)
        for i, part in enumerate(self.parts):
            self.face_part[part['face_ids']] = i
        self.alias_rows = manifest['aliases']
        groups = {g['id']: g for g in inv['coincident_face_groups']}
        relations = {g['group_id']: g['kind'] for g in review['overlaps']}
        parent = np.arange(len(self.rest))
        def find(v):
            while parent[v] != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            return int(v)
        hidden = set(); mapped = {}
        skins = plan['geometry']['skin_weights']
        for row in self.alias_rows:
            gid = row['group_id']; group = groups.get(gid)
            if not group or relations[gid] != 'display_duplicate' or not group['matching_skin_weights']:
                raise ValueError('未审核或不同蒙皮重合面不能映射')
            a, b = row['display_face'], row['canonical_face']
            if (type(a) is not int or type(b) is not int or a == b or
                    a not in group['face_ids'] or b not in group['face_ids'] or a in hidden):
                raise ValueError('显示面映射重复或越界')
            if self.parts[self.face_part[b]]['category'] == 'display_overlay':
                raise ValueError('主面不能是显示副本')
            pairs = row['vertex_pairs']
            if (len(pairs) != 3 or any(len(p) != 2 or any(type(v) is not int for v in p) for p in pairs)
                    or sorted(p[0] for p in pairs) != sorted(self.triangles[a].tolist())
                    or sorted(p[1] for p in pairs) != sorted(self.triangles[b].tolist())):
                raise ValueError('显示面顶点对应必须是源三角的双射')
            for v, w in pairs:
                if np.linalg.norm(self.rest[v] - self.rest[w]) > 2e-6 or skins[v] != skins[w]:
                    raise ValueError('显示点位置或蒙皮不等价')
                x, y = find(v), find(w)
                parent[max(x,y)] = min(x,y)
            hidden.add(a); mapped.setdefault(gid, []).append((a,b))
        for gid, kind in relations.items():
            if kind == 'display_duplicate':
                rows = mapped.get(gid, [])
                if len(rows) != len(groups[gid]['face_ids']) - 1 or len({b for a,b in rows}) != 1:
                    raise ValueError('显示重叠组映射不完整')
        declared = {f for p in self.parts if p['category'] == 'display_overlay' for f in p['face_ids']}
        if not declared <= hidden:
            raise ValueError('显示副本存在未映射源面')
        self.canonical_vertex = np.array([find(i) for i in range(len(self.rest))])
        self.physical_face_ids = np.array([f for f in range(len(self.triangles)) if f not in hidden])
        self.physical_triangles = self.canonical_vertex[self.triangles[self.physical_face_ids]]
        self.pairs = np.array([p for r in self.alias_rows for p in r['vertex_pairs']],dtype=int).reshape(-1,2)
        fixed = {v for r in plan['regions'] if r['role'] in ('body','boots') for v in r['vertex_ids']}
        fixed.update(plan['geometry'].get('unreferenced_vertex_ids', []))
        self.fixed_canonical = np.unique(self.canonical_vertex[sorted(fixed)])
        self.fixed_vertices = np.flatnonzero(np.isin(self.canonical_vertex, self.fixed_canonical))
        self._validate_authored_references(plan, manifest)

    def _validate_authored_references(self, plan, manifest):
        rows = manifest['parts']
        if len(rows) != len(self.parts) or {r['part_id'] for r in rows} != {p['id'] for p in self.parts}:
            raise ValueError('每个真实部件必须有源绑定记录')
        parts = {p['id']: p for p in self.parts}; skins = plan['geometry']['skin_weights']
        bones = {b['id'] for b in plan['bones']}
        for row in rows:
            vs = {v for f in parts[row['part_id']]['face_ids'] for v in self.triangles[f]}
            expected = sorted({int(b) for v in vs for b,w in skins[v].items() if w > 0})
            if row['source_bone_ids'] != expected or not set(expected) <= bones:
                raise ValueError('部件源骨骼记录与实际蒙皮不一致')
            rs = [r['id'] for r in plan['rigid_bodies'] if r['bone_id'] in expected]
            js = [j['id'] for j in plan['joints'] if set(j['rigid_ids']) & set(rs)]
            if row['native_rigid_ids'] != rs or row['native_joint_ids'] != js:
                raise ValueError('部件原生刚体/关节引用不完整或被改写')
            if row.get('attachment_reference') != 'immutable_source_skin_and_authored_graph':
                raise ValueError('源映射禁止使用最终衣料跟随')
        self.authored_parts = rows

    def check_positions(self, positions, tolerance=2e-6):
        x = np.asarray(positions, dtype=float)
        if x.shape != self.rest.shape or not np.isfinite(x).all():
            raise ValueError('原生源顶点数量或数值错误')
        error = float(np.linalg.norm(x[self.pairs[:,0]]-x[self.pairs[:,1]],axis=1).max()) if len(self.pairs) else 0.
        if error > tolerance:
            raise ValueError('当前姿态/形态已使显示映射分离，禁止复用物理主面')
        return error

    def expand_delta(self, source_positions, canonical_delta):
        """Return a separate array; no replacement of source skin/shape/animation."""
        self.check_positions(source_positions)
        delta = np.asarray(canonical_delta,dtype=float)
        if delta.shape != self.rest.shape or not np.isfinite(delta).all():
            raise ValueError('增量必须覆盖源顶点且为有限值')
        # Noncanonical rows cannot be silently ignored.
        noncanonical = np.flatnonzero(self.canonical_vertex != np.arange(len(self.rest)))
        if np.any(delta[noncanonical] != 0) or np.any(delta[self.fixed_canonical] != 0):
            raise ValueError('显示副本或受保护表面不能独立移动')
        return np.asarray(source_positions,dtype=float) + delta[self.canonical_vertex]

    def summary(self):
        return {'source_faces':len(self.triangles),'physical_faces':len(self.physical_face_ids),
                'display_alias_faces':len(self.alias_rows),'semantic_parts':len(self.parts),
                'fixed_vertices_including_aliases':len(self.fixed_vertices),
                'source_geometry_mapping_prepared':True,'native_pose_check_required_each_sample':True,
                'raw_audit_uses_all_source_faces':True,'contact_solver_connected':False,
                'dynamic_attachment_constraints_complete':False,'simulation_ready':False}


def compile_source_map(plan, review, manifest):
    mapping = SourceSurfaceMap(plan,review,manifest)
    return {**mapping.summary(),'manifest_sha256':_sha(manifest),
            'parts':mapping.authored_parts,
            'blockers':['native_joint_to_surface_constraints_pending',
                        'full_mesh_contact_self_contact_and_ccd_pending']}
