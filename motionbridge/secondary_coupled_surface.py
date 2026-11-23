"""Experimental source-index joint solve: mounts, cloth structure and contacts.

No production enablement, time integration, CCD or material calibration. The
source pose is immutable. Failed candidates return the original source array.
"""
from collections import defaultdict
import numpy as np
from .secondary_attachment_driver import exp_rotation
from .secondary_mount_reference import mount_reference
from .secondary_structure_reference import build_reference, check_reference

CLOTH = {'inner_cloth', 'outer_cloth', 'flexible_trim'}


def cloth_structure(mapping):
    """Native indexed edges and opposite-vertex bending distances per part.

    This bending approximation is not a signed dihedral-angle constraint; it
    cannot distinguish a reflected fold. No unreviewed UV seams are welded.
    """
    edges = set(); bends = set(); nonmanifold = []; active = set()
    physical = set(map(int, mapping.physical_face_ids))
    for part in mapping.parts:
        if part['category'] not in CLOTH:
            continue
        fans = defaultdict(list)
        for f in part['face_ids']:
            if f not in physical:
                continue
            tri = list(map(int, mapping.canonical_vertex[mapping.triangles[f]]))
            if len(set(tri)) != 3:
                raise ValueError('Collapsed physical cloth triangle')
            active.update(tri)
            for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
                edge = tuple(sorted((tri[i], tri[j])))
                edges.add(edge); fans[edge].append(tri[k])
        for edge, opposite in fans.items():
            if len(opposite) == 2 and opposite[0] != opposite[1]:
                bends.add(tuple(sorted(opposite)))
            elif len(opposite) > 2:
                nonmanifold.append({'part_id': part['id'], 'edge': list(edge)})
    return {'edges': sorted(edges), 'bends': sorted(bends),
            'cloth_vertices': sorted(active), 'nonmanifold_edges': nonmanifold,
            'bending_model': 'opposite_vertex_distance_not_signed_dihedral',
            'unreviewed_seams_welded': False}


def compile_coupled_surface(plan, mapping):
    s = cloth_structure(mapping)
    return {'implementation': 'source_mounted_cloth_discrete_v2_streamed',
            'contact_scan': {'maximum_pairs_per_block': 4096, 'resident_pair_limit': 100000,
                             'source_precision': 'float64', 'exit_rule_schema': 1,
                             'automatic_asset_exit_side_inference': False},
            'edge_count': len(s['edges']), 'bend_count': len(s['bends']),
            'cloth_vertices': len(s['cloth_vertices']),
            'nonmanifold_edges': s['nonmanifold_edges'],
            'bending_model': s['bending_model'],
            'mount_parts': len(plan['surface_mounts']['mounts']),
            'collision_backend_required': True, 'simulation_ready': False,
            'full_physics_accepted': False}


class MountedClothFrame:
    """One shared state for every constraint, rigid whole-part DOFs included."""
    def __init__(self, source, mapping, compiled, connections, inverse_mass,
                 rigid_metrics, predicted_offsets=None):
        mapping.check_positions(source)
        self.source = np.array(source, float, copy=True)
        self.mapping = mapping; self.mounts = compiled['mounts']
        self.x = self.source.copy(); self.structure = cloth_structure(mapping)
        if self.structure['nonmanifold_edges']:
            raise ValueError('Nonmanifold cloth needs reviewed bending topology')
        self.weights = np.array(inverse_mass, float, copy=True)
        if self.weights.shape != (len(source),) or not np.isfinite(self.weights).all() or (self.weights < 0).any():
            raise ValueError('Invalid cloth inverse masses')
        self.fixed = np.unique(np.r_[mapping.fixed_canonical,
            mapping.canonical_vertex[connections['source_root_vertex_ids']]])
        active = np.asarray(self.structure['cloth_vertices'], int)
        self.weights[~np.isin(np.arange(len(source)), active)] = 0
        self.weights[self.fixed] = 0
        self.stitches = connections['stitch_vertex_pairs']
        self.states = {}; self.owner = {}; self.local = {}; self.metrics = {}
        reference_points = self.source.copy()
        self.mount_reference_audit = {}
        pids = {m['part_id'] for m in self.mounts}
        if set(rigid_metrics) != pids or set(predicted_offsets or {}) - pids:
            raise ValueError('Incomplete rigid metrics or unknown prediction')
        for m in self.mounts:
            pid = m['part_id']; ids = m['canonical_vertex_ids']
            if set(ids) & set(self.fixed) or any(v in self.owner for v in ids):
                raise ValueError('Rigid part overlaps fixed root or another body')
            metric = np.asarray(rigid_metrics[pid], float)
            offset = np.asarray((predicted_offsets or {}).get(pid, [0, 0, 0]), float)
            if metric.shape != (2,) or not np.isfinite(metric).all() or (metric < 0).any():
                raise ValueError('Invalid rigid inverse metrics')
            if offset.shape != (3,) or not np.isfinite(offset).all() or (metric[0] == 0 and np.any(offset)):
                raise ValueError('Invalid prediction for rigid state')
            rot, center, local, world, reference_audit = mount_reference(self.source, mapping, m)
            reference_points[ids] = world
            self.states[pid] = [rot, center + offset]
            self.local[pid] = local
            self.mount_reference_audit[pid] = reference_audit
            self.metrics[pid] = metric
            self.owner.update({v: pid for v in ids})
            self.weights[ids] = 0
        self.edges = np.array(self.structure['edges'], int).reshape(-1, 2)
        self.bends = np.array(self.structure['bends'], int).reshape(-1, 2)
        self.reference, self.reference_binding, self.reference_issues = build_reference(
            self, reference_points)
        self.edge_lengths = self.lengths(self.reference, self.edges)
        self.bend_lengths = self.lengths(self.reference, self.bends)
        self.edge_colors = self.color(self.edges)
        self.bend_colors = self.color(self.bends)

    @staticmethod
    def lengths(x, edges):
        return np.linalg.norm(x[edges[:, 1]] - x[edges[:, 0]], axis=1)

    @staticmethod
    def color(edges):
        occupied = defaultdict(set); batches = []
        for i, (a, b) in enumerate(edges):
            c = 0
            while c in occupied[a] or c in occupied[b]: c += 1
            while len(batches) <= c: batches.append([])
            batches[c].append(i); occupied[a].add(c); occupied[b].add(c)
        return [np.array(batch, int) for batch in batches]

    def point(self, vertex):
        v = int(self.mapping.canonical_vertex[vertex])
        pid = self.owner.get(v)
        return self.x[v] if pid is None else self.states[pid][0] @ self.local[pid][v] + self.states[pid][1]

    def positions(self):
        delta = np.zeros_like(self.x); canonical = np.unique(self.mapping.canonical_vertex)
        delta[canonical] = self.x[canonical] - self.source[canonical]
        for v in self.owner: delta[v] = self.point(v) - self.source[v]
        return self.mapping.expand_delta(self.source, delta)

    def project_scalar(self, ids, gradients, value):
        blocks = {}
        for raw, gradient in zip(ids, gradients):
            v = int(self.mapping.canonical_vertex[raw]); pid = self.owner.get(v)
            key = ('cloth', v) if pid is None else ('rigid', pid)
            if pid is None:
                g = np.asarray(gradient); w = np.full(3, self.weights[v])
            else:
                arm = self.states[pid][0] @ self.local[pid][v]
                g = np.r_[gradient, np.cross(arm, gradient)]
                w = np.repeat(self.metrics[pid], 3)
            if key not in blocks: blocks[key] = [g.copy(), w]
            else: blocks[key][0] += g
        effective = sum(float((g*w) @ g) for g, w in blocks.values())
        if effective <= 1e-20: return
        updates = {key: -value*g*w/effective for key, (g, w) in blocks.items()}
        angle = max((np.linalg.norm(d[3:]) for key, d in updates.items() if key[0] == 'rigid'), default=0.)
        scale = min(1., .25/max(angle, 1e-30))
        for (kind, idx), d in updates.items():
            d *= scale
            if kind == 'cloth': self.x[idx] += d
            else:
                self.states[idx][1] += d[:3]
                self.states[idx][0] = exp_rotation(d[3:]) @ self.states[idx][0]

    def project_connections(self):
        pairs = self.stitches + [p for m in self.mounts for p in m['anchors']]
        for a, b in pairs:
            for axis in np.eye(3):
                self.project_scalar([a, b], [axis, -axis], float((self.point(a)-self.point(b)) @ axis))

    def project_structure(self):
        # Edge-coloured Gauss-Seidel: no endpoints shared inside each batch.
        for edges, rest, colors in ((self.edges, self.edge_lengths, self.edge_colors),
                                     (self.bends, self.bend_lengths, self.bend_colors)):
            for batch in colors:
                a, b = edges[batch].T
                # Vertex colouring is sufficient only for independent cloth
                # DOFs. Shared rigid endpoints use the whole-body Jacobian,
                # sequentially, since different vertices may own the same body.
                shared = np.array([int(u) in self.owner or int(v) in self.owner for u,v in zip(a,b)])
                for index in batch[shared]:
                    u,v = map(int, edges[index])
                    delta = self.point(v)-self.point(u)
                    length = float(np.linalg.norm(delta))
                    if length <= 1e-15:
                        if rest[index] > 2e-6: raise ValueError('Collapsed structural edge has no reliable direction')
                        continue
                    normal = delta/length
                    self.project_scalar([u,v], [-normal,normal], length-rest[index])
                batch = batch[~shared]
                a,b = edges[batch].T
                delta = self.x[b] - self.x[a]; length = np.linalg.norm(delta, axis=1)
                weight = self.weights[a] + self.weights[b]
                factor = np.divide(length-rest[batch], length*weight,
                    out=np.zeros_like(length), where=(length > 1e-15)&(weight > 0))
                correction = factor[:,None]*delta
                self.x[a] += self.weights[a,None]*correction
                self.x[b] -= self.weights[b,None]*correction

    def audit(self):
        x = self.positions(); pairs = self.stitches + [p for m in self.mounts for p in m['anchors']]
        edge = np.abs(self.lengths(x, self.edges)-self.edge_lengths)
        source_lengths = self.lengths(self.source, self.edges)
        source_edge_error = np.abs(self.lengths(x, self.edges)-source_lengths)
        bend = np.abs(self.lengths(x, self.bends)-self.bend_lengths)
        return {'attachment_residual_m': max((float(np.linalg.norm(x[a]-x[b])) for a,b in pairs), default=0.),
                'edge_error_m': float(edge.max(initial=0)), 'bend_distance_error_m': float(bend.max(initial=0)),
                'edges_over_25_percent': int(np.sum((source_lengths > 5e-6)&(source_edge_error > .25*source_lengths))),
                'source_edge_error_m': float(source_edge_error.max(initial=0)),
                'max_edge_relative_change': float(np.max(np.divide(source_edge_error, source_lengths, out=np.zeros_like(edge), where=source_lengths > 5e-6), initial=0)),
                'fixed_unchanged': bool(np.array_equal(x[self.fixed], self.source[self.fixed]))}


def solve_coupled_surface(frame, contacts, *, iterations=128, tolerance=2e-6):
    """Single bounded candidate, fail closed to source. No retries or mutation.

    contacts.scan performs triangle queries again after the LAST correction.
    A pre-existing crossing without an authored exit side cannot be guessed.
    """
    import copy
    if type(iterations) is not int or not 1 <= iterations <= 256 or not np.isfinite(tolerance) or not 0 < tolerance <= 2e-6:
        raise ValueError('Invalid coupled budget/tolerance')
    original = (frame.x.copy(), copy.deepcopy(frame.states)); used = 0
    report = {'candidate_committed': False, 'full_physics_accepted': False,
              'continuous_collision_checked': False, 'integration_checked': False,
              'bending_model': frame.structure['bending_model'],
              'mount_references': frame.mount_reference_audit}
    try:
        report['reference_preflight'] = check_reference(frame, tolerance)
        if not report['reference_preflight']['passed']:
            report['reason'] = 'structural_reference_infeasible'
            return frame.source.copy(), report
        baseline = contacts.scan(frame.source)
        report['source_contact_audit'] = baseline['audit']
        if baseline['blocking_reason']:
            report['reason'] = baseline['blocking_reason']
            return frame.source.copy(), report
        for step in range(iterations+1):
            current = contacts.scan(frame.positions()); audit = frame.audit()
            report.update(candidate_contact_audit=current['audit'], candidate_shape_audit=audit)
            if current['blocking_reason']:
                report['reason'] = current['blocking_reason']; break
            ok = (audit['fixed_unchanged'] and audit['edges_over_25_percent'] == 0 and
                  max(audit['attachment_residual_m'], audit['edge_error_m'], audit['bend_distance_error_m'],
                      current['audit']['maximum_deficit_m']) <= tolerance)
            if ok:
                report.update(candidate_committed=True, reason='discrete_constraints_converged')
                return frame.positions(), report
            if step == iterations:
                report['reason'] = 'joint_constraint_budget_exhausted'; break
            frame.project_connections(); frame.project_structure()
            # Regenerate contacts after connection/structure moved the triangles.
            generated = contacts.project_all(frame) if hasattr(contacts, 'project_all') else contacts.scan(frame.positions())
            if generated['blocking_reason']:
                report['candidate_contact_audit'] = generated['audit']
                report['reason'] = generated['blocking_reason']; break
            for c in generated['constraints']:
                # Re-evaluate this linearized contact on the shared current
                # state; repeated features must not apply a stale penetration.
                value=sum(float(frame.point(v) @ g) for v,g in zip(c['ids'],c['gradients']))-c['offset']
                if value < 0: frame.project_scalar(c['ids'], c['gradients'], value)
            used = step+1
        return frame.source.copy(), report
    finally:
        report['iterations_used'] = used
        report['rolled_back'] = not report['candidate_committed']
        if not report['candidate_committed']: frame.x, frame.states = original
