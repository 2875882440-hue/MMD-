"""Immutable per-frame assembly reference; never inferred from simulated cloth.

Only authored zero-distance connections propagate a rigid reference. Cloth
outside those classes retains source-pose targets. This is a discrete reference
policy, not persistent material strain or a complete feasibility proof.
"""
from collections import defaultdict, deque
import numpy as np


def build_reference(frame, rigid_reference):
    parent = {}; adjacency = defaultdict(set)
    canonical = frame.mapping.canonical_vertex
    def find(raw):
        v = int(canonical[raw]); parent.setdefault(v, v)
        while parent[v] != v:
            parent[v] = parent[parent[v]]; v = parent[v]
        return v
    pairs = frame.stitches + [p for m in frame.mounts for p in m['anchors']]
    for a, b in pairs:
        ca,cb=int(canonical[a]),int(canonical[b]); adjacency[ca].add(cb); adjacency[cb].add(ca)
        a, b = find(a), find(b); parent[max(a,b)] = min(a,b)
    for v in frame.owner: find(v)
    groups = defaultdict(list)
    for v in list(parent): groups[find(v)].append(v)
    reference = frame.source.copy(); binding = {}; issues = []
    for ids in groups.values():
        rigid = sorted(v for v in ids if v in frame.owner)
        if not rigid:
            # Authored stitch equality requires one material reference for the
            # entire equivalence class, not contradictory source-edge lengths.
            locked = [v for v in ids if frame.weights[v] == 0]
            if locked:
                target = frame.source[locked[0]].copy()
                spread = max(float(np.linalg.norm(frame.source[v]-target)) for v in locked)
                if spread > 2e-6:
                    issues.append({'reason':'incompatible_immovable_stitch_reference',
                                   'vertices':sorted(ids),'error_m':spread})
                    continue
            else:
                masses = 1. / frame.weights[ids]
                target = np.average(frame.source[ids], axis=0, weights=masses)
            reference[ids] = target
            continue
        owners = {frame.owner[v] for v in rigid}
        if len(owners) != 1:
            issues.append({'reason':'multiple_rigid_reference_owners','vertices':sorted(ids)})
            continue
        anchor = rigid[0]; pid = frame.owner[anchor]
        depth={anchor:0};queue=deque([anchor])
        while queue:
            current=queue.popleft()
            for v in adjacency[current]:
                if v not in depth:depth[v]=depth[current]+1;queue.append(v)
        spread = max(float(np.linalg.norm(frame.local[pid][v]-frame.local[pid][anchor])) for v in rigid)
        if spread > 2e-6:
            issues.append({'reason':'distinct_rigid_points_forced_coincident','vertices':sorted(ids),'error_m':spread})
            continue
        for v in ids:
            # Preserve each actual rigid point, even within numerical tolerance.
            ref_v = v if v in frame.owner else anchor
            reference[v] = rigid_reference[ref_v]
            binding[v] = (pid, ref_v, 0 if v in frame.owner else depth[v])
    return reference, binding, issues


def check_reference(frame, tolerance):
    issues = list(frame.reference_issues); fixed = set(map(int, frame.fixed))
    for v in fixed:
        error = float(np.linalg.norm(frame.reference[v]-frame.source[v]))
        if error > tolerance:
            issues.append({'reason':'reference_moves_protected_point','vertex':v,'error_m':error})
    same_body = 0; changes = {}; source_differences = []
    for kind, edges, targets in [('edge',frame.edges,frame.edge_lengths),('bend',frame.bends,frame.bend_lengths)]:
        source_lengths = frame.lengths(frame.source, edges)
        deltas = np.abs(targets-source_lengths)
        changes[kind] = {'count':int(np.sum(deltas > tolerance)), 'maximum_m':float(deltas.max(initial=0))}
        for index, (a,b) in enumerate(edges):
            a,b=int(a),int(b); left=frame.reference_binding.get(a); right=frame.reference_binding.get(b)
            if not left or not right or left[0] != right[0]: continue
            same_body += 1; pid = left[0]
            length = float(np.linalg.norm(frame.local[pid][left[1]]-frame.local[pid][right[1]]))
            slack=(left[2]+right[2])*tolerance
            error = abs(length-float(targets[index]))
            row={'kind':kind,'vertices':[a,b],'rigid_part':pid,'invariant_m':length,
                 'target_m':float(targets[index]),'source_m':float(source_lengths[index]),'attachment_slack_m':slack}
            if abs(length-source_lengths[index]) > tolerance: source_differences.append(row)
            if error > tolerance+slack: issues.append(dict(row,reason='rigid_invariant_target_conflict',error_m=error))
            if kind == 'edge' and source_lengths[index] > 5e-6 and abs(length-source_lengths[index]) > .25*source_lengths[index]+slack:
                issues.append(dict(row,reason='rigid_invariant_exceeds_source_deformation_gate'))
    return {'passed':not issues,'policy':'source_pose_with_authored_rigid_attachment_reference_v1',
            'complete_feasibility_proof':False,'issues':issues,'same_body_constraints_checked':same_body,
            'target_changes_from_source':changes,'rigid_source_differences':source_differences,
            'source_deformation_gate_preserved':True,'prediction_used_in_reference':False}
