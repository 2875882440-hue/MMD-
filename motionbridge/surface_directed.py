"""Coherent surface-side escape for skinned flexible meshes (NumPy only).

Rest triangle orientation selects the escape side; the animated triangle field
transports it with the surface. Bones and recognition tracks are never changed.
This is bounded corrective geometry, not a physical cloth simulation.
"""
import numpy as np
from .surface_clearance import capsule_distance, project_surface, deficits
from .surface_topology import edge_changes


def surface_profile(rest, faces, rest_capsules):
    rest = np.asarray(rest, float)
    faces = np.asarray(faces)
    caps = np.asarray(rest_capsules, float)
    if (rest.ndim != 2 or rest.shape[1] != 3 or not len(rest)
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or faces.dtype.kind not in 'iu' or faces.min() < 0 or faces.max() >= len(rest)
            or caps.ndim != 3 or caps.shape[1:] != (2, 3) or not len(caps)
            or not np.isfinite(rest).all() or not np.isfinite(caps).all()):
        raise ValueError('Invalid directed surface profile')
    # Coincident back/front faces must use the same field even with opposite
    # winding. Only direction is welded; independent skin weights are retained.
    _, groups = np.unique(np.round(rest, 6), axis=0, return_inverse=True)
    triangle = rest[faces]
    normals = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    centers = triangle.mean(1)
    signs = np.array([np.where(np.sum(normals * capsule_distance(centers, *c)[1], axis=1) >= 0, 1., -1.) for c in caps])
    return faces, groups, signs


def surface_directions(points, profile):
    faces, groups, signs = profile
    triangle = points[faces]
    normals = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    corner_groups = groups[faces.ravel()]
    result = []
    for sign in signs:
        field = np.stack([np.bincount(corner_groups, weights=np.repeat(normals[:, k] * sign, 3),
                                     minlength=groups.max() + 1) for k in range(3)], axis=1)[groups]
        field /= np.maximum(np.linalg.norm(field, axis=1, keepdims=True), 1e-12)
        result.append(field)
    return np.asarray(result)


def project_directed(raw, candidate, caps, limits, directions, max_shift, iterations=2):
    p = np.asarray(candidate, float).copy()
    for _ in range(iterations):
        for c, ((a, b), minimum) in enumerate(zip(caps, limits)):
            distance, vector = capsule_distance(p, a, b)
            ids = np.flatnonzero(distance < minimum - 1e-8)
            if not len(ids):
                continue
            direction = directions[c, ids].copy()
            bad = np.linalg.norm(direction, axis=1) < .5
            direction[bad] = vector[ids][bad] / np.maximum(distance[ids][bad, None], 1e-12)
            lo = np.zeros(len(ids)); hi = np.full(len(ids), 2 * (max_shift + float(minimum.max())))
            for _ in range(16):
                mid = (lo + hi) / 2
                inside = capsule_distance(p[ids] + direction * mid[:, None], a, b)[0] < minimum[ids]
                lo = np.where(inside, mid, lo); hi = np.where(inside, hi, mid)
            p[ids] += direction * hi[:, None]
            delta = p[ids] - raw[ids]
            p[ids] = raw[ids] + delta * np.minimum(1, max_shift / np.maximum(np.linalg.norm(delta, axis=1), 1e-12))[:, None]
    return p


def relax_surface(raw, candidate, caps, limits, edges, max_shift, iterations=24, directions=None):
    """Preserve posed edge lengths while staying outside capsule proxies."""
    p = np.asarray(candidate, float).copy()
    i, j = edges.T
    degree = np.maximum(np.bincount(edges.ravel(), minlength=len(p)), 1)
    lengths = np.linalg.norm(raw[i] - raw[j], axis=1)
    for _ in range(iterations):
        vector = p[j] - p[i]
        current = np.linalg.norm(vector, axis=1)
        excess = current - np.clip(current, np.maximum(0, lengths * .90 - .0005), lengths * 1.10 + .0005)
        correction = vector * (excess / np.maximum(current, 1e-10))[:, None]
        move = np.stack([np.bincount(i, weights=correction[:, k], minlength=len(p))
                         - np.bincount(j, weights=correction[:, k], minlength=len(p))
                         for k in range(3)], axis=1) / degree[:, None]
        if directions is None:
            p = project_surface(raw, caps, limits, max_shift=max_shift, initial=p + .8 * move, iterations=2)
        else:
            delta = p + .8 * move - raw
            p = raw + delta * np.minimum(1, max_shift / np.maximum(np.linalg.norm(delta, axis=1), 1e-12))[:, None]
            p = project_directed(raw, p, caps, limits, directions, max_shift)
    return p


def directed_surface(raw, caps, limits, directions, edges, max_shift=.08):
    p = np.asarray(raw, float).copy()
    for _ in range(4):
        for c, ((a, b), minimum) in enumerate(zip(caps, limits)):
            distance, vector = capsule_distance(p, a, b)
            ids = np.flatnonzero(distance < minimum - 1e-7)
            if not len(ids):
                continue
            direction = directions[c, ids].copy()
            bad = np.linalg.norm(direction, axis=1) < .5
            direction[bad] = vector[ids][bad] / np.maximum(distance[ids][bad, None], 1e-12)
            lo = np.zeros(len(ids))
            hi = np.full(len(ids), 2 * (max_shift + float(minimum.max())))
            for _ in range(18):
                mid = (lo + hi) / 2
                inside = capsule_distance(p[ids] + direction * mid[:, None], a, b)[0] < minimum[ids]
                lo = np.where(inside, mid, lo)
                hi = np.where(inside, hi, mid)
            p[ids] += direction * hi[:, None]
            delta = p - raw
            p = raw + delta * np.minimum(1, max_shift / np.maximum(np.linalg.norm(delta, axis=1), 1e-12))[:, None]
    p = project_surface(raw, caps, limits, max_shift=max_shift, initial=p, iterations=5)
    return relax_surface(raw, p, caps, limits, edges, max_shift)


def smooth_corrections(raw, proposed, fps):
    """Symmetric five samples at 30fps; do not filter source movement."""
    radius = max(1, round(fps / 15))
    weights = np.r_[np.arange(1, radius + 2), np.arange(radius, 0, -1)].astype(float)
    weights /= weights.sum()
    padded = np.pad(proposed - raw, [(radius, radius), (0, 0), (0, 0)], mode='edge')
    return raw + sum(w * padded[k:k + len(raw)] for k, w in enumerate(weights))


def anticipate_corrections(raw, proposed, fps):
    """Spread contact magnitude before smoothing, retaining required peaks.

    Purely offline, geometry-only lookahead. Averaging a short contact away and
    restoring it with a hard projection can create a new impulse; preserve its
    magnitude envelope while smoothing the direction and approach/release.
    """
    delta = proposed - raw
    radius = max(1, round(fps * 2 / 15))
    magnitude = np.linalg.norm(delta, axis=-1, keepdims=True)
    padded = np.pad(magnitude, [(radius, radius), (0, 0), (0, 0)], mode='edge')
    envelope = magnitude.copy()
    for k in range(2 * radius + 1):
        np.maximum(envelope, padded[k:k + len(raw)], out=envelope)
    envelope = smooth_corrections(np.zeros_like(envelope), envelope, fps)
    direction = smooth_corrections(np.zeros_like(delta), delta, fps * 3)
    direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-12)
    return raw + direction * envelope


def track_metrics(raw, track, caps, limits, edges, fps):
    hits = edge_hits = 0
    depth = peak = energy = 0.
    for f, points in enumerate(track):
        penetration = deficits(points, caps[f], limits).max(0)
        changes = edge_changes(points, raw[f], edges)
        hits += int((penetration > .001).sum())
        depth = max(depth, float(penetration.max()))
        edge_hits += int((changes > .005).sum())
        peak = max(peak, float(changes.max(initial=0)))
        energy += float(np.sum(changes ** 2))
    correction = track - raw
    acceleration = np.linalg.norm(np.diff(correction, n=2, axis=0) * fps ** 2, axis=-1)
    actual_acceleration = np.linalg.norm(np.diff(track, n=2, axis=0) * fps ** 2, axis=-1)
    return {'vertex_frames_over_1mm': hits, 'max_penetration_m': depth,
            'edge_frames_over_5mm': edge_hits, 'max_edge_change_m': peak, 'edge_energy': energy,
            'maximum_shift_m': float(np.linalg.norm(correction, axis=-1).max()),
            'correction_acceleration_p99': float(np.percentile(acceleration, 99)) if acceleration.size else 0.,
            'correction_acceleration_max': float(acceleration.max()) if acceleration.size else 0.,
            'vertex_acceleration_p99': float(np.percentile(actual_acceleration, 99)) if actual_acceleration.size else 0.,
            'vertex_acceleration_max': float(actual_acceleration.max()) if actual_acceleration.size else 0.}


def select_surface_track(raw, baseline, candidate, caps, limits, edges, fps, max_shift):
    """Accept/reject a whole sequence, never splice opposite escape branches.

    Aggregate topology must improve, with no newly deep (>1mm) vertex contact
    and no greater maximum contact depth or actual vertex acceleration peak.
    The full diagnostics remain visible even when retaining the prior solver.
    """
    before = track_metrics(raw, baseline, caps, limits, edges, fps)
    after = track_metrics(raw, candidate, caps, limits, edges, fps)
    new_contacts = 0
    for f in range(len(raw)):
        old = deficits(baseline[f], caps[f], limits).max(0)
        new = deficits(candidate[f], caps[f], limits).max(0)
        new_contacts += int(((new > .001) & (old <= .001)).sum())
    failures = []
    if not np.isfinite(candidate).all(): failures.append('non_finite')
    if after['maximum_shift_m'] > max_shift + 5e-7: failures.append('shift_cap')
    if new_contacts: failures.append('new_deep_contacts')
    if after['max_penetration_m'] > before['max_penetration_m'] + 1e-5: failures.append('peak_penetration')
    for key in ['edge_frames_over_5mm', 'edge_energy', 'max_edge_change_m']:
        if after[key] > before[key] + 1e-7: failures.append(key)
    # Judge the geometry that is actually rendered. A correction can cancel
    # acceleration already present in skinning; its magnitude is diagnostic,
    # not a reliable proxy for the final motion. Keep both sets in the report.
    if after['vertex_acceleration_max'] > before['vertex_acceleration_max'] + .1:
        failures.append('peak_vertex_acceleration')
    if after['vertex_acceleration_p99'] > before['vertex_acceleration_p99'] + .1:
        failures.append('p99_vertex_acceleration')
    return (baseline.copy() if failures else candidate), {
        'enabled': True, 'method': 'rest_side_escape_temporal_topology_v1', 'gate_version': 'actual_vertex_motion_v2', 'accepted': not failures,
        'rejection_reasons': failures, 'max_vertex_shift_m': max_shift,
        'new_vertex_frames_over_1mm': new_contacts, 'before': before, 'candidate': after,
        'after': before if failures else after,
        'note': 'Whole-track fallback avoids seams and framewise branch switching. Actual vertex acceleration gates allow 0.1m/s^2 numerical/low-amplitude tolerance; correction acceleration is reported separately. Capsule vertices only; no triangle/self-collision guarantee. Source animation is not filtered.'}


def connected_surfaces(rest, edges):
    """Group all welded neighbours so an acceptance boundary cannot cut an edge."""
    _, groups = np.unique(np.round(rest, 6), axis=0, return_inverse=True)
    parents = np.arange(groups.max() + 1)
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    for a, b in groups[edges]:
        parents[root(a)] = root(b)
    return np.unique([root(i) for i in groups], return_inverse=True)[1]


def select_surface_components(raw, baseline, candidate, caps, limits, edges, rest, fps, max_shift):
    labels = connected_surfaces(rest, edges)
    final = baseline.copy()
    records = []
    for label in np.unique(labels):
        ids = np.flatnonzero(labels == label)
        mapping = np.full(len(rest), -1); mapping[ids] = np.arange(len(ids))
        local_edges = mapping[edges[labels[edges[:, 0]] == label]]
        if not len(local_edges):
            continue
        solved, report = select_surface_track(raw[:, ids], baseline[:, ids], candidate[:, ids], caps,
                                               limits[:, ids], local_edges, fps, max_shift)
        final[:, ids] = solved
        report.update(component=int(label), vertices=len(ids))
        records.append(report)
    final, report = select_surface_track(raw, baseline, final, caps, limits, edges, fps, max_shift)
    report['components'] = records
    report['accepted_components'] = sum(r['accepted'] for r in records)
    report['retained_components'] = sum(not r['accepted'] for r in records)
    return final, report


def coherent_surface_tracks(raw, baseline, caps, limits, edges, rest, faces, rest_caps, fps, max_shift=.12):
    raw = np.asarray(raw, float); baseline = np.asarray(baseline, float)
    caps = np.asarray(caps, float); limits = np.asarray(limits, float); edges = np.asarray(edges)
    profile = surface_profile(rest, faces, rest_caps)
    if (raw.ndim != 3 or raw.shape[1:] != np.asarray(rest).shape or len(raw) < 1
            or baseline.shape != raw.shape or caps.shape != (len(raw), len(rest_caps), 2, 3)
            or limits.shape != (len(rest_caps), raw.shape[1]) or edges.ndim != 2
            or edges.shape[1] != 2 or not len(edges) or edges.dtype.kind not in 'iu'
            or edges.min() < 0 or edges.max() >= raw.shape[1]
            or not all(np.isfinite(x).all() for x in [raw, baseline, caps, limits, fps, max_shift])
            or fps <= 0 or max_shift <= 0 or (limits < 0).any()):
        raise ValueError('Invalid coherent surface tracks')
    proposed = np.empty_like(raw)
    for f in range(len(raw)):
        proposed[f] = directed_surface(raw[f], caps[f], limits, surface_directions(raw[f], profile), edges, min(.08, max_shift))
        if f % 150 == 0: print('SURFACE_DIRECTED_PROPOSAL', f + 1, len(raw), flush=True)
    smoothed = anticipate_corrections(raw, proposed, fps)
    for f in range(len(raw)):
        p = project_surface(raw[f], caps[f], limits, max_shift=max_shift, initial=smoothed[f], iterations=5)
        proposed[f] = relax_surface(raw[f], p, caps[f], limits, edges, max_shift, iterations=16)
        if f % 150 == 0: print('SURFACE_DIRECTED_CONTACTS', f + 1, len(raw), flush=True)
    return select_surface_components(raw, baseline, proposed, caps, limits, edges, np.asarray(rest), fps, max_shift)
