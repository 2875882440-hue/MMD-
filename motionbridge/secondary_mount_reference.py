"""Explicit rest-rigid versus immutable source-skinned attachment references.

Source-skinned mode retains the INPUT deformation for this frame, then permits
only whole-part corrections. It is not a constant-shape rigid body over time.
Neither mode samples final clothing or a previous solver result.
"""
import numpy as np
from .secondary_surface import rigid_fit

REST_RIGID = 'rest_rigid'
SOURCE_SKINNED = 'source_skinned_attachment'


def mount_mode(row):
    mode = row.get('motion_mode', REST_RIGID)
    if mode not in (REST_RIGID, SOURCE_SKINNED):
        raise ValueError('Unknown attachment motion mode')
    if mode == SOURCE_SKINNED:
        evidence = row.get('motion_mode_evidence')
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError('Source-skinned attachment requires explicit reviewed evidence')
    return mode


def mount_reference(source, mapping, row):
    mode = mount_mode(row)
    ids = row['canonical_vertex_ids']
    static = mapping.rest[ids]
    _, rotation, translation = rigid_fit(static, source[ids])
    static_fit = static @ rotation.T + translation
    static_error = float(np.linalg.norm(static_fit-source[ids], axis=1).max())
    if mode == SOURCE_SKINNED:
        shape = np.array(source[ids], dtype=float, copy=True)
        center = shape.mean(0)
        rotation = np.eye(3)
        world_center = center.copy()
        world = shape.copy()
    else:
        shape = np.array(static, dtype=float, copy=True)
        center = shape.mean(0)
        world_center = rotation @ center + translation
        world = static_fit
    local = {v: shape[i]-center for i,v in enumerate(ids)}
    return rotation, world_center, local, world, {
        'motion_mode': mode,
        'reference_source_fit_error_m': float(np.linalg.norm(world-source[ids],axis=1).max()),
        'source_rigid_fit_error_m': static_error,
        'constant_static_shape_across_frames': mode == REST_RIGID,
        'reference_from_simulated_clothing': False,
    }
