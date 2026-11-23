import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from motionbridge.surface_clearance import project_surface, deficits
from motionbridge.surface_topology import edge_changes
from motionbridge.surface_directed import surface_profile, surface_directions, directed_surface, smooth_corrections, anticipate_corrections, select_surface_track, select_surface_components, coherent_surface_tracks


def sheet():
    raw = np.array([[-.004, 0, .4], [.004, 0, .4], [-.004, 0, .5], [.004, 0, .5]])
    rest = raw + [0, .06, 0]
    caps = np.array([[[0., 0, 0], [0, 0, 1]]])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    edges = np.array([[0, 1], [1, 2], [2, 0], [1, 3], [3, 2]])
    limits = np.full((1, 4), .035)
    return raw, rest, caps, faces, edges, limits


def test_same_side_escape_repairs_opposite_radial_split_without_changing_input():
    raw, rest, caps, faces, edges, limits = sheet()
    original = raw.copy()
    old = project_surface(raw, caps, limits)
    result = directed_surface(raw, caps, limits, surface_directions(raw, surface_profile(rest, faces, caps)), edges)
    assert np.array_equal(raw, original)
    assert (result[:, 1] > .03).all()
    assert deficits(result, caps, limits).max() < 1e-7
    assert edge_changes(result, raw, edges).max() < edge_changes(old, raw, edges).max() * .1
    assert np.linalg.norm(result - raw, axis=1).max() <= .08 + 1e-10


def test_winding_and_rigid_transform_do_not_change_escape_side():
    raw, rest, caps, faces, edges, limits = sheet()
    solve = lambda p, r, c, f: directed_surface(p, c, limits, surface_directions(p, surface_profile(r, f, c)), edges)
    expected = solve(raw, rest, caps, faces)
    assert np.allclose(solve(raw, rest, caps, faces[:, ::-1]), expected)
    rotation = Rotation.from_euler('xyz', [20, 30, 40], degrees=True).as_matrix()
    t = np.array([2., -3., 4.])
    actual = solve(raw @ rotation.T + t, rest @ rotation.T + t, caps @ rotation.T + t, faces)
    assert np.allclose(actual, expected @ rotation.T + t, atol=1e-7)


def test_clear_sheet_unchanged_and_too_small_budget_is_bounded():
    raw, rest, caps, faces, edges, limits = sheet()
    profile = surface_profile(rest, faces, caps)
    result = directed_surface(rest, caps, limits, surface_directions(rest, profile), edges)
    assert np.array_equal(result, rest)
    limited = directed_surface(raw, caps, limits, surface_directions(raw, profile), edges, max_shift=.005)
    assert np.linalg.norm(limited - raw, axis=1).max() <= .005 + 1e-10
    assert deficits(limited, caps, limits).max() > .01


def test_temporal_filter_only_changes_correction_not_source_translation():
    raw = np.zeros((9, 4, 3)); raw[:, :, 0] = np.arange(9)[:, None] ** 2
    proposed = raw.copy(); proposed[4, :, 1] = .06
    actual = smooth_corrections(raw, proposed, 30)
    assert np.array_equal(actual[:, :, 0], raw[:, :, 0])
    assert actual[:, :, 1].max() < .03
    assert np.linalg.norm(np.diff(actual - raw, n=2, axis=0)).max() < np.linalg.norm(np.diff(proposed - raw, n=2, axis=0)).max()


def test_invalid_profile_fails_before_indexing():
    _, rest, caps, faces, _, _ = sheet()
    faces[0, 0] = 99
    with pytest.raises(ValueError, match='profile'):
        surface_profile(rest, faces, caps)


def test_sequence_guard_rejects_new_contact_or_temporal_spike_without_frame_splicing():
    raw, rest, caps, faces, edges, limits = sheet()
    tracks = np.tile(rest, (9, 1, 1)); cs = np.tile(caps, (9, 1, 1, 1))
    bad = tracks.copy(); bad[4] = raw
    result, report = select_surface_track(tracks, tracks, bad, cs, limits, edges, 30, .08)
    assert np.array_equal(result, tracks)
    assert not report['accepted'] and 'new_deep_contacts' in report['rejection_reasons']
    bad = tracks.copy(); bad[4, :, 1] += .02
    result, report = select_surface_track(tracks, tracks, bad, cs, limits, edges, 30, .08)
    assert np.array_equal(result, tracks) and 'peak_vertex_acceleration' in report['rejection_reasons']


def test_complete_sequence_repairs_static_split_and_validates_inputs():
    raw, rest, caps, faces, edges, limits = sheet()
    original = np.tile(raw, (5, 1, 1)); baseline = np.tile(project_surface(raw, caps, limits), (5, 1, 1))
    cs = np.tile(caps, (5, 1, 1, 1))
    result, report = coherent_surface_tracks(original, baseline, cs, limits, edges, rest, faces, caps, 30)
    assert report['accepted'] and report['after']['edge_energy'] < report['before']['edge_energy']
    assert np.allclose(result, result[:1]) and np.array_equal(original, np.tile(raw, (5, 1, 1)))
    with pytest.raises(ValueError, match='tracks'):
        coherent_surface_tracks(original, baseline, cs, limits, edges, rest, faces, caps, 0)


def test_contact_envelope_retains_required_peak_but_softens_approach_and_release():
    raw = np.zeros((31, 4, 3)); raw[:, :, 0] = np.arange(31)[:, None] * .1
    candidate = raw.copy(); candidate[15, :, 1] = .06
    result = anticipate_corrections(raw, candidate, 30)
    assert np.array_equal(result[:, :, 0], raw[:, :, 0])
    assert np.allclose(result[15, :, 1], .06)
    assert np.all(result[12, :, 1] > 0) and np.all(result[18, :, 1] > 0)
    assert np.linalg.norm(result - raw, axis=-1).max() <= .06 + 1e-10
    assert np.linalg.norm(np.diff(result - raw, n=2, axis=0), axis=-1).max() < .02
    assert np.array_equal(anticipate_corrections(raw, raw, 30), raw)


def test_component_acceptance_keeps_failed_piece_without_splicing_connected_vertices():
    raw, rest, caps, faces, edges, limits = sheet()
    other = raw + [.3, 0, 0]
    all_raw = np.tile(np.concatenate([raw, other]), (5, 1, 1))
    old = all_raw.copy(); old[:, :4, 1] = .06
    old[:, 4:, 1] = [.018, -.018, .018, -.018]
    candidate = all_raw.copy()
    all_edges = np.concatenate([edges, edges + 4]); all_rest = np.concatenate([rest, other])
    cs = np.tile(caps, (5, 1, 1, 1)); all_limits = np.tile(limits, (1, 2))
    result, report = select_surface_components(all_raw, old, candidate, cs, all_limits, all_edges, all_rest, 30, .12)
    assert report['accepted'] and report['accepted_components'] == 1 and report['retained_components'] == 1
    assert np.array_equal(result[:, :4], old[:, :4])
    assert np.array_equal(result[:, 4:], candidate[:, 4:])
