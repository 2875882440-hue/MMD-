import copy
import pytest
from motionbridge.secondary_guard import actual_surface_gate

@pytest.fixture
def audit():
 row=dict(samples=1799,body_cloth_crossing_pairs=100,body_hair_crossing_pairs=100,cloth_hair_crossing_pairs=100,
          dynamic_floor_frames_over_1mm=5,dynamic_floor_vertex_samples_over_1mm=25,dynamic_floor_max_depth_m=.02)
 return dict(protection_passed=True,before=row,after=copy.deepcopy(row))

def test_gate_accepts_preserved_source_but_does_not_claim_clean(audit):
 gate=actual_surface_gate(audit);assert gate['accepted'] and 'not every-frame' in gate['scope']

@pytest.mark.parametrize('key',['body_hair_crossing_pairs','cloth_hair_crossing_pairs','body_cloth_crossing_pairs'])
def test_one_contact_channel_must_not_hide_regression_in_another(audit,key):
 audit['after']['body_cloth_crossing_pairs']=0;audit['after'][key]=101
 assert key in actual_surface_gate(audit)['rejection_reasons']

def test_floor_and_primary_protection_cannot_be_traded_for_fewer_crossings(audit):
 audit['after']['body_cloth_crossing_pairs']=0;audit['after']['dynamic_floor_max_depth_m']+=.001
 audit['protection_passed']=False
 assert set(actual_surface_gate(audit)['rejection_reasons'])=={'dynamic_floor_max_depth_m','primary_motion_or_data_changed'}

def test_different_sample_grids_and_invalid_measurements_fail(audit):
 audit['after']['samples']=900
 assert 'sample_grid_mismatch' in actual_surface_gate(audit)['rejection_reasons']
 audit['after']['body_cloth_crossing_pairs']=float('nan')
 with pytest.raises(ValueError):actual_surface_gate(audit)
