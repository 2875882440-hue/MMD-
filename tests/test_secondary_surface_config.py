import pytest
from motionbridge.secondary_surface_config import surface_settings

def test_surface_defaults_are_model_owned_and_copy_safe():
 profile={'surface_dynamics':{'floor_z_m':-.1,'minimum_pose_guide':.2}}
 result=surface_settings(profile)
 assert result['floor_z_m']==-.1 and result['minimum_pose_guide']==.2
 assert surface_settings({})['minimum_pose_guide']==.03
 assert profile=={'surface_dynamics':{'floor_z_m':-.1,'minimum_pose_guide':.2}}

@pytest.mark.parametrize('bad',[{'body_gap_m':float('nan')},{'minimum_pose_guide':1.1},
 {'cloth_quality':2.5},{'body_materials':[]},{'body_gap_m':.5},{'not_a_setting':2}])
def test_invalid_physics_config_rejected_before_simulation(bad):
 with pytest.raises(ValueError):surface_settings({'surface_dynamics':bad})
