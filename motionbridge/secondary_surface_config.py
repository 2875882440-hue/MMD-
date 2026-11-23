"""Model-owned cloth settings. Never infer physics material from the video."""
import math

DEFAULTS={
 'method':'native_cloth_actual_triangles_v1','warmup_seconds':5.,
 'body_materials':['肌','肌2','襪','顏','靴'],
 'ornament_bone_tokens':['髮飾','髪飾'],'maximum_rigid_extent_m':.16,
 'cloth_quality':8,'mass_kg':.05,'tension_stiffness':40.,'compression_stiffness':40.,
 'shear_stiffness':25.,'bending_stiffness':1.,'air_damping':5.,'material_damping':10.,
 'pin_stiffness':3.,'minimum_pose_guide':.03,'collision_quality':6,
 'body_gap_m':.002,'self_gap_m':.001,'impulse_clamp':.2,'self_impulse_clamp':.1,
 'floor_z_m':0.,'anchor_band_m':.015,'max_displacement_m':2.,
 'contact_pin_collar':4,'contact_edge_iterations':48,'contact_max_strain':.05,
 'attachment_bind_scale':.2,'prewarm_p95_limit_m_s':.02,'prewarm_max_limit_m_s':.1,
 'maximum_extra_edge_strain':.25,'shape_projection_iterations':256,'shape_projection_passes':4,
 'attachment_rotation_prior':1.
}

def surface_settings(profile):
 settings={**DEFAULTS,**profile.get('surface_dynamics',{})}
 if set(settings)-set(DEFAULTS):raise ValueError('未知衣物物理配置项')
 if settings['method']!=DEFAULTS['method']:raise ValueError('不支持的衣物求解器')
 for key,value in settings.items():
  if key in ('method','body_materials','ornament_bone_tokens'):continue
  if type(value) not in (int,float) or not math.isfinite(value):raise ValueError(f'物理配置必须有限: {key}')
  if key!='floor_z_m' and value<=0:raise ValueError(f'物理配置必须为正: {key}')
 for key in ('cloth_quality','collision_quality'):
  if type(settings[key]) is not int or not 1<=settings[key]<=80:raise ValueError('求解质量必须是1至80的整数')
 if not 0<settings['minimum_pose_guide']<=1:raise ValueError('衣物引导权重必须在0到1之间')
 if settings['body_gap_m']>.05 or settings['self_gap_m']>.05:raise ValueError('碰撞间距过大')
 for key in ('contact_pin_collar','contact_edge_iterations','shape_projection_iterations'):
  if type(settings[key]) is not int or not 1<=settings[key]<=256:raise ValueError('Contact iterations/collar must be bounded positive integers')
 if settings['contact_max_strain']>=1 or settings['attachment_bind_scale']>.5:raise ValueError('Invalid contact strain/binding scale')
 if settings['prewarm_p95_limit_m_s']>settings['prewarm_max_limit_m_s']:raise ValueError('Invalid prewarm velocity limits')
 if settings['maximum_extra_edge_strain']>=1:raise ValueError('Invalid garment deformation budget')
 if settings['attachment_rotation_prior']>10:raise ValueError('Invalid attachment rotation prior')
 if type(settings['shape_projection_passes']) is not int or not 1<=settings['shape_projection_passes']<=8:raise ValueError('Invalid bounded shape projection passes')
 for key in ('body_materials','ornament_bone_tokens'):
  values=settings[key]
  if not isinstance(values,list) or not values or any(not isinstance(v,str) or not v.strip() for v in values):
   raise ValueError('身体材质与饰品骨名需要明确配置')
 return settings
