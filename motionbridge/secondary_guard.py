"""Whole-track actual-surface regression gate, independent of model and clip."""
import math

def attachment_repair_gate(audit):
 """Compare with input before, never a worse previous experimental result."""
 gate=actual_surface_gate(audit);reasons=list(gate['rejection_reasons'])
 for key in ('accessory_body_crossing_pairs','accessory_cloth_crossing_pairs','accessory_hair_crossing_pairs'):
  if key not in audit['before'] or key not in audit['after']:reasons.append('attachment_contacts_unverified')
 if audit['after']['body_cloth_crossing_pairs']>=audit['before']['body_cloth_crossing_pairs']:
  reasons.append('body_cloth_not_strictly_improved')
 gate.update(accepted=not reasons,rejection_reasons=list(dict.fromkeys(reasons)),reference='captured_input_before',
  requirement='all three accessory channels nonworsening AND body/cloth strictly reduced; original protection/floor/hair gates retained')
 return gate

def prewarm_stop_gate(report,settings):
 """Reject before expensive hair/assembly work when warmup is not ready."""
 warm=report.get('prewarm_convergence',{});values=[warm.get('maximum_frame_p95_m_s'),warm.get('maximum_vertex_speed_m_s')]
 valid=all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in values)
 passed=(warm.get('passed') is True and type(warm.get('trailing_frames')) is int and warm['trailing_frames']>0 and valid
  and values[0]<=settings['prewarm_p95_limit_m_s'] and values[1]<=settings['prewarm_max_limit_m_s'])
 return {'accepted':bool(passed),'rejection_reasons':[] if passed else ['cloth_prewarm_unsettled_or_unverified'],
  'prewarm_convergence':warm,'stop_before':'hair_and_surface_assembly' if not passed else None}

def actual_surface_gate(audit):
 before,after=audit['before'],audit['after'];reasons=[]
 if audit.get('protection_passed') is not True:reasons.append('primary_motion_or_data_changed')
 if before['samples']!=after['samples'] or before['samples']<=0:reasons.append('sample_grid_mismatch')
 for key in ('body_cloth_crossing_pairs','body_hair_crossing_pairs','cloth_hair_crossing_pairs',
             'dynamic_floor_frames_over_1mm','dynamic_floor_vertex_samples_over_1mm'):
  old,new=before[key],after[key]
  if not isinstance(new,(int,float)) or not math.isfinite(new) or new<0:raise ValueError('无效的实际表面指标')
  if new>old:reasons.append(key)
 if not math.isfinite(after['dynamic_floor_max_depth_m']):raise ValueError('无效地面深度')
 # Old stored audits remain readable. New audits must measure every attachment
 # channel; fewer cloth crossings cannot conceal a worse metal/body contact.
 attachment_keys=('accessory_body_crossing_pairs','accessory_cloth_crossing_pairs','accessory_hair_crossing_pairs')
 if audit.get('attachment_audit_required') or any(k in before or k in after for k in attachment_keys):
  for key in attachment_keys:
   if key not in before or key not in after:
    reasons.append('attachment_contacts_unverified');continue
   old,new=before[key],after[key]
   if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in (old,new)):raise ValueError('Invalid attachment contact metric')
   if new>old:reasons.append(key)
 if after['dynamic_floor_max_depth_m']>before['dynamic_floor_max_depth_m']+1e-5:reasons.append('dynamic_floor_max_depth_m')
 return {'accepted':not reasons,'rejection_reasons':reasons,
         'scope':'whole-track summed proper crossings per channel and floor metrics; not every-frame or zero-collision proof'}
