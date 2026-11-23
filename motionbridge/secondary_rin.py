"""Named Rin adapter, independent of the existing Nahida algorithms."""
from pathlib import Path
import json
from .retarget import run_script,character_timeouts

def profile(root,blender):
 asset=Path(root)/'assets/characters/rin/Rin_Model.blend'
 if not asset.is_file():return []
 return [{'id':'rin','label':'远坂凛 · 已配置 MMD 骨架','ready':bool(blender),'reason':'' if blender else '未找到Blender',
  'scene_path':str(asset),'weights_scene_path':str(asset),'surface_clearance':False}]

def export_character(root,blender,profile,motion_path,output,cancelled,progress):
 root=Path(root);output=Path(output);motion=json.loads(Path(motion_path).read_text(encoding='utf-8'))
 bake,audit=character_timeouts(len(motion['frames']),False)
 progress(35,'将现有动作转接到远坂凛，衣发随后独立处理')
 run_script(blender,root/'scripts/retarget_secondary_rin.py',[profile['scene_path'],motion_path,profile['weights_scene_path'],output,'false'],output/'character_export.log',cancelled,timeout=bake)
 progress(85,'重开远坂凛工程，验证动作转接')
 run_script(blender,root/'scripts/audit_secondary_rin.py',[output],output/'character_audit.log',cancelled,timeout=audit)
 report=json.loads((output/'saved_project_audit.json').read_text(encoding='utf-8'))
 if not report.get('passed'):raise ValueError('远坂凛工程回读未通过；旧任务保持')
 return report
