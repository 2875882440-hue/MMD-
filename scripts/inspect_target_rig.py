"""Read-only inspection of user-owned Blender rigs; reports stay in MotionBridge."""
import bpy
import json
import sys
from pathlib import Path

args = sys.argv[sys.argv.index('--') + 1:]
output = Path(args[0])
output.mkdir(parents=True, exist_ok=True)
for path in args[1:]:
    bpy.ops.wm.open_mainfile(filepath=path, load_ui=False)
    scene = bpy.context.scene
    report = {'file': path, 'fps': scene.render.fps, 'frames': [scene.frame_start, scene.frame_end],
              'camera': scene.camera.name if scene.camera else None, 'objects': [], 'rigs': []}
    for obj in scene.objects:
        report['objects'].append({'name': obj.name, 'type': obj.type, 'hide_render': obj.hide_render,
            'parent': obj.parent.name if obj.parent else None,
            'matrix_world': [list(row) for row in obj.matrix_world],
            'modifiers': [{'name': m.name, 'type': m.type,
                'object': getattr(getattr(m, 'object', None), 'name', None),
                'show_render': m.show_render, 'filepath': getattr(m, 'filepath', None)} for m in obj.modifiers],
            'action': obj.animation_data.action.name if obj.animation_data and obj.animation_data.action else None})
        if obj.type == 'ARMATURE':
            bones = []
            for b in obj.pose.bones:
                bones.append({'name': b.name, 'parent': b.parent.name if b.parent else None,
                    'head': list(b.bone.head_local), 'tail': list(b.bone.tail_local),
                    'rest_matrix': [list(row) for row in b.bone.matrix_local],
                    'constraints': [{'name': c.name, 'type': c.type, 'mute': c.mute,
                        'influence': c.influence, 'target': getattr(getattr(c, 'target', None), 'name', None),
                        'subtarget': getattr(c, 'subtarget', None)} for c in b.constraints]})
            report['rigs'].append({'name': obj.name, 'bones': bones,
                'drivers': len(obj.animation_data.drivers) if obj.animation_data else 0,
                'nla': [t.name for t in obj.animation_data.nla_tracks] if obj.animation_data else []})
    report['images'] = [{'name': i.name, 'path': i.filepath, 'packed': bool(i.packed_file)} for i in bpy.data.images]
    (output / (Path(path).stem + '.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('INSPECTED', path, [(r['name'], len(r['bones'])) for r in report['rigs']], flush=True)
