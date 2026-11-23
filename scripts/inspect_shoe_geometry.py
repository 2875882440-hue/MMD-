import bpy,json,sys
from pathlib import Path
bpy.ops.wm.open_mainfile(filepath=sys.argv[sys.argv.index('--')+1],load_ui=False)
rig=bpy.data.objects['Nahida_arm']
rig.data.pose_position='REST';bpy.context.view_layer.update()
report=[]
for obj in bpy.context.scene.objects:
    if obj.type!='MESH' or not obj.name.startswith('Nahida') or 'contour' in obj.name:continue
    groups={g.index:g.name for g in obj.vertex_groups}
    indices=[]
    for v in obj.data.vertices:
        if any(('足首' in groups.get(g.group,'') or 'つま先' in groups.get(g.group,'')) and g.weight>.1 for g in v.groups):indices.append(v.index)
    report.append({'name':obj.name,'vertices':len(obj.data.vertices),'foot_vertices':len(indices),'hide':obj.hide_render,
        'min_z':min((obj.matrix_world@obj.data.vertices[i].co).z for i in indices) if indices else None,
        'modifiers':[(m.type,m.show_render) for m in obj.modifiers]})
print('SHOE_GEOMETRY',json.dumps(report,ensure_ascii=False),flush=True)
