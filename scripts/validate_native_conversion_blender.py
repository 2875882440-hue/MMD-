"""Independent comparison against installed MMD Tools import, no assembly/sim."""
from pathlib import Path
import argparse, importlib.util, json, sys
import bpy

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--mmd-tools', type=Path, required=True)
parser.add_argument('--dependencies', type=Path, required=True)
parser.add_argument('--plan', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
assert args.output.resolve().is_relative_to(ROOT/'validation/secondary-motion-dev2')
assert not args.output.exists()
sys.dont_write_bytecode = True
sys.path.insert(0, str(args.dependencies))
spec = importlib.util.spec_from_file_location('mmd_tools', args.mmd_tools/'__init__.py',
                                             submodule_search_locations=[str(args.mmd_tools)])
plugin = importlib.util.module_from_spec(spec)
sys.modules['mmd_tools'] = plugin; spec.loader.exec_module(plugin); plugin.register()
plan = json.loads(args.plan.read_text(encoding='utf-8'))
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mmd_tools.import_model(filepath=str(ROOT/'assets/characters/rin/TohsakaRin.pmx'), scale=plan['units']['scale'],
                              types={'ARMATURE','PHYSICS'}, clean_model=False, rename_bones=False)
if bpy.context.scene.rigidbody_world:
    bpy.context.scene.rigidbody_world.enabled = False
rigids = [o for o in bpy.context.scene.objects if o.mmd_type == 'RIGID_BODY']
joints = [o for o in bpy.context.scene.objects if o.mmd_type == 'JOINT']
errors = {}


def check(key, actual, expected, tolerance=2e-6):
    if isinstance(expected, (list, tuple)):
        error = max((abs(a-b) for a,b in zip(actual,expected)), default=0)
    else:
        error = abs(actual-expected)
    errors[key] = max(errors.get(key,0),error)
    assert error <= tolerance, (key, error, actual, expected)


# MMD Tools MoveObject.set_index writes a three digit source index prefix.
rigids.sort(key=lambda o:int(o.name[:3])); joints.sort(key=lambda o:int(o.name[:3]))
assert len(rigids) == len(plan['rigid_bodies']) == 344
assert len(joints) == len(plan['joints']) == 500
for obj, row in zip(rigids, plan['rigid_bodies']):
    p = row['blender']; assert obj.mmd_rigid.name_j == row['name']
    check('rigid_position_m', obj.location, p['position_m'])
    check('rigid_euler_rad', obj.rotation_euler, p['rotation_rad'])
    assert obj.rotation_mode == p['rotation_order']
    shape = p['shape']
    size = list(obj.mmd_rigid.size)
    if shape['kind'] == 'box': check('box_half_extents_m',size,shape['half_extents_m'])
    else:
        check('radius_m',size[0],shape['radius_m'])
        if shape['kind'] == 'capsule': check('capsule_cylinder_m',size[1],shape['cylinder_height_m'])
    check('mass_import_value',obj.rigid_body.mass,p['mass_import_value'])
    check('linear_damping',obj.rigid_body.linear_damping,p['linear_damping'])
    check('angular_damping',obj.rigid_body.angular_damping,p['angular_damping'])
    check('friction',obj.rigid_body.friction,p['friction'])
    check('bounce',obj.rigid_body.restitution,p['bounce'])
    assert obj.mmd_rigid.collision_group_number == p['collision_group']
    assert [i for i,allow in enumerate(obj.mmd_rigid.collision_group_mask) if allow] == p['allowed_groups']
for obj,row in zip(joints,plan['joints']):
    p=row['blender']; assert obj.mmd_joint.name_j == row['name']
    check('joint_position_m',obj.location,p['position_m'])
    check('joint_euler_rad',obj.rotation_euler,p['rotation_rad'])
    assert obj.rotation_mode == p['rotation_order']
    constraint=obj.rigid_body_constraint
    for name,prefix,suffix in [('linear_min_m','lin','lower'),('linear_max_m','lin','upper'),
                               ('angular_min_rad','ang','lower'),('angular_max_rad','ang','upper')]:
        check(name,[getattr(constraint,f'limit_{prefix}_{axis}_{suffix}') for axis in 'xyz'],p[name])
    check('linear_spring',obj.mmd_joint.spring_linear,p['linear_spring_import_value'])
    check('angular_spring',obj.mmd_joint.spring_angular,p['angular_spring_import_value'])
    for actual,index in zip((constraint.object1,constraint.object2),row['rigid_ids']):
        assert actual == (rigids[index] if index is not None else None)
report={'passed':True,'reference':'MMD Tools 4.5.14 real Blender import', 'reference_path':str(args.mmd_tools),
        'rigids_checked':len(rigids),'joints_checked':len(joints),'maximum_errors':errors,
        'simulation_performed':False,'note':'只导入原生参数，不执行物理装配、预热、帧推进或保存源文件。'}
args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('NATIVE_CONVERSION_PASSED',json.dumps(report,ensure_ascii=False),flush=True)
