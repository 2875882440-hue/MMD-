"""Compile explicitly selected native joint islands; never freeze dynamic parents."""
import copy
from .secondary_segmentation import _sha

def compile_articulated(plan, config):
    if config.get('schema_version')!=1 or config.get('model_sha256')!=plan['model_sha256']:
        raise ValueError('关节岛配置指纹或版本无效')
    ids=config.get('dynamic_rigid_ids',[])
    if not ids or len(set(ids))!=len(ids) or any(type(i)!=int or not 0<=i<len(plan['rigid_bodies']) for i in ids):
        raise ValueError('动态刚体编号无效')
    dynamic=set(ids);protected=set(plan['protected_bone_ids']);bodies={};joints=[]
    for i in ids:
        b=plan['rigid_bodies'][i]
        if b['native']['mode'] not in (1,2) or b['bone_id'] in protected or b['bone_id'] is None:
            raise ValueError('不能把保护骨或运动学刚体设为动态')
        bodies[i]=copy.deepcopy(b)
    touching=[j for j in plan['joints'] if set(j['rigid_ids'])&dynamic]
    if config.get('joint_ids')!=[j['id'] for j in touching]:raise ValueError('必须保留全部原生跨链连接，不能删环路')
    fixed=set()
    for j in touching:
        if any(i is None for i in j['rigid_ids']):raise ValueError('关节端点缺失')
        for i in set(j['rigid_ids'])-dynamic:
            b=plan['rigid_bodies'][i]
            if b['native']['mode']!=0 or b['bone_id'] not in protected:raise ValueError('动态父级遗漏，禁止隐式固定')
            fixed.add(i);bodies[i]=copy.deepcopy(b)
        parameters=j['blender']
        if any(parameters['linear_spring_import_value']+parameters['angular_spring_import_value']):raise ValueError('原生弹簧尚未适配')
        joints.append(copy.deepcopy(j))
    reachable=set(fixed)
    while True:
        old=set(reachable)
        for j in joints:
            if set(j['rigid_ids'])&reachable:reachable.update(j['rigid_ids'])
        if reachable==old:break
    if not dynamic<=reachable:raise ValueError('关节岛没有保护源锚点')
    return {'schema_version':1,'dynamic_rigid_ids':sorted(dynamic),'fixed_rigid_ids':sorted(fixed),
        'bodies':[bodies[i] for i in sorted(bodies)],'joints':joints,'config_sha256':_sha(config),
        'complete_physics':False,'simulation_ready':False,'individual_render_links_bound':False}
