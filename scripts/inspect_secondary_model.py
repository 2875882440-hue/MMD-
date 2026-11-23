"""Read-only character physics preflight. Not a physics simulation command."""
from pathlib import Path
import argparse,json,re,sys
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))
from motionbridge.secondary_motion import load_and_inspect

def main():
    parser=argparse.ArgumentParser(description='检查模型原生物理，尚不运行衣发仿真')
    parser.add_argument('--profile',default='rin')
    parser.add_argument('--output',type=Path,default=ROOT/'output/secondary_motion/rin-inspection.json')
    args=parser.parse_args()
    if not re.fullmatch('[a-z0-9_-]{1,32}',args.profile):parser.error('非法角色配置标识')
    output=args.output.resolve()
    if output.suffix.lower()!='.json' or not output.is_relative_to(ROOT/'output'):
        parser.error('报告必须输出到开发副本output内的JSON，禁止覆盖模型或配置')
    report=load_and_inspect(ROOT/'profiles/secondary_motion'/f'{args.profile}.json',ROOT)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'profile':report['profile_id'],'rigid_bodies':report['native_physics']['rigid_body_count'],
        'joints':report['native_physics']['joint_count'],'blocking_issues':report['blocking_issues'],
        'simulation_performed':False,'report':str(output)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
