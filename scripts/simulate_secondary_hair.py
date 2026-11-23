"""Opt-in experimental hair solve for an existing same-character Blender scene."""
from pathlib import Path
import argparse,json,re,subprocess,sys,time,uuid
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True


def build_command(profile,scene,output,enabled,blender):
    scene=Path(scene).resolve();output=Path(output).resolve();blender=Path(blender)
    if enabled is not True:raise ValueError('请显式添加--enable-hair，当前为实验性功能')
    if not re.fullmatch('[a-z0-9_-]{1,32}',profile):raise ValueError('非法角色标识')
    if not scene.is_relative_to(ROOT) or scene.suffix.lower()!='.blend' or not scene.is_file():
        raise ValueError('源工程必须为本副本内存在的blend文件')
    if not output.is_relative_to(ROOT/'output/secondary_motion') or output.exists():
        raise ValueError('输出必须为output/secondary_motion内的新子目录')
    if not blender.is_file():raise ValueError('未找到Blender，请检查config.json')
    return [str(blender),'--background','--factory-startup','--disable-autoexec','--threads','4',
            '--python-exit-code','23','--python',str(ROOT/'scripts/solve_secondary_hair_blender.py'),
            '--','--profile',profile,'--scene',str(scene),'--output-dir',str(output),'--enable-hair']


def main():
    parser=argparse.ArgumentParser(description='只对已配置角色的现有工程施加头发力学，不重算识别或身体动作')
    parser.add_argument('--profile',default='rin');parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--enable-hair',action='store_true')
    args=parser.parse_args()
    config=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
    command=build_command(args.profile,args.scene,args.output_dir,args.enable_hair,config.get('blender_path',''))
    log=ROOT/'logs'/f'hair-{uuid.uuid4().hex[:12]}.log';log.parent.mkdir(exist_ok=True)
    print(json.dumps({'log':str(log),'output':str(args.output_dir),'method':'spring_bones_v1'},ensure_ascii=False),flush=True)
    with log.open('x',encoding='utf-8') as stream:
        process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        try:
            code=process.wait(timeout=3600)
        except (KeyboardInterrupt,subprocess.TimeoutExpired):
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            raise
    if code:raise SystemExit(f'头发处理失败（{code}），保留失败目录及日志：{log}')
    report=json.loads((args.output_dir/'physics-validation.json').read_text(encoding='utf-8'))
    if not report['passed'] or report['simulation_performed'] is not True:
        raise SystemExit('物理回归未通过，不作为成功结果发布')
    print(json.dumps({'passed':True,'hair_bones':report['hair_bones'],'protected_bones':report['protected_bones'],
        'source_actions_unchanged':report['source_actions_unchanged'],'output':str(args.output_dir),
        'hair_tip_change_p95_m':report['hair_tip_change_p95_m']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
