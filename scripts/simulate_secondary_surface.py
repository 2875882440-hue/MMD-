"""Explicit standalone model-owned hair/cloth workflow; main mocap unchanged."""
from pathlib import Path
import argparse,json,os,re,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT))
from motionbridge.secondary_surface_config import surface_settings
from motionbridge.secondary_guard import actual_surface_gate

def validate_request(scene,output,profile,enabled,resume=False):
 scene=Path(scene).resolve();output=Path(output).resolve()
 if enabled is not True:raise ValueError('衣物/耦合发束是实验功能，必须显式--enable-surface')
 if not re.fullmatch('[a-z0-9_-]{1,32}',profile):raise ValueError('非法角色配置名')
 if not scene.is_relative_to(ROOT) or not scene.is_file() or scene.suffix.lower()!='.blend':raise ValueError('输入必须为开发副本内同角色blend')
 if not output.is_relative_to(ROOT/'output/secondary_motion') or (output.exists() and not resume):raise ValueError('只允许全新独立输出目录，或显式--resume已管理运行')
 return scene,output

def main():
 from motionbridge.secondary_pipeline import run_pipeline
 parser=argparse.ArgumentParser(description='保留主动作，分阶段缓存/恢复衣发计算；实际碰撞门限仍有效')
 parser.add_argument('--scene',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
 parser.add_argument('--profile',default='rin');parser.add_argument('--enable-surface',action='store_true')
 parser.add_argument('--resume',action='store_true');parser.add_argument('--no-cache',action='store_true');parser.add_argument('--compare',action='store_true')
 args=parser.parse_args();scene,out=validate_request(args.scene,args.output_dir,args.profile,args.enable_surface,resume=args.resume)
 result=run_pipeline(ROOT,scene,out,args.profile,resume=args.resume,use_cache=not args.no_cache,compare=args.compare)
 print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
 if not result['physics_result_published']:raise SystemExit(3)

if __name__=='__main__':main()
