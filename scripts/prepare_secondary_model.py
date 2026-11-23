"""Prepare a reusable model binding; no video, retarget or dynamics operation."""
from pathlib import Path
import argparse
import json
import re
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from motionbridge.secondary_plan import prepare


def main():
    parser = argparse.ArgumentParser(description='编译模型衣发分区、固定点和原生物理绑定；不运行仿真')
    parser.add_argument('--profile', default='rin')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch('[a-z0-9_-]{1,32}', args.profile):
        parser.error('非法角色配置标识')
    result = prepare(ROOT / 'profiles/secondary_motion' / f'{args.profile}.json', ROOT, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
