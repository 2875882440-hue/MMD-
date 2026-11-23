"""Read-only step-six readiness and archived evidence review; no simulation."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT))
from motionbridge.secondary_plan import compile_profile
from motionbridge.secondary_regression import regression_readiness,review_archived_case,markdown_report
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();out=args.output.resolve()
if not out.is_relative_to(ROOT/'validation') or out==ROOT/'validation' or out.exists():raise ValueError('Write only a new review directory under validation')
profile=json.loads((ROOT/'profiles/secondary_motion/rin.json').read_text(encoding='utf8'))
plan=compile_profile(profile,ROOT)
report={'schema_version':1,'profile':'rin','scope':'current_readiness_and_archived_reports_only',
    'readiness':regression_readiness(plan),'historical_cases':{},'new_native_regression_completed':False,
    'publication_accepted':False,'automatic_retry':False}
for name in ('xu-jiaqi90','momo-full'):
    case=ROOT/'output/secondary_motion/repair'/name
    report['historical_cases'][name]=review_archived_case(ROOT,case,case/'result-repair2-stable')
out.mkdir(parents=True,exist_ok=False)
(out/'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
(out/'REPORT.md').write_text(markdown_report(report),encoding='utf8')
print(json.dumps({'report':str(out/'REPORT.md'),'native_execution_allowed':False,
                  'historical_gates':{k:v['numeric_regression_passed'] for k,v in report['historical_cases'].items()}},ensure_ascii=False))
sys.exit(3)  # Explicit blocked/not-accepted status, not a successful native run.
