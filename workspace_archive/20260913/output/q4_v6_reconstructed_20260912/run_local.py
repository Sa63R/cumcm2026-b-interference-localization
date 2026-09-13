"""Run a fresh, feedback-only V4/V5/V6 session on the reconstruction."""
import argparse
from pathlib import Path
import benchmark as bench
from jammers_local.core import Scenario


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',choices=['v4','v5','v6'],default='v6')
    parser.add_argument('--seed',default='q4-v6-user-demo')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be absent or empty; prior records are preserved')
    bench.verify_sources();bench.initialize(1.005)
    case=dict(key='local-demo',group='user_local',scenario=Scenario.generate(4,args.seed).as_dict())
    rows=bench.run_case((case,[args.method],str(args.output)))
    bench.dump(args.output/'summary.json',rows)
    row=rows[0]
    print(f'{args.method}: cleared {row["cleared_count"]}/{row["source_total"]}; '
          f'task {row["virtual_time_s"]:.3f}s; {row["seconds_per_source"]:.3f}s/source; '
          f'wall {row["wall_seconds"]:.3f}s; error={row["error"]}')
    raise SystemExit(1 if row['error'] else 0)
