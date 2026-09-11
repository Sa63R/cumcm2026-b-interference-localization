"""Warm-process CPU inference timing on already opened development cases only."""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--spec', type=Path, required=True)
parser.add_argument('--freeze-record', type=Path, required=True)
parser.add_argument('--environment', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--repeats', type=int, default=2)
args = parser.parse_args()
if sys.flags.optimize:
    raise RuntimeError('Do not run audited timing with Python assertions disabled')
if args.repeats < 1:
    parser.error('repeats must be positive')
if args.output.exists():
    parser.error('Refuse overwriting an existing benchmark')
source = args.source.resolve()
spec_path = args.spec.resolve()
freeze_path = args.freeze_record.resolve()
output_path = args.output.resolve()
environment = json.loads(args.environment.resolve().read_text(encoding='utf-8'))
assert Path(environment['python_executable']).resolve() == Path(sys.executable).resolve()
assert platform.system() == environment['system']
assert platform.python_version() == environment['python_version']
actual_packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
assert actual_packages == environment['packages'], 'Installed packages changed'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.chdir(source)
sys.path[:0] = [str(source/'src'), str(source)]
from experiments.research_v1_eval import identity, make_case, read_json, run_case
from experiments.run_q3_comparison import percentile
protocol = read_json(source/'research/v1_protocol.json')
spec = read_json(spec_path)
frozen = read_json(freeze_path)
assert identity(spec, protocol) == frozen['identity'], 'Frozen identity differs'
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
assert commit == frozen['git_commit'], 'Actual Git HEAD differs from the freeze'
assert spec.get('kwargs', {}).get('device', 'cpu') == 'cpu', 'This benchmark requires CPU inference'

def load_average():
    try:
        return list(os.getloadavg())
    except (OSError, AttributeError):
        return None

records = []
started = time.time()
initial_load = load_average()
output_path.parent.mkdir(parents=True, exist_ok=True)
row_log = output_path.with_suffix('.jsonl')
error = None
in_progress = None

def snapshot(complete=False):
    timed = [r for r in records if not r['warmup']]
    times = [r['program_runtime_s'] for r in timed]
    stable = complete and all(len({r['action_history_sha256'] for r in timed if r['seed'] == seed}) == 1
                              for seed in range(6000, 6016))
    return dict(strategy=spec['name'], identity=frozen['identity'], git_commit=commit,
                worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                environment_sha256=hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest(),
                started_epoch=started, elapsed_wall_s=time.time()-started,
                initial_load_average=initial_load, final_load_average=load_average(),
                repeats=args.repeats, warmup_cases=sum(r['warmup'] for r in records), timed_cases=len(timed),
                complete=complete, error=error, in_progress=in_progress,
                all_successful=complete and all(r['successful'] and r['failed_clear_count'] == 0 for r in records),
                repeated_action_histories_identical=stable,
                mean_program_runtime_s=statistics.mean(times) if times else None,
                median_program_runtime_s=statistics.median(times) if times else None,
                p95_program_runtime_s=percentile(times, .95) if times else None,
                max_program_runtime_s=max(times) if times else None,
                scope='Serial CPU whole-episode strategy plus simulator-interface time, after one excluded warmup; opened development seeds 6000..6015. Excludes Python process startup and is not pure neural-network latency or independent final performance evidence. An external deadline supervisor must bound this worker process.',
                rows=records)

def save_progress(result):
    temporary = output_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(output_path)

with row_log.open('x', encoding='utf-8') as stream:
    save_progress(snapshot())
    try:
        for repeat in range(-1, args.repeats):
            seeds = [6000] if repeat == -1 else list(range(6000, 6016))
            for seed in seeds:
                remaining = datetime.fromisoformat(protocol['hard_deadline']).timestamp()-time.time()-30
                if remaining < 5:
                    raise TimeoutError('Hard deadline reached; benchmark is incomplete')
                in_progress = dict(seed=seed, repeat=repeat, started_epoch=time.time())
                save_progress(snapshot())
                record = run_case(make_case('validation', seed, protocol), spec, protocol,
                                  min(protocol['limits']['real_seconds_per_case'], remaining))
                row = record['row']
                actions = (record.get('summary') or {}).get('action_history')
                action_bytes = json.dumps(actions, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
                item = {**row, 'repeat': repeat, 'warmup': repeat == -1,
                        'action_history_sha256': hashlib.sha256(action_bytes).hexdigest()}
                records.append(item)
                stream.write(json.dumps(item, allow_nan=False)+'\n')
                stream.flush()
                in_progress = None
                save_progress(snapshot())
                print(json.dumps({k: item[k] for k in ('seed', 'repeat', 'successful', 'program_runtime_s')}), flush=True)
        assert identity(spec, protocol) == frozen['identity'], 'Identity changed during timing'
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip() == commit
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        result = snapshot(complete=error is None and len(records) == 1+16*args.repeats)
        save_progress(result)
print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'identity')}), flush=True)
