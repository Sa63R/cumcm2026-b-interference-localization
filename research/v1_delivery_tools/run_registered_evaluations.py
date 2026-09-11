"""Run only the identities fixed by the first-version registry/selection."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--phase', choices=('extended', 'final'), required=True)
parser.add_argument('--tool', type=Path, required=True)
parser.add_argument('--registry', type=Path, required=True)
parser.add_argument('--selection', type=Path)
parser.add_argument('--run-root', type=Path, required=True)
parser.add_argument('--workers', type=int, default=3)
args = parser.parse_args()
if sys.flags.optimize:
    raise RuntimeError('Do not run audited orchestration with Python assertions disabled')
if not 1 <= args.workers <= 4:
    parser.error('Use one to four isolated CPU processes')
module_spec = importlib.util.spec_from_file_location('registered_selection_tool', args.tool.resolve())
selection_tool = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(selection_tool)
registry_path = args.registry.resolve()
registry = selection_tool.verify_registry(registry_path)
run_root = args.run_root.resolve()
run_root.mkdir(parents=True, exist_ok=True)
if args.phase == 'extended':
    if args.selection is not None:
        parser.error('Extended evaluation must precede selection')
    ids = list(registry['entries'])
    splits = ['validation_extended']
else:
    if args.selection is None:
        parser.error('Final evaluation requires a sealed selection decision')
    chosen = selection_tool.read_sealed(args.selection, 'selection_decision')
    assert chosen['registry_file_sha256'] == selection_tool.file_hash(registry_path)
    assert chosen['registry_payload_sha256'] == registry['payload_sha256']
    assert chosen['rule_sha256'] == registry['rule_sha256']
    assert selection_tool.timestamp(chosen['finalized_at']) >= selection_tool.timestamp(registry['registered_at'])
    previous = selection_tool.load_partition(
        {k: v['directory'] for k, v in chosen['evaluations'].items()},
        registry, 'validation_extended', registry['entries'])
    assert selection_tool.evidence(previous) == chosen['evaluations']
    assert selection_tool.select(registry, previous) == chosen['decisions']
    ids = [registry['baseline']] + [chosen['decisions'][d]['selected'] for d in selection_tool.DIRECTIONS]
    splits = ['final_random', 'final_stress']

environment = next(iter(registry['entries'].values()))['environment_value']
assert Path(environment['python_executable']).resolve() == Path(sys.executable).resolve(), 'Use the registered interpreter'
import importlib.metadata
import platform
assert platform.system() == environment['system']
assert platform.python_version() == environment['python_version']
actual_packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
assert actual_packages == environment['packages'], 'Installed packages changed since registration'
env = dict(os.environ, PYTHONPATH='src', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
hard_deadline = selection_tool.timestamp(registry['protocol_value']['hard_deadline']).timestamp()
index = {split: {key: str(run_root/'evaluations'/split/key) for key in ids} for split in splits}
index_path = run_root/('extended-index.json' if args.phase == 'extended' else 'final-index.json')
index_value = index['validation_extended'] if args.phase == 'extended' else index
if index_path.exists():
    assert selection_tool.read_json(index_path) == index_value
else:
    with index_path.open('x') as stream:
        json.dump(index_value, stream, indent=2)
logs = run_root/'logs'
logs.mkdir(exist_ok=True)

def execute(split, key):
    entry = registry['entries'][key]
    command = [sys.executable, 'experiments/research_v1_eval.py', 'run', '--split', split,
               '--spec', entry['spec'], '--freeze-record', entry['freeze'], '--output', index[split][key]]
    log = logs/(split+'-'+key+'.log')
    before = time.time()
    remaining = hard_deadline-before-15
    timed_out = False
    started_process = remaining > 5
    returncode = None
    if started_process:
        with log.open('a') as stream:
            child = subprocess.Popen(command, cwd=entry['source_dir'], env=env,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                returncode = child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    returncode = child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    returncode = child.wait(timeout=5)
    record = dict(split=split, id=key, command=command, cwd=entry['source_dir'],
                  started_epoch=before, elapsed_wall_s=time.time()-before,
                  returncode=returncode, started_process=started_process, timed_out=timed_out,
                  not_started_due_to_deadline=not started_process, log=str(log), output=index[split][key])
    print(json.dumps(record), flush=True)
    return record

def execute_guarded(split, key):
    try:
        return execute(split, key)
    except Exception as exc:
        record = dict(split=split, id=key, returncode=None, timed_out=False,
                      orchestration_error=f'{type(exc).__name__}: {exc}', output=index[split][key])
        print(json.dumps(record), flush=True)
        return record

started = time.time()
results = []
events_path = run_root/(args.phase+'-events-'+str(int(started))+'.jsonl')
with events_path.open('x') as events:
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(execute_guarded, split, key) for split in splits for key in ids]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            events.write(json.dumps(result)+'\n')
            events.flush()
record = dict(phase=args.phase, registry_sha256=selection_tool.file_hash(registry_path),
              launcher_sha256=selection_tool.file_hash(__file__),
              selection_sha256=selection_tool.file_hash(args.selection) if args.selection else None,
              started_epoch=started, elapsed_wall_s=time.time()-started, workers=args.workers,
              scope='Concurrent evaluation wall times measure budget use, not serial inference speed.',
              process_results=results)
record_path = run_root/(args.phase+'-execution-'+str(int(started))+'.json')
with record_path.open('x') as stream:
    json.dump(record, stream, indent=2)
if time.time() >= hard_deadline-60:
    print('HARD_DEADLINE_AUDIT_PENDING; completed process records retained', flush=True)
    sys.exit(3)
selection_tool.verify_registry(registry_path)
assert all(r['returncode'] == 0 and not r['timed_out'] for r in results), 'One or more evaluator processes failed; records/logs are retained'
if args.phase == 'extended':
    # This is a deterministic rule, not human selection after inspecting results.
    audit_command = [sys.executable, str(args.tool.resolve()), 'finalize', '--registry', str(registry_path),
                     '--evaluations', str(index_path), '--output', str(run_root/'selected.json')]
else:
    audit_command = [sys.executable, str(args.tool.resolve()), 'audit-final', '--registry', str(registry_path),
                     '--selection', str(args.selection.resolve()), '--evaluations', str(index_path),
                     '--output', str(run_root/'final-identity-audit.json')]
audit = subprocess.Popen(audit_command, start_new_session=True)
try:
    audit_returncode = audit.wait(timeout=max(1, hard_deadline-time.time()-15))
except subprocess.TimeoutExpired:
    try:
        os.killpg(audit.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        audit.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(audit.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        audit.wait(timeout=5)
    raise TimeoutError('Hard deadline reached during archive audit; completed evaluations are retained')
assert audit_returncode == 0, 'Archive audit failed; all records are retained'
print('Completed and audited', args.phase, flush=True)
