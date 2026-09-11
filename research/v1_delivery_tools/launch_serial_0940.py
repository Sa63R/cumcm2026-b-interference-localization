"""Time frozen selected policies after final evaluations and audits have finished."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

root = Path('/home/volleyball/q3-research-v1')
read = lambda name: json.loads((root / name).read_text())
assert not sys.flags.optimize
assert read('v1-selection/final-identity-audit.json')['identity_and_archive_checks_passed']
assert read('v1-selection/statistics-status.json')['status'] == 'completed'
physical = read('v1-selection/final-physical-supervisor-status.json')
assert physical['phase'] == 'complete' and physical['summary']['audit_passed']
assert physical['summary']['records'] == 1136
for pid in (1041644, 1068490, 1079581, physical['audit_pid']):
    stat = Path(f'/proc/{pid}/stat')
    if stat.exists():
        assert stat.read_text().split(') ', 1)[1].split()[0] == 'Z', f'Heavy worker {pid} is active'
for name, expected in {
    'run_serial_benchmarks_r1.py': '8f4f9a363547103dc6ec583210fe1026b206b31e23a146ab3c42bbc4665e84cd',
    'serial_runtime_benchmark.py': 'bdab1cea20fcdab8de05da466a481a5f437e49456ba52141109a77ed74112ca0',
}.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
script = root / 'run_serial_benchmarks_r1.py'
sys.argv = [str(script), '--tool', str(root / 'evaluation-tools-0830-git/experiments/research_v1_selection.py'),
            '--registry', str(root / 'v1-selection/registry.json'),
            '--selection', str(root / 'v1-selection/selected.json'),
            '--worker-script', str(root / 'serial_runtime_benchmark.py'),
            '--run-root', str(root / 'v1-selection/serial-r1')]
for trial in ('rl-cold-route-v3-001', 'rl-cold-route-v4-001',
              'rl-cold-strong-attention-001', 'rl-cold-gae095-001'):
    sys.argv.extend(['--training-process-record', str(root / f'logs/{trial}-process.json')])
runpy.run_path(str(script), run_name='__main__')
