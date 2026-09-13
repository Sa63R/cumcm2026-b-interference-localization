"""Three frozen strategies; the slim candidate is the deployable module."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'reference'))
import benchmark as bench
sys.path.insert(0, str(ROOT / 'source_v6_lite'))
from q4_v6_lite import solve_v6_lite, default_config

VARIANTS = {
    'v4': 'original V4',
    'full': 'original V6, all defaults',
    'lite': 'remove learned route ranking and posterior shared scoring; keep rollout, learned transit and four-station guard',
}


def solve(device, method):
    if method == 'v4':
        return bench.solve_v4(device)
    if method == 'full':
        return bench.solve_v6(device, bench.V6_CONFIG)
    if method == 'lite':
        return solve_v6_lite(device)
    raise ValueError(method)
