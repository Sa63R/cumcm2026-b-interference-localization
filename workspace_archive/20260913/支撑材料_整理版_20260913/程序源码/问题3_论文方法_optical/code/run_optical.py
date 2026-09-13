"""Run one Q3 case using the frozen optical policy and reconstructed engine.

Use this entry point, not the historical main() in vendor/policies.
The policy and adapter are identical to those used for the paired benchmark.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import runner


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', default='q3-top5-holdout-20260912-0')
    parser.add_argument('--method', choices=runner.METHODS, default='optical')
    parser.add_argument('--output', type=Path, help='New .json or .json.gz run export')
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error('Output already exists; choose another path.')
    runner.check_inputs()
    runner.initialize()
    row, data = runner.run_one(runner.Scenario.generate(3, args.seed), args.method)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
        if args.output.suffix == '.gz':
            raw = gzip.compress(raw, mtime=0)
        with args.output.open('xb') as stream:
            stream.write(raw)
    print(json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if row['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
