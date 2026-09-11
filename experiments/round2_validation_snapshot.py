"""Freeze the user's growing practice DB without changing it or reading labels.

This is a storage backup only. Policy validation is a separate frozen-prefix
operation. The private 40+ MB database remains outside the Git worktree.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    source, target = args.source.resolve(strict=True), args.snapshot.resolve()
    if source == target or target.exists() or args.receipt.exists():
        raise ValueError('Use a new snapshot and receipt; do not overwrite evidence')
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source.as_uri() + '?mode=ro', uri=True, timeout=10) as src:
        src.execute('PRAGMA query_only=ON')
        with sqlite3.connect(target) as dst:
            src.backup(dst, pages=256)
    result = dict(validation_role='validation_only', copied_at_utc=datetime.now(timezone.utc).isoformat(),
                  source=str(source), snapshot=str(target), snapshot_sha256=sha(target),
                  snapshot_bytes=target.stat().st_size,
                  backup_method='sqlite online backup from read-only connection; no labels or feedback queried',
                  contents_scope='historical action/feedback only; not a source-truth counterfactual simulator')
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open('x', encoding='utf-8') as output:
        json.dump(result, output, indent=2, ensure_ascii=False)
    print(json.dumps({'snapshot_bytes': result['snapshot_bytes'], 'sha256': result['snapshot_sha256']}))


if __name__ == '__main__':
    main()
