"""Restore original Git identities locally; no training, simulation or HTTP."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                     separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify_files():
    manifest = read(ROOT / 'checksums.json')
    errors = [name for name, value in manifest['files'].items()
              if not (ROOT / name).is_file() or sha(ROOT / name) != value]
    if errors:
        raise ValueError(f'Package file hashes differ: {errors}')
    print(f"Verified {len(manifest['files'])} package files", flush=True)


def git(*args, cwd=None):
    subprocess.run(['git', *map(str, args)], cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    verify_files()
    if args.verify_only:
        return
    if args.output is None:
        parser.error('Supply --output as a new directory, or use --verify-only')
    out = args.output.resolve()
    if out.exists():
        raise ValueError('Output already exists; retained as-is. Choose a new directory.')
    out.mkdir(parents=True)
    meta = read(ROOT / 'source-identities.json')
    repo = out / 'history'
    git('-c', 'core.autocrlf=false', 'clone', '--no-checkout',
        ROOT / 'sources/source-history.bundle', repo)
    git('config', 'core.autocrlf', 'false', cwd=repo)
    restored = {}
    for name, entry in meta['algorithms'].items():
        target = out / name
        git('worktree', 'add', '--detach', target, entry['git_commit'], cwd=repo)
        spec_target = target / entry['spec_relative_path']
        spec_target.parent.mkdir(parents=True, exist_ok=True)
        spec_bytes = (ROOT / entry['package_spec']).read_bytes()
        if spec_target.exists() and spec_target.read_bytes() != spec_bytes:
            raise ValueError(f'Tracked spec differs in {name}')
        spec_target.write_bytes(spec_bytes)
        for model in entry['models']:
            destination = target / model['relative_path']
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / model['package_path'], destination)
        frozen = read(ROOT / entry['package_freeze'])
        spec = read(spec_target)
        source_files = sorted((target / 'src').rglob('*.py'))
        source_files += [target / 'experiments/research_v1_eval.py',
                         target / 'experiments/run_q3_comparison.py']
        observed = {
            'source_sha256': {p.relative_to(target).as_posix(): sha(p) for p in source_files},
            'spec_sha256': canonical_sha(spec),
            'checkpoint_sha256': {key: sha(target / value) for key, value in spec.get('kwargs', {}).items()
                                  if key in ('checkpoint', 'weights') and value},
            'protocol_sha256': canonical_sha(read(target / 'research/v1_protocol.json')),
        }
        if observed != frozen['identity'] or frozen['git_commit'] != entry['git_commit']:
            raise ValueError(f'Restored identity mismatch: {name}')
        restored[name] = {'git_commit': entry['git_commit'], 'identity_matches': True,
                          'source': str(target), 'spec': str(spec_target)}
    for name, commit in meta['tools'].items():
        git('worktree', 'add', '--detach', out / name, commit, cwd=repo)
    (out / 'restoration.json').write_text(json.dumps(restored, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(restored, indent=2))
    print('Only sources and original model/config bytes restored; no simulator was called.')


if __name__ == '__main__':
    main()
