"""Unlink only the user's explicitly approved, freshly audited historical files."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import time

BASE = Path('/home/dataset-assist-0/usr/lh/ysh/bwc/shumo')
OWN = BASE / 'q4-rl-gae-v5-20260912'
MANIFEST_SHA = 'ee3448179cd5cf1db269ea6989918533ee480c37d7343321c047d41e5db18330'
AUDIT_SHA = '1f5bddb51f97040ebd2b30e7504134b01b6463d71a29ecaf493fe70fcfe4598b'
TASKS = {'q4-rl-memory-v4-20260912': 'train-memory-v4', 'q4-rl-bundle-v3-20260912': 'train-bundle-v3'}

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def consumers(roots):
    refs, own_unreadable, other_unreadable, unreadable_pids = [], 0, 0, []
    zombies, restricted_ssh = 0, 0
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        owned = None
        try:
            owned = proc.stat().st_uid == os.getuid()
            fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] == 'Z':
                zombies += 1
                continue  # Exited processes have no live address space or open descriptors.
            command = (proc / 'cmdline').read_bytes()
            if any(str(root).encode() in command for root in roots):
                refs.append(int(proc.name))
            if owned:
                for name in ('cwd', 'exe'):
                    try:
                        target = os.readlink(proc / name)
                    except FileNotFoundError:
                        continue
                    if any(target == str(root) or target.startswith(str(root) + '/') for root in roots):
                        refs.append(int(proc.name))
            for fd in (proc / 'fd').iterdir():
                try:
                    target = os.readlink(fd)
                except FileNotFoundError:
                    continue
                if any(target.startswith(str(root) + '/') for root in roots):
                    refs.append(int(proc.name))
        except (FileNotFoundError, ProcessLookupError):
            pass
        except PermissionError:
            unreadable_pids.append(int(proc.name))
            if owned is False:
                other_unreadable += 1
            else:
                try:
                    transport = (proc / 'comm').read_text().strip() == 'sshd_static'
                except OSError:
                    transport = False
                if transport:
                    # Verified SSH transport daemons, not a training/evaluation
                    # command. Their protected descriptors remain a disclosed
                    # visibility gap; no daemon or session is changed or stopped.
                    restricted_ssh += 1
                else:
                    own_unreadable += 1
    if refs or own_unreadable:
        print(json.dumps(dict(visible_reference_pids=sorted(set(refs)), unreadable_pids=unreadable_pids,
            own_unreadable=own_unreadable, other_unreadable=other_unreadable)), flush=True)
    assert not refs and own_unreadable == 0, 'Visible consumer or uninspectable own process'
    return dict(visible_consumers=0, same_uid_unreadable=own_unreadable,
                other_uid_unreadable=other_unreadable, exited_zombies=zombies,
                restricted_same_uid_ssh_transport=restricted_ssh)

def signature(s):
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_nlink)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--approved-manifest-sha256', required=True)
    args = p.parse_args()
    assert args.approved_manifest_sha256 == MANIFEST_SHA
    assert Path.cwd().resolve() == OWN.resolve() == OWN
    manifest_path = OWN / 'launch/archive-cleanup-preflight-001/approval-manifest.json'
    audit_path = OWN / 'runs/archive-cleanup-preflight-001/READ_ONLY_RESULT.json'
    assert sha(manifest_path) == MANIFEST_SHA and sha(audit_path) == AUDIT_SHA
    plan, audit = json.loads(manifest_path.read_bytes()), json.loads(audit_path.read_bytes())
    assert {t['task']: t['run'] for t in plan['tasks']} == TASKS
    assert len(audit['tasks']) == 2 and all(t['all_current_remote_sha256_match'] and
        t['all_object_archive_names_and_sizes_present'] for t in audit['tasks'])
    audit_epoch = datetime.fromisoformat(audit['started_utc']).timestamp()
    assert 0 <= time.time() - datetime.fromisoformat(audit['finished_utc']).timestamp() < 1800
    roots = [BASE / t for t in TASKS]
    dependency = consumers(roots)
    selected, total, rehashed = [], 0, 0
    for task in plan['tasks']:
        root = BASE / task['task'] / 'runs' / task['run']
        assert root.resolve() == root and str(root) == task['remote_run_root']
        for item in task['files']:
            rel = PurePosixPath(item['name'])
            assert not rel.is_absolute() and '..' not in rel.parts
            assert ('-episode-' in rel.name or '-leg-' in rel.name) and rel.name.endswith('.json.gz')
            path = root.joinpath(*rel.parts)
            assert path.resolve() == path and path.is_relative_to(root)
            s = path.lstat()
            assert stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and s.st_uid == os.getuid()
            assert s.st_size == item['bytes']
            # The fresh audit hashed every byte. ctime/mtime before its start
            # establishes that the ordinary file has not changed since then.
            if max(s.st_ctime, s.st_mtime) > audit_epoch:
                before_hash = signature(s)
                assert sha(path) == item['sha256']
                rehashed += 1
                s = path.lstat()
                assert signature(s) == before_hash, 'File changed while hashing'
            selected.append((path, signature(s), task['task'], item))
            total += item['bytes']
    assert len(selected) == 5184 and total == 16473084571
    assert len({str(p) for p, *_ in selected}) == len(selected)
    consumers(roots)
    out = OWN / 'runs/approved-archive-cleanup-001'
    out.mkdir(exist_ok=False)
    result = dict(manifest_sha256=MANIFEST_SHA, read_only_audit_sha256=AUDIT_SHA,
        user_approval='Allow cleanup of the exact listed old trajectories',
        dependency_check=dependency, limitation='Other-UID restricted processes remain unobservable; this limitation was disclosed before explicit user approval.',
        started_utc=datetime.now(timezone.utc).isoformat(), free_bytes_before=shutil.disk_usage(OWN).free,
        selected_files=len(selected), selected_bytes=total, freshly_rehashed_after_audit=rehashed,
        deleted_files=0, deleted_bytes=0, complete=False)
    try:
        with (out / 'deleted.jsonl').open('x') as journal:
            for path, expected, task, item in selected:
                assert signature(path.lstat()) == expected, 'File changed after preflight'
                path.unlink()
                journal.write(json.dumps(dict(task=task, **item)) + '\n')
                journal.flush()
                result['deleted_files'] += 1
                result['deleted_bytes'] += item['bytes']
        result['complete'] = True
    finally:
        result.update(finished_utc=datetime.now(timezone.utc).isoformat(),
            free_bytes_after=shutil.disk_usage(OWN).free)
        with (out / 'RESULT.json').open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
        print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
