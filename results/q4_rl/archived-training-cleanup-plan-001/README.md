# Approval proposal: archived Q4 training payload cleanup

The exact cleanup was explicitly approved by the user and completed at 20:04:33
UTC (September 12 04:04 Beijing). All 5184 approved historical payload copies
were removed from the server; both archives, models, code and other files remain.
The original proposal and its read-only audit below are preserved as provenance.

The exact 5,184 files in `approval-manifest.json` total **16,473,084,571 bytes (15.342 GiB)**. Delete only those ordinary files if later approved; do not delete entire directories or expand filesystem globs.

| Completed task | Payload group | Files | Logical bytes |
|---|---|---:|---:|
| q4-rl-memory-v4-20260912 | `g1_h128/training` | 656 | 1,960,299,042 |
| q4-rl-memory-v4-20260912 | `g1_h64/training` | 864 | 2,744,052,666 |
| q4-rl-memory-v4-20260912 | `g3_h128/training` | 624 | 2,281,198,298 |
| q4-rl-memory-v4-20260912 | `g3_h64/training` | 784 | 3,460,145,263 |
| q4-rl-bundle-v3-20260912 | `attention_scratch/training` | 336 | 1,132,188,895 |
| q4-rl-bundle-v3-20260912 | `mlp_scratch/training` | 416 | 1,459,851,854 |
| q4-rl-bundle-v3-20260912 | `ppo_initialized/training` | 400 | 1,424,520,960 |
| q4-rl-bundle-v3-20260912 | `scst_initialized/training/raw` | 1104 | 2,010,827,593 |

Remote task roots, supplied and previously checked by the main task:

- `/home/dataset-assist-0/usr/lh/ysh/bwc/shumo/q4-rl-memory-v4-20260912`
- `/home/dataset-assist-0/usr/lh/ysh/bwc/shumo/q4-rl-bundle-v3-20260912`

Each allowlist path is relative to its task's `remote_run_root`; `object_archive_prefix` plus the same relative path locates the object backup. Local backup folders, complete readback hashes, per-file SHA/size, and exact allowlist hashes are embedded. This check rehashed only 324 small indexes (183 memory + 72 bundle episode + 69 SCST), joined every one of the 5,184 expected payload hashes to the complete readback records, and checked local existence and sizes. No archived payload was bulk rehashed.

Keep **all models/checkpoints, code, indexes, logs, configurations, source archives and evaluation outputs**. Exclude the entire current `q4-rl-memory-v4-eval-20260912` and `q4-rl-gae-v5-20260912` tasks, and all other users/tasks. Keep both local and object-store archives, including failed and administratively interrupted attempts.

Before any deletion: freshly resolve approved roots, verify no symlinks or live PID/open-file consumers, and compare exact relative filenames and sizes against the allowlist. Any discrepancy must stop the proposed cleanup. These are historical archive hashes, not fresh remote-content hashes. Approval must cover the exact manifest, not a blanket directory removal.

**Space limit:** 15.342 GiB is logical payload size; actual reclaimed blocks may differ. From roughly 5.9 GB/GiB free this would leave only a narrow margin over the 20 GiB guard, and concurrent shared-disk growth remains uncontrolled. This proposal does **not** establish enough space to resume 32-episode-batch training. Recheck actual free space and growth after any approved cleanup before considering a restart.

Manifest SHA256: `ee3448179cd5cf1db269ea6989918533ee480c37d7343321c047d41e5db18330`.

## Fresh remote read-only audit

`remote_preflight.py` has no deletion mode. From 19:50:54 to 19:54:46 UTC it
verified all 5184 current server payload SHA256 values against the immutable
allowlist, regular-file status, one hard link, exact canonical paths and sizes.
The object-store archive listing contains every corresponding name and size.
Allocated blocks total 16,514,613,248 bytes; logical payload bytes remain the
number above. No large file was transferred; hashing ran on one server CPU.

No matching process command or open file was found in visible processes. However,
21 restricted processes could not be fully inspected before or after the audit,
so `ready_for_explicit_approval` is false under this script's strict all-process
visibility gate. This is a visibility limit, not evidence of an actual consumer;
do not claim that all possible users of the files were ruled out. No deletion was
performed. The result is `READ_ONLY_RESULT.json`, SHA256
1f5bddb51f97040ebd2b30e7504134b01b6463d71a29ecaf493fe70fcfe4598b.
Free disk after auditing was 5,672,534,016 bytes. Even adding the allocated bytes
would leave only about 0.66 GiB above the 20 GiB guard, before other disk growth.

## Object traffic constraint

Latest AGENTS.md caps object-store traffic at 100G. Historical unique readback
versions account for 18,875,406,183 bytes, recorded downloads for 19,223,606,200
bytes. Adding at least one upload of the unique versions gives a known lower
bound of about 38.10 GB; retries, changing files, incomplete transfers and other
tasks prevent treating the difference from 100 GB as a verified remaining budget.
The current stopped jobs used ordinary incremental rclone copy, not forced full
reuploads. Avoid additional full TRAIN readbacks. Future large exchanges need
explicit byte accounting and a delivery reserve; the existing supervisor does
not implement a cumulative 100G limiter.

## Approved execution

The user replied “允许清理清单中的旧轨迹” after the exact scope, backup evidence,
limited expected headroom and process-visibility gap were disclosed. No second
approval was requested. `approved_cleanup.py` is the executed revision, SHA256
1a711b8fb1c145a9cea72388e055e98ccb9fa940e638b35cc8b54d19aed2822b.
It binds the exact manifest and recent remote content audit, validates every
regular file's canonical path, ownership, single-link status, size and unchanged
mtime/ctime, and checks its full stat signature again immediately before unlink.
Any file changed since the content audit must be rehashed with unchanged stat
signatures on both sides. The whole allowlist is validated before the first unlink.

The strict first consumer gate rejected before any deletion. Read-only process
inspection classified the inaccessible entries as exited zombies and SSH
transport daemons. The final gate checks command, cwd, executable and open-file
references against both old task roots. It reported no visible consumers, no
unclassified restricted same-UID process, 15 exited zombies, 2 protected same-UID
SSH daemons and 4 protected other-UID processes. The protected SSH/system
descriptors remain a recorded visibility limit, not a claim of complete access.
No process or session was stopped or changed.

The immutable `releases/archive-cleanup-approved-001` through `004` object
directories retain the preparatory revisions. Only revision004 deleted files;
the earlier checks stopped before mutation. `APPROVED_CLEANUP_RESULT.json`
(SHA256 59f7369cba0497cc912bf3f54f2e3c0572db95994bacdafc0680c621fb55cbdd)
reports all 5184 files and 16,473,084,571 logical bytes removed. `deleted.jsonl`
(SHA256 2dd7aa1d3cc44bd58f7cbaf54a86539dc5129d770d0eae4569da13df3babfeb8)
was read back and matched one-for-one to the complete approved manifest, with no
extra or missing entry. Object copies and local archives were not deleted.

The immediate statfs reading lagged. At 20:05 UTC, `df -B1` showed
21,835,022,336 bytes free: only 360,185,856 bytes (0.335 GiB) above the 20 GiB
guard. Old memory and bundle task allocation had fallen to 170224 and 870024 KiB.
This does not provide enough growth margin for the remaining experiment, so no
training or evaluation was restarted. Resume tools were prepared and tested
separately; more stable free space is still required.
