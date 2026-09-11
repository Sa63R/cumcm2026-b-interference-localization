# GAE v5 disk-stop resume sidecar

Prepared locally only. No upload, remote preflight, restart or training was performed.
The original 2026-09-11 19:27:10 UTC launch stopped administratively under the disk
guard; this is not a policy-failure result. Source remains `df36f0ce` and the
original config/source/models are unchanged.

## Resume contract

`resume.py` is bound to the four exact terminal `latest.pt`, last-complete and
status hashes already verified in `results/q4_rl/gae-v5-disk-stop-001`. It hashes
the immutable r2 archive, release manifest, all 131 source files, original config
and both BC artifacts before using the frozen trainer. It rejects another root,
module, configuration, output, deadline, pending reservation or checkpoint.

| Job | Committed batches | Prior cumulative wall (s) | Remaining max wall (s) |
|---|---:|---:|---:|
| g1_h128_mc | 13 | 836.039170 | 2763 |
| g1_h128_gae097 | 13 | 823.933948 | 2776 |
| g3_h128_mc | 12 | 842.946975 | 2757 |
| g3_h128_gae097 | 13 | 854.546776 | 2745 |

The helper computes `floor(3600 - latest.state.wall_time_s)`, keeping the original
absolute deadline **2026-09-11 22:00 UTC** (September 12 06:00 Beijing) and **32
total committed batches**, not 32 additional batches. The frozen trainers otherwise
start a new relative wall clock on every process start; passing 3600 again would
incorrectly grant another hour. Final checkpoint/pool shutdown overhead remains
reported in cumulative wall time; it is not erased or treated as strategy work.

Original `runs/train-gae-v5/<job>/training` directories remain in place. The helper
uses each directory's own `latest.pt`, retains `next_seed`, `next_attempt`, model,
Adam and Python/Torch RNG, and never clears `pending_batch`. The pending 16 seeds
and their already reserved action seeds replay as one transaction into a **new
attempt filename**. Previously committed batches are not repeated. Existing
partial attempt records remain preserved. The local audit does not have the
pending raw payloads, so original server directories are required. The sidecar
checks the last committed index and referenced files, any existing pending index,
and absence of the next attempt's output; it does not rehash large raw payloads.
A zero-result pending attempt can legitimately have no index yet.

**G1 and G3 resume flags differ.** G1 must retain its exact
`--initialize-micro-warmstart` and `--initialize-sha256`: `micro_train.py:214`
builds config binding from them, and `micro_train.py:238` validates BC bytes before
`micro_train.py:245` restores latest. BC weights are applied only in the non-resume
branch. G3 must omit both initialization flags: `memory_train.py:192` rejects
init+resume; `memory_train.py:242` restores its binding from latest. No new BC,
optimizer reset, random-stream reset or scenario interval is introduced.

## Root-reviewed execution only

Before approving execution, root must independently verify port 42222 identity,
old process groups exited, current CPU/load, restored disk headroom **above 20
GiB with expected growth allowed**, and remaining object-store traffic under the
100 G cap. This helper reserves at most 11 other compute cores: four jobs each
5 samplers + 4 learner threads = 36, combined 47 under the common 50-core ceiling.
Any additional workload requires a new resource review. It cannot infer other
tasks' future demand or the shared object-store account's remaining traffic.

After root review and object-only delivery of this helper to the existing task's
`launch/train-gae-v5-resume-001/resume.py`, these are the exact Linux commands:

```sh
cd /home/dataset-assist-0/usr/lh/ysh/bwc/shumo/q4-rl-gae-v5-20260912
taskset -c 0-49 /home/dataset-assist-0/usr/lh/ysh/bwc/shumo/q4-deep-rl-20260911/.venv-cpu/bin/python -B launch/train-gae-v5-resume-001/resume.py
```

That default command only checks and prints a plan. It refuses an existing new
run directory. To actually start after approval, append **`--supervise`** to the
same command. It repeats preflight, atomically creates the previously absent
`runs/train-gae-v5-resume-001`, records ownership, and `exec`s the unchanged
supervisor with CPU budget 50, its existing 8-core reserve logic, the 20 GiB disk
guard and original absolute deadline. The direct worker verifies supervisor PID,
full command, sidecar hash and the fresh claim, then repeats checks and invokes
the original `run_pair` with the four old output directories and resume argv.
It does not create independent sessions for learners; existing TERM/80-second
checkpoint grace and outer cleanup apply. Do not call hidden `--worker` directly.

The supervisor's existing root lock is not deleted or bypassed. It synchronizes
**all of this task's `runs`**, so both old training outputs and the new supervisor
logs continue to the same approved object prefix using its existing
`--s3-no-check-bucket` invocation. Failed new run directories are retained; this
one-use helper refuses to reuse them. After another interruption, freeze the new
latest/status hashes and recalculate cumulative budgets in a newly reviewed
sidecar rather than reusing this binding.

## Offline evidence

`test_resume.py`: **20 passed in 6.62 seconds**, one CPU, no rollout or optimizer
update. Tests use all four immutable latest files and the exact source archive;
they check source tampering, SHA/path/config/algorithm/cursor/pending rejection,
fresh-directory refusal, 47-core arithmetic and fixed deadline/remaining budget.

An actual G1 and G3 `main` invocation executes restore with the loop disabled at
the already completed batch count and all writes intercepted. Every attempted
save has the exact latest model, full Adam, Python/Torch RNG and pending state.
The G1 test additionally proves its loaded BC differs from latest and is not
applied. G1 missing binding is incompatible; G3 init+resume is rejected. No
simulator, formal test, validation database, object transfer or remote process
was used. Linux supervisor ownership and real pending-journal checks remain a
server preflight obligation; no policy efficacy or virtual-time claim is made.

Archive portability check: after copying these files into the tracked results
directory, `test_resume.py` was changed to locate the repository by its source
and deployment directories instead of assuming the original handoff depth.
The tracked-path command `python -B -m pytest
results/q4_rl/gae-v5-resume-preparation-001/test_resume.py -q` passed all 20 checks
in 6.65 seconds. `PORTABLE_TESTS.json` records its current test hash and unchanged
sidecar hash; `OFFLINE_VERIFIED.json` retains the original handoff execution hashes.
