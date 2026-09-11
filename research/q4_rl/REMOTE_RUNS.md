# Remote Q4 experiment ledger

All runs use Q4 port 42222 and the independent `q4-deep-rl-20260911`
directory/prefix. Files travel through the approved object store; SSH carries
commands and process/resource checks only. No validation database or official
test has been used by these runs.

## CPU environment and transport

The current visible cgroup quota is 96 CPUs, independently checked on Q4.
Other Ray/Python work is present; it is not part of this task. The supervisor
uses a default 50-core affinity ceiling shared by its complete process tree,
lower scheduling priority, one numerical thread per worker, and reduces the
allocation under observed shared-quota pressure. Invisible host ancestors are
explicitly unobservable. The environment was verified as Torch 2.9.1+cpu,
NumPy 2.2.6, CUDA build null, with `pip check` passing.

The exact dependency versions are preserved in
`results/q4_rl/server-training-pilot-001/requirements-resolved.log`.
The Linux dependency wheel bundle SHA-256 is
`1c13e388ea64dd6476b81a3eb658f55d5cab59bdb9470eaa3cd804681bf93638`;
the object key is `releases/q4-dependencies-20260911-r1.tar.gz` within this
task's prefix. The fetched wheels and installation outputs remain inside the
independent task directory. Failed remote downloads and setup logs remain in
`runs/setup`; no shared environment was repaired or upgraded.

## Completed unified controls

`baseline-dev-003/evaluation`: 8 development seeds, two source modes, five
controls, 6 workers under an 8-core ceiling. All 80 runs completed and passed
physical/completion audits. See `BASELINE_REMOTE_001.md` and the complete
object readback in `results/q4_rl/server-baseline-001`.

## MLP128 training pilot

The live source release was built from commit
`90a3dad0bcb1d37cb4b9a7874dd545cd32411bf6`; its archive SHA-256 is
`bcad884b9cf4ef4cefd269a868780c2beffe7bb7ce4c1338bfb17423d5e6bb5c`.
The source and dependency byte hashes were verified after object transfer.
Subsequent local diagnostics and Git byte-protection changes do not alter the
already running trainer.

`train-v1` was rejected before training because the supervisor had already
created its log/status in the trainer's requested empty output directory.
The refusal and successful supervisor cleanup are retained in
`results/q4_rl/training-launch-failure-001`. No scene or model update occurred.

The corrected run is `train-v1b`, with nested output
`runs/train-v1b/training`. Its configuration was fixed in `PILOT_MLP_V1.md`:
8 workers, 16 episodes/batch, 64 synthetic imitation episodes, then PPO;
128 decisions followed by the fully charged fixed continuation, 1800 wall
seconds. Use the following arguments from the independent server directory:

```text
.venv-cpu/bin/python -B -m q4_rl.train --output runs/train-v1b/training --workers 8 --cpu-budget 50 --batch-episodes 16 --warmstart-episodes 64 --max-decisions 128 --hidden 64 --epochs 3 --minibatch-size 128 --max-wall-seconds 1800 --deadline 2026-09-12T10:00:00+08:00
```

Launch this through `scripts/q4_cpu_supervisor.py` with `--root` set to the
approved independent task directory, a fresh `--run` label, `--cpu-budget 50`,
and `--sync-seconds 30`. The trainer output must be a new subdirectory inside
the supervisor's run directory. Resuming uses `--resume` and exactly the same
training configuration, with a separately chosen wall limit.

At one mid-pilot resource check, all 15 observed threads in 11 task processes
had a 50-CPU affinity set; their combined RSS was 3,874,304,000 bytes and
cumulative CPU time 1293.59 seconds after about 831 wall seconds. These are
point observations, not a claim to have retrospectively saved every earlier
resource sample. A subsequent local supervisor revision adds append-only
resource history for future runs.

The first raw batch and checkpoint have been fetched through object storage,
and all 16 complete histories/returns independently checked offline. See
`TRAINING_REVIEW.md`. Training diagnostics are not independent performance
estimates. The pilot endpoint, paired development evaluation and any decision
to continue must be recorded after they actually complete.

## Evidence recovery

`.gitattributes` now preserves exact Q4 RL source/evidence bytes. The initial
Windows smoke JSON files needed their original line endings restored to the
Git index; all 145 then-tracked relevant files were checked against their
working bytes, and all four evaluation bundles' record/manifest/summary hashes
were verified. This changes transport fidelity, not scene outcomes or metrics.
