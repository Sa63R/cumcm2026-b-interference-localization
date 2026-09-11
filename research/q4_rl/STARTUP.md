# Q4 reinforcement learning: starting evidence and controls

Started 2026-09-11 22:20 Beijing. Main training deadline is 2026-09-12
10:00 Beijing; final assessment and handoff are due before noon. This is a
research log, not a paper or a claim of improved performance.

## Protected state and scope

The independent `research/q4-deep-rl` worktree starts at
`dc8651b7ac93d4825242086520a43f62e90ab8bc`. Existing branches and uncommitted
changes are inventoried in `protected_worktrees.json`; they were not moved,
reset, merged, deleted or edited. The checkout is sparse to avoid duplicating
old result archives, which remain in Git history and their original worktrees.
The designated SQLite database remains unopened by this task.

Frozen comparisons are state search, compact 22-point coverage, range/scheduling
(`combo`), and the qualified `center_once` clear-before-probe variant (`r8`).
R8's independent evidence is restricted to synthetic mixed-source scenarios:
64 random cases and 42 stress cases, all cleared. Its ratio of mean time to
mean common lower bound was 3.148221 and 4.022831 respectively. Optical trial
misses occurred and were charged; the evidence does not imply zero failed
clear attempts. These historical figures are context, not paired RL evidence.

## Resource and identity check

Q4 uses port **42222**, with the pre-existing server host key checked strictly.
The inspected container exposes 160 logical CPUs but has a **96 CPU** cgroup
v1 quota (`9600000 / 100000`). Initial load averages were approximately
13.71/13.77/15.80; short host samples showed 94% idle. Existing Ray and Python
tasks were present. The available memory and host idle percentages are not a
reservation of those resources.

All new server files are under the independent `q4-deep-rl-20260911` directory
inside the user-approved base. No pre-existing Q3 directory or task is altered.
Initial dependency setup uses only two CPUs. Training will start with a small
pilot, then the supervisor can restrict the entire job tree to at most 50 CPUs
by default (never over 60); numerical libraries use one thread per worker.
The supervisor lowers affinity if other work consumes the shared quota and
uses lower scheduling priority. GPU visibility is empty and the installed
Torch build must be CPU-only.

Release archives and results travel only through the authorized object-store
task prefix. `rclone` operations use `--s3-no-check-bucket`. SSH is used for
commands and process/resource checks, not file payloads. External credentials
are read in memory and excluded from source, releases, manifests and logs.

## First-stage hypothesis

A shared candidate scorer chooses a single physical scan location/channel or
a bounded inherited source-service macro from legal public observations.
Unlike a frozen station order, the policy can move, interrupt scanning or
service a discovered source at each decision. Completion still requires the
public 16-source limit with 16 actual clears, or complete certified discovery
and actual clearance of all known sources. The frozen 22-point set remains a
fallback proof; arbitrary current-position scans do not receive unearned
coverage credit.

The initial model is a candidate MLP with permutation-invariant pooled context.
PPO is used as a first baseline, with undiscounted total billed costs and a
fixed physical scale. PPO's clipped objective is grounded in
[Schulman et al., 2017](https://arxiv.org/abs/1707.06347); it does not guarantee
improvement in this environment. Candidate-set encoders and later attention
ablations are motivated by
[Kool et al., 2019](https://arxiv.org/abs/1803.08475), whose routing results do
not establish Q4 exploration or clearance performance.

## Required next stages and decision evidence

1. Reproduce frozen controls and heuristic action-extension baselines on the
   identical new development scenes; test raw action and completion audits.
2. Run a small CPU training pilot, measuring full episode accounting, resource
   use, checkpoint recovery, and object-store synchronization.
3. Compare random initialization, synthetic-only imitation warm start, and PPO
   endpoints. Development comparisons, not training loss, determine expansion.
4. Test joint localization/clear actions, adaptive discovery positions with
   actual per-channel coverage certificates, and history-aware encoders on
   separate experiment branches. The first-stage service macro does not meet
   the eventual full autonomy objective by itself.
5. Freeze candidates before independent confirmation. Preserve consumed splits,
   failures and adverse cases. Retain R8 as the recommended control until an
   independently audited candidate satisfies the preregistered gates.

All reports use `q4-common-source-edge-v1`, distinguish completed actual T/L
from penalized incomplete outcomes, and retain wall-clock/CPU overhead. No
official test or practice session is started by this startup work.
