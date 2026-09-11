# Micro SCST v1: implementation and functional check

This is a separate algorithm experiment over the unchanged 13 global / 50 candidate public features and 512-decision micro controller. It is not a recommended policy or independent efficacy result. The SCST checkpoint version is `q4-micro-scst-cpu-v1`; evaluation uses `q4_rl.scst_network:load_policy` with deterministic inference.

For every new training scene, one sampled trajectory and one current-policy greedy trajectory run in separate real synthetic environments. With complete billed time `C` (including fallback and terminal costs), failure accounting is `max(C, 360000)`. The detached advantage is `(C_greedy - C_sample) / 1000`. The loss is the batch mean of `-advantage * sum(log_probability)` over whole sampled trajectories. There is one optimizer step after all trajectory/microbatch gradients accumulate, with no trajectory-length normalization, critic loss, entropy bonus, winning-case selection, or discounting. Both rollouts count toward resource use. The self-critical baseline follows [Rennie et al.](https://arxiv.org/abs/1612.00563).

Initialization verifies the exact micro BC file SHA before decoding, requires complete BC / zero PPO / no pending transaction, and copies only weights. Fresh Adam, counters and explicit experiment RNG are used. `initialization` records only the shared `frozen-micro-bc-weights-v1` type and content hash. A reservation contains one action seed per scene, shared by its two legs; neither the seed nor the baseline/hidden truth/lower bound enters the policy. Lower bounds are calculated only after both the policy termination and each leg's reward accounting.

The trainer stores an immutable sampling-policy checkpoint per attempt, one compressed raw file per returned leg, a small manifest and append-only hash index. It does not repeatedly serialize earlier episodes. An administrative interruption leaves the complete reserved scene batch pending. Resume restores the committed model/optimizer/RNG and replays that reservation, while preserving previous raw attempts. The outer CPU supervisor remains the authority for complete process-tree CPU overhead; the trainer separately reports both-leg CPU, update CPU, and measured parent-plus-leg CPU without double counting inline workers.

## Functional verification

Eleven fixture tests pass, covering gradient sign/reduction, one optimizer step, greedy/truth/bound independence, full failure/fallback/entry-terminal billing, strict initialization, checkpoint/RNG restoration, interrupted-update rollback, and non-overwriting raw/replay behavior. Independent review found no substantive algorithm issue. The post-smoke billing guard additionally accepts the controller's existing nonzero `uncovered_cost_s` entry/terminal subtotal; a fixture checks this without another real rollout.

One authorized synthetic training scene (`8006000`, one sampled and one greedy rollout) used the frozen 64-episode micro BC file with SHA256 `cb5df8dd338cc740222fc0edd5259ea63a64ddd7cb12398d245e781eb7843757`. CPU affinity and Torch were both limited to one CPU/thread. These are pre-update rollouts, not evaluations of the updated model.

| Leg | All cleared | Failed clears | Actual time (s) | Common lower bound (s) | T/L | Fallback (s) |
|---|---:|---:|---:|---:|---:|---:|
| Sample | 1/1 | 36 | 12470.172592 | 2096.484437 | 5.948135 | 0 |
| Greedy | 1/1 | 5 | 8470.453858 | 2096.484437 | 4.040313 | 612.267170 |

The sample recorded 508 decisions, complete cost sum `12470.172592`, and advantage `-3.999718734`; exactly one SCST update ran. Training wall time was 12.32 s, both rollout CPU times summed to 9.00 s, and update wall/CPU were 0.915/0.875 s. With one scene, mean and p95 equal the displayed time and statistical uncertainty cannot support an efficacy claim. Failed-clear attempts remain visible; completion alone is not evidence for a safety upgrade. The raw leg/index hashes and immutable policy hash match. Raw actions, observations, post-termination evaluation, source seed/factory specification, and bound evidence are retained in `results/scst-smoke-8006000`; `smoke-manifest.json` records source hashes at execution. Later changes only cover the billing guard and CPU diagnostic accounting.

## Running

Set `PYTHONPATH=src` and use a CPU Torch environment. The original one-pair smoke command was:

```text
python -m q4_rl.scst_train --output results/scst-smoke-8006000 --initialize-micro-warmstart results/scst-input/micro-warmstart.pt --initialize-sha256 cb5df8dd338cc740222fc0edd5259ea63a64ddd7cb12398d245e781eb7843757 --workers 1 --cpu-budget 1 --batch-pairs 1 --minibatch-size 128 --scenario-start 8006000 --scenario-end 8006000 --max-batches 1 --max-wall-seconds 300
```

New experiments require a fresh output directory and explicit unused training scene range. Resume with `--resume <same-output>/latest.pt`, keeping the same numeric configuration and omitting initialization flags. The default absolute deadline remains Beijing 2026-09-12 10:00. No long SCST training, development/confirmation-set evaluation, formal simulator, or remote operation was started by this implementation subtask.
