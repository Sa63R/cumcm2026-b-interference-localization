# First remote MLP pilot: prospective plan

This is a training and pipeline pilot, not independent efficacy confirmation.
The frozen R8 control remains the recommended strategy. All scenes come from
the isolated synthetic training partition; no validation database is opened.

## Configuration fixed before launch

- Candidate MLP, hidden 64; unchanged v1 public 10/16 feature schema and legal
  actions. The inherited source resolver and certified completion fallback
  remain fixed.
- 8 sampler workers, 16 episodes/batch, 64 imitation episodes followed by PPO.
- 128 learned decisions/episode, then the complete charged fallback. This
  optimizes a 128-decision prefix plus its fixed continuation, not an unlimited
  autonomous policy. Evaluation must use the same 128-decision limit.
- 3 epochs, minibatch 128, learning rate 0.0003, entropy coefficient 0.005.
- Gamma=lambda=1, undiscounted billed virtual time; fixed 1000-second scale.
  Lower bounds are computed only after rewards and termination for reporting.
- Initial 1800 wall seconds; every complete batch saves weights, optimizer,
  RNG, seed reservation and raw physical histories. An interrupted batch is
  retained and replayed on resume, never selectively included in updates.
- Supervisor default total ceiling 50 CPUs, all numerical libraries one
  thread, CPU-only Torch, lower scheduling priority, adaptive reduction under
  shared resource pressure. This small pilot normally uses far fewer cores.

## Development evaluation and continuation gates

Use the same 8 development seeds 8100000..8100007 and mixed/all-directional
modes for R8, heuristic-128, random initialization, imitation-only and the last
complete PPO checkpoint. Freeze each checkpoint hash and source before its
evaluation. Any longer decision horizon is a separate configuration ablation.
No confirmation/final partition may be used to choose the pilot configuration.

Report all-clear rate, failed clears, mean/median/p90/p95/max actual time,
common lower bound and ratio of mean time to mean bound; failed episodes remain
in penalized statistics. Paired uncertainty is clustered by seed, since the
two source modes from a seed are correlated. Keep raw adverse cases.

Also inspect fallback share, decision count, movement and switching costs,
actor entropy/KL, critic loss, full transaction recovery, CPU/peak-memory and
object readback integrity. Low imitation loss is not a sufficient gate: the
teacher breaks some observationally identical channel ties by ID, which is
intentionally absent from model inputs. A nonzero tie cross-entropy can be
unavoidable. The teacher is the same-action heuristic, not R8's full scheduler.

Continue or expand only after inspecting these diagnostics. Poor returns do
not by themselves prove PPO or joint decision making ineffective; separate
representation, action horizon, warm-start and optimization limitations. Do
not promote on this repeatedly inspected development set.

## Deployment failures preserved

The first source bundle omitted two imported geometry packages; the second
used epoch-zero tar timestamps incompatible with the evaluator's ZIP archive.
Both jobs failed before policy execution, and their logs remain under
`results/q4_rl/deployment-failure-*`. The corrected package passes an isolated
extraction plus evaluator `--freeze-only` run, including source.zip creation.

Remote PyPI downloads repeatedly stalled. Linux CPU dependency wheels were
downloaded separately and exchanged via the approved object-store prefix,
with byte hashes checked after upload and on the server. Only this task's pip
process was stopped; cached CPU Torch files and other tasks were preserved.
