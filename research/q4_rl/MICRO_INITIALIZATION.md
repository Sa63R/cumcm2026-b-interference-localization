# Frozen BC weights for a new PPO experiment

`micro_train` accepts the paired flags `--initialize-micro-warmstart PATH --initialize-sha256 SHA256`. The file bytes must match the explicit digest before checkpoint decoding. The shared `micro_initialization.load_micro_warmstart` then enforces the micro schema, CPU and objective contracts, complete positive BC count, `episodes == warmstart_completed == config.warmstart_episodes`, zero PPO batches and an explicitly empty pending transaction. The chosen PPO architecture and hidden size must match the source network metadata exactly.

Only weights transfer. A new Adam uses the new run's learning rate, all rollout/update counters reset, and Python/PyTorch RNG restart from the new `--random-seed` after model construction and weight loading. The source optimizer, source RNG and source training cursor do not transfer. The new output must be empty; the initial imported state is retained as `initial.pt`. Default scratch runs remain unchanged and retain `random.pt`.

Initialization requires `--warmstart-episodes 0` and explicit training seed start/end. The launcher must reserve an unused training interval disjoint from the source BC episodes. The configuration stores only `initialization = {type: frozen-micro-bc-weights-v1, sha256: ...}`, without a source path. Resume repeats the initialization flags, verifies the source content binding, then restores the latest new-run transaction without reapplying BC weights. An identical file at a different path is accepted; removing or changing the digest binding is rejected.

For the planned PPO/SCST comparison, use the same frozen BC bytes, new seed interval from 8007000, random seed, fresh Adam learning rate 0.0003, 512-decision limit and shared journal implementation. Each training scene receives one independently reserved `random.randrange(2**31)` action seed. SCST's second greedy rollout is additional computation and must be reported when comparing equal wall/CPU budgets; equal budgets do not imply equal sampled episodes. The separate scratch-MLP/scratch-attention comparison keeps its own matched 64-episode BC procedure.

Append these flags to the explicit PPO run configuration, using the copied source within the task directory:

```text
--initialize-micro-warmstart inputs/micro-warmstart.pt --initialize-sha256 cb5df8dd338cc740222fc0edd5259ea63a64ddd7cb12398d245e781eb7843757 --warmstart-episodes 0 --architecture mlp --hidden 64 --learning-rate 0.0003 --scenario-start 8007000 --scenario-end 8099999
```

Validation: **41 tests passed in 6.41 seconds** with one CPU thread, including 14 initialization tests and the existing attention/training/journal checks. Tests use tiny fabricated features and mocked rollouts. They verify checksum-before-decoding, incomplete/incorrect-source rejection, exact transferred weights, fresh optimizer/counters/RNG, architecture rejection, new-output protection, interrupted-update replay and content-bound resume. The real digest above was loaded read-only and matched the expected MLP-64 metadata: 64 BC episodes, zero PPO, no pending batch, source training seeds 8001000–8001063. No scenario, additional training run or validation data was used in this change.
