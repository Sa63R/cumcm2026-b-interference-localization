# Frozen endpoint development comparison

Before viewing the completed pair-v2 summary, fix this 12-arm panel: R8, unchanged micro rule512, original macro PPO128, pair-v2 macro PPO512 and micro PPO512, the shared original micro BC initializer, and the four bundle-v3 last-complete trained endpoints. Both from-scratch models also have their own BC endpoint, allowing architecture and training effects to be distinguished.

Use all 32 development seeds 8101000 through 8101031, all eight predeclared synthetic families and both source modes, exactly matching the pair-v2 scene requests. This yields 512 scenarios, 6144 runs and 32 independent seed clusters. Six evaluator workers, one numerical thread each, share the task CPU allowance with any concurrent training. Use the unchanged common lower bound, actual all-task billing, safety audits and 5000 seed-cluster bootstrap samples. Report random-family strata separately from the aggregate stress-weighted mixture.

Model aliases under models/eval-bundle-v3 are immutable copies of the last complete training checkpoint, never selected by best observed performance. Record their original checkpoint counters and SHA256 before execution. The original shared BC initializer is identical for initialized PPO and SCST. Equal training wall allowances do not mean equal numbers of scenes, updates or CPU seconds. No confirmation or final seeds are accessed. This development comparison can guide the next experiment, but cannot by itself promote a recommended model.

Remote command (inside the approved independent bundle-v3 directory, under the CPU supervisor):

```text
python experiments/q4_rl_evaluate.py --output runs/eval-bundle-v3/evaluation --specs configs/bundle-v3-evaluation-specs.json --stage development --start 8101000 --count 32 --families random minimum_radius boundary_outward cluster positive_error negative_error alternating_error narrow_strip --source-modes mixed all_directional --workers 6 --reference r8 --bootstrap-samples 5000
```

The family names match the frozen generator. All arms run in the same current source/environment; the evaluator freezes actual source, model and request hashes before its first case. All failures and raw feedback are retained.
