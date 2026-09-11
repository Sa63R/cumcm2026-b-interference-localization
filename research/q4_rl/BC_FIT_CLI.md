# Reusable BC fit diagnostic

`experiments/q4_bc_fit_diagnostic.py` classifies fixed recorded synthetic TRAIN teacher inputs. It performs no rollout, weight update, normalization fitting, or database access. Accepted loaders are strictly `q4_rl.micro_network:load_policy` (G1, 13/50) and `q4_rl.memory_network:load_policy` (G3, 13/58). Set `PYTHONPATH` to the source tree providing that loader; the CLI does not copy or relax model implementations.

Example, using explicit local evidence paths:

```text
python experiments/q4_bc_fit_diagnostic.py --batches batch-000000-attempt-000000.json.gz batch-000001-attempt-000001.json.gz --checkpoint warmstart.pt --checkpoint-sha256 <64-hex-SHA256> --policy-loader q4_rl.memory_network:load_policy --output results/new-bc-fit
```

`--output` must be a new directory. `--batches` accepts explicit old gzip JSON-list batches and new EpisodeJournal gzip indices. Indexed episode files are loaded in order with SHA256/seed checks and directory-escape rejection. No input globbing, favorable-case selection, implicit database lookup or overwrite occurs. Non-TRAIN evidence, duplicated scenes/batch paths, PPO sampled-action labels and mismatched controller/feature schemas are rejected. Administrative episodes are counted separately; failed completed teacher episodes remain in classification and retain their penalized context.

`results.json` reports all records, by action kind, role, kind/role and input batch: top-1, cross entropy, teacher-action probability, equivalent-candidate probability/top-1, identical-input multiplicity and the associated cross-entropy floor. The equivalence class requires exact equality of all actual float32 candidate inputs within a decision. It also records every source file hash, checkpoint hash/schema, sample counts, elapsed/CPU time and original-teacher T/L context. That time ratio is never presented as the classified model's counterfactual performance.

Verification: eight small tests pass, including real strict G3 loading and 13/58 packing, cross-loader rejection, truncated features, old/indexed equality, hashes/path containment, forbidden inputs, output protection and retained failed episodes. One existing G1 batch (16 scenes, 5,650 records) took 18.93 seconds on one CPU; every summary value equals the old diagnostic's batch 1 exactly (`top1=0.6653097345132744`, `CE=1.9457639421331807`). Small regression evidence is in `research/q4_rl/bc_fit_cli_g1_one_batch`. The old four-batch diagnostic remains unchanged; it was not rerun.
