# Micro induced-attention architecture ablation

This branch adds `--architecture induced` to the G1 micro trainer. It changes only the candidate network: the same public 13/50 features, legal candidates, controller, BC/PPO functions, undiscounted billed-time returns and training split remain in use. It does not establish a better policy and has not run a real scenario.

The new block projects each encoded candidate to width 16, exchanges information through 16 learned latent slots with two attention heads, and returns a residual to the original hidden-64 representation. There is one block and no dropout, positional encoding or identity embedding. The original mean/max context pooling, actor and critic follow it unchanged. Attention storage grows with 16 times the candidate count rather than its square. Parameters increase from 29,058 to 34,226.

The default `mlp` retains its original initialization, tensor outputs and checkpoint configuration shape. Induced checkpoints use architecture `q4-micro-candidate-induced-v1` with all five attention settings explicitly stored. The factory validates exact metadata; worker reconstruction, restore and `load_policy` use the same factory. Resuming an induced run without `--architecture induced` is rejected. No MLP or macro weights are migrated. Both next-run architectures must include the shared episode-journal fix from `a4fb730`.

Validation after journal integration: **27 tests passed in 5.01 seconds**, one CPU thread. This includes default-MLP exact equality, candidate permutation and padding, masked NaN isolation and gradients, CPU enforcement, metadata rejection, attention parameter updates under shared BC/PPO, save/restore RNG and next-update equality, interrupted-update replay, and legacy/new raw-evidence reading. Only fabricated features and mocked rollouts were used; no validation database, simulator or evaluation partition was accessed.

The short CPU benchmark used hidden 64, two warmups and 12 paired repetitions with alternating architecture order. Inference below is one tensor forward pass, without callback feature packing; PPO is the real shared update with packing, backward, clipping and Adam, batch size 16 and one epoch.

| Candidates | MLP forward ms | Induced forward ms | Ratio | MLP PPO ms | Induced PPO ms | Ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.680 | 1.686 | 2.480 | 11.657 | 18.561 | 1.592 |
| 440 | 1.385 | 3.410 | 2.461 | 64.855 | 98.080 | 1.512 |
| 636 | 1.528 | 3.703 | 2.422 | 91.275 | 137.532 | 1.507 |

Raw repetitions and measured source hashes are in `micro-attention-benchmark.json`. The benchmark preceded the journal integration; the journal changes driver output only and is outside these measured operations. The update cost is below 2x, while pure forward cost exceeds 2x. These timings justify at most a controlled trial with complete worker/learner CPU and wall accounting; they do not predict whole-episode throughput or quality. There is no task T/L because no task scenario was run.

For a paired launch, use the same explicitly reserved, unused **training** interval, random seed, resource allocation, journal implementation and remaining arguments for both architectures. From each fresh task run output, the only model difference is `--architecture mlp` versus `--architecture induced`:

```text
python -B -m q4_rl.micro_train --architecture induced --hidden 64 --workers 8 --cpu-budget 50 --warmstart-episodes 64 --batch-episodes 16 --max-decisions 512 --epochs 3 --minibatch-size 128 --max-wall-seconds 1800 --deadline 2026-09-12T10:00:00+08:00 --scenario-start <reserved-training-start> --scenario-end <reserved-training-end> --output runs/<supervisor-run>/<job>/training
```

The outer CPU supervisor owns the combined budget and synchronization. The existing macro/micro pair wrapper has an argument whitelist that does not yet admit `--architecture`; it also requires one macro and one micro module. Use a separately reviewed launch arrangement for two micro arms. This branch starts neither job and does not promote a checkpoint. A subsequent frozen comparison must retain all failures and report full-clear rate, failed clears, paired T/L, mean/tail time, uncertainty and complete computation costs.
