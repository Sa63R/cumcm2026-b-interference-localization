# Local G3 h128 learner thread microbenchmark

The local fixed-work test supports a **controlled two-thread learner trial**.
Four threads were slower than two and used substantially more CPU. Production
trainers, live servers, worker settings and checkpoints were not changed.

| Torch intra-op threads | Three timed wall seconds | Median wall | Median process CPU | Wall speedup vs one | CPU ratio vs one |
|---|---|---:|---:|---:|---:|
| 1 | 6.6185, 8.4957, 7.7861 | 7.7861 | 7.6406 | 1.000 | 1.000 |
| 2 | 4.3975, 4.6991, 4.4354 | 4.4354 | 8.7188 | 1.755 | 1.141 |
| 4 | 5.0832, 5.3019, 5.1070 | 5.1070 | 19.4219 | 1.525 | 2.542 |

Each independent child had the same four-logical-CPU affinity [0,1,2,3],
interop=1, other-library environment thread limits=1, CUDA disabled. Children
ran sequentially. All measured child startup, loading, warmup, trials and probe
checks consumed **157.34375 process CPU seconds** in total, below the five-minute
CPU limit. No model parameters or usable trained checkpoints were exported.
An initial preflight failed before loading/training because optional psutil was
unavailable; affinity was implemented with the Windows standard-library API.

Inputs were already verified immutable G3 warmstart weights after 256 BC episodes,
zero PPO and no pending batch: SHA256
`61d4ac338cdfc0b62fcbb0cf9c4f2baaf9f832c8d203ac8d33a0f9b308f5ce49`.
The model is the actual 13-global/58-candidate MLP with hidden width 128.
The predeclared first 512 teacher records from the first BC batch comprise all
355 records of TRAIN 8012000 and the first 157 of TRAIN 8012001. Candidate counts
range 1..440, mean 237.580078125, total 121641. Input hashes are recorded; no
validation or later batch was used and no files were downloaded again.

Each thread setting ran one full warmup, then three trials. Every trial restored
the same weights, created a fresh Adam at 0.0003 and reset Python/Torch RNG to
424445. Timing begins after reset and covers the unmodified production
`imitation_update`: three epochs, minibatch 128, twelve optimizer steps,
including full feature packing and finite-value checks on every minibatch.
Warmup/loading, reset and diagnostic output probing are outside trial timing.
Trials are independent measurements, not successive training improvements.

All losses and resulting parameters were finite. Mean imitation loss was exactly
1.7951742211977642 in every trial. The fixed first-four-observation probe showed
maximum differences against one thread of 4.768371582e-6 for legal logits and
4.768371582e-7 for values, within predeclared atol=2e-4/rtol=1e-4; loss tolerance
was atol=rtol=1e-5. This checks approximate numerical agreement, not byte-identical
models or guaranteed identical later stochastic trajectories.

The two-thread wall range is below all one-thread observations, but this is one
local machine, one fixed data prefix, fixed order 1/2/4 and three trials. It does
not prove server speedup or end-to-end training speedup. Extra learner threads
must remain inside the total concurrent CPU budget; samplers stay one-threaded.
Serialization, raw output, rollout and terminal auditing were not timed here.
There is no policy rollout, actual task-time result or T/L claim in this benchmark.

`run.py` is the bounded runner; `threads-*.json` and `timings.json` contain raw
timings, source/input hashes and non-parameter numerical probes. Run from the
repository with the existing CPU Python environment after restoring the verified
immutable inputs under `handoff/v4-bc-fit-readback/g3_h128`. Existing timings are
protected from overwrite. No further thread settings were tried.
