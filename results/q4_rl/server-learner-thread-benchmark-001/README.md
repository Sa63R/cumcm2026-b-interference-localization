# Q4 server learner thread verification

The isolated server test supports **both two and four intra-op threads as
controlled follow-up candidates**. Four threads were fastest here, reversing the
local machine's four-thread result. No production thread setting was changed.

| Threads | Three update wall seconds | Median wall seconds | Median process CPU seconds | Wall speedup | CPU ratio |
|---|---|---:|---:|---:|---:|
| 1 | 5.0015, 4.9882, 4.9349 | 4.9882 | 4.9619 | 1.000 | 1.000 |
| 2 | 3.2309, 3.1914, 3.1634 | 3.1914 | 5.1445 | 1.563 | 1.037 |
| 4 | 2.3905, 2.3500, 2.6516 | 2.3905 | 5.7584 | 2.087 | 1.161 |

Relative to two threads, four reduced median update wall by 25.1% while adding
11.9% process CPU. Every four-thread wall measurement was below every two-thread
measurement. This is fixed order 1/2/4, three trials, one 512-record prefix under
the then-current shared-server load, not an end-to-end training speedup proof.
Any production trial must retain the shared 50-core default budget and count
learner threads together with samplers and all other concurrent jobs.

The exact script, frozen model, selected records and three update/network source
files have the same SHA256 values as the local benchmark. The checkpoint is G3
13/58, hidden128, complete BC256, zero PPO. Selected records are TRAIN8012000's
355 records plus TRAIN8012001's first157: 512 total, 1..440 candidates each,
mean237.580078125. Each setting used one full warmup then three independent
trials from identical weights, fresh Adam(0.0003), RNG424445, three epochs,
minibatch128 and twelve optimizer steps including unchanged feature packing.
No new trained weights or usable training checkpoint were exported.

All losses/parameters were finite. Maximum output differences against one thread
were 5.3883e-5/8.7261e-5 in legal logits for two/four threads; value differences
were 9.5367e-7/0. Loss differences were 9.9341e-9/2.9802e-8. All pass the same
predeclared local tolerances (output atol2e-4/rtol1e-4, loss atol/rtol1e-5).
Approximate equality does not guarantee bit-identical later training trajectories.

The Q4 port42222 hostname was checked before and after. Initial available disk
was38,789,185,536 bytes; final SSH metadata showed36,878,417,920 bytes, both above
20GiB. Each benchmark process had affinityCPU0-3, nice10, interop1 and other
library environment limits1; CUDA was disabled. Python3.10.12/Torch2.9.1+cpu came
from the existing rootA environment used read-only, with bytecode writes disabled
and TMPDIR under the new task root. All live B/C code, configuration and processes
were left untouched. Both benchmark and enclosing launcher had a360-second
timeout; exit code was0 and no benchmark processes remained afterward.

Total measured child CPU, including loading, warmup, timed trials and checks, was
**81.121118 seconds**; child wall totals were26.1109+18.7058+15.6204 seconds.
These are sequential child measurements, not the entire deployment/upload wall.
No simulator, validation database, task-time evaluation or T/L claim was involved.

All files traveled through the approved independent object prefix
`q4-rl-cpu-microbench-20260912`; server rclone commands used
`--s3-no-check-bucket`. SSH returned commands/process/identity metadata only.
One intermediate result snapshot was synchronized before completion. Final
object readback contains17 files/7,027,820 bytes; archive, script, model, index,
episode and update-source hashes were verified by `download_results.py`.

`readback/` preserves timing JSON, stdout/stderr, pre/post metadata, immutable
inputs, original123-file source archive and hash lists. Source archive SHA256:
`c37ddfed148a4713812d48c70d9501e9e6104e23fd8c93ee4d99712b611b39d9`.
`OBJECT_READBACK.json` records every received object hash. `server_run.sh` records
the isolated setup/run procedure. This directory is separate from the original
local benchmark evidence and does not overwrite it.
