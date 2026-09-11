# Journal compression-level microbenchmark

I/O microbenchmark only, one prespecified complete synthetic TRAIN episode. No policy/training/validation execution or checkpoint changes.

Current production already uses one native json.dumps/UTF-8 encoding followed by gzip level 6; native serialization is not a new change here.

| Scheme | Mean CPU s | Mean wall s | Output MB | CPU speedup | Size / current | Peak working set MB |
|---|---:|---:|---:|---:|---:|---:|
| current_function | 1.192708 | 1.258177 | 3.782 | 1.000 | 1.0000 | 502.12 |
| level1 | 0.838542 | 0.891430 | 4.927 | 1.422 | 1.3026 | 502.16 |
| level3 | 0.880208 | 0.955740 | 4.508 | 1.355 | 1.1919 | 501.95 |
| level6_equivalence | 1.177083 | 1.213119 | 3.782 | 1.013 | 1.0000 | 502.08 |
| level9 | 2.536458 | 2.779659 | 3.626 | 0.470 | 0.9588 | 502.04 |

Five sequential subprocesses, each pinned to one permitted CPU; one warmup plus three measured complete writes per scheme. Atomic temporary.replace retained; no fsync, same as production. Input parsing and full decode/equality verification excluded from write timing.

Windows process-lifetime peak working set, including input parsing, warmup and prior verification; not incremental per-write allocation. Separate subprocess per scheme limits cross-scheme contamination.

Same json.dumps ensure_ascii=False, allow_nan=False and default separators; every decoded field preserved, and all 20 decompressed byte strings have identical SHA256.

This single episode can establish a compression tradeoff, not explain all 1190.6 seconds of SCST parent overhead. Multiprocessing transfer, model/state copies, checkpoint writes, index writes, parsing and other bookkeeping were not measured. No production change is made; validate other episode sizes and real end-to-end overhead before deployment.

Reproduce with `python research/q4_rl/journal_benchmark/benchmark.py` in the feature-cache worktree after selecting a fresh benchmark output path. Large temporary gzip outputs remain under handoff/journal-benchmark-8007000 and need not be committed.

Observed tradeoff for this episode:
- level1: 29.69% less CPU time, 30.26% larger files.
- level3: 26.20% less CPU time, 19.19% larger files.
- level6_equivalence: 1.31% less CPU time, equivalent file size.
- level9: 112.66% more CPU time, 4.12% smaller files.

These estimates use three measured repetitions; they do not guarantee production timing.

Level 3 is a candidate to test if approximately 19% larger logs are acceptable; level 1 buys slightly more CPU saving at approximately 30% larger logs. Level 9 is unattractive for this CPU-constrained sample. No level is promoted without an end-to-end storage/transfer check.
Gzip filenames differ, so tiny header-byte differences in the equivalent level-6 files do not indicate a payload change.
