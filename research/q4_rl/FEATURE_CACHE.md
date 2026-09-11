# Decision-local micro geometry cache

Based on `89e518e0`, this isolated change caches only safe-point/readiness results per channel and region distance per `(point, channel)` during one candidate/feature construction. `decision_cache=False` retains the uncached path. A small base-controller distance hook shares the identical geometry calculation between pruning and micro features; formulas, iteration order, masks, public feature schema and frozen checkpoint compatibility are unchanged.

The cache starts fresh at candidate construction and is discarded after features, on construction failure, and before/after every physical request (including fallback and rejection). Public position, channel, virtual time or action-count changes also invalidate it. No cached object survives into policy selection or physical actions. Actual clear certificates and time/resource budget checks remain fresh. The Python finiteness checks were intentionally left unchanged.

All 37 relevant tests pass (29 existing controller checks plus 8 cache cases), including exact ordered candidates/features, reduced geometry calls, changed public state, new actual near/clear observations, request rejection, feature failure, next-decision geometry changes and fallback invalidation.

A single CPU-affined deterministic off/on pair used the existing synthetic training scene 8006000 and frozen micro BC SHA `cb5df8dd338cc740222fc0edd5259ea63a64ddd7cb12398d245e781eb7843757`. All 512 candidate lists and every numerical feature matched exactly in memory. Selected indices, complete action history, micro steps, clear certificates and all four billing subtotals also matched exactly. Both runs cleared all sources, retained 5 failed-clear attempts, and billed 8470.453858 s against the same lower bound 2096.484437 s: **T/L = 4.040313**. This is a repeated training-case CPU diagnostic, not independent efficacy evidence.

| Measured quantity | Cache off | Cache on |
|---|---:|---:|
| Feature wall time excluding comparison | 1.891710 s | 1.630497 s |
| Controller wall time excluding comparison | 2.910257 s | 2.518545 s |
| Raw controller CPU, including comparison | 2.843750 s | 2.562500 s |

The feature portion fell 13.8% in this pair. The fixed off/on order, network warm-up and normal timing variation prevent a statistical speed claim or attribution of the entire wall-time difference to caching. The original small JSON also retained per-step CPU-subtraction diagnostics; those are **not used** here because Windows process-clock granularity makes tiny per-step CPU differences unreliable. The reusable runner now reports only whole-controller CPU. No policy performance improvement is claimed.

`experiments/q4_feature_cache_diagnostic.py` repeats the exact comparison with a provided `--checkpoint`, explicit `--sha256` and fresh `--output`; run under a single-core affinity with `PYTHONPATH=src;.` on Windows. Large feature matrices are compared only in memory, never written as logs. Evidence is `results/q4_feature_cache/diagnostic-8006000.json`. No training, remote source change, validation database or formal simulator was used.
