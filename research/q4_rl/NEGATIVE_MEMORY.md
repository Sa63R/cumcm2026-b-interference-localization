# Public negative-observation memory, first module

G1 keeps actual measured positions, fixed-cover credits and negative counts. Its 13/50 feature vector exposes the count but does not describe where those negative replies occurred or which source headings remain compatible with them. Two histories with the same negative count can therefore offer different future scan information without the actor seeing that difference. Positive localization geometry and first/latest bearing anchors are already represented separately.

The physical contract was checked directly against supplied Appendix 2, section 2 and measure/clear response descriptions, and against `simulator_client/rules.py` and `simulation/engine.py`: sources lie in the closed 1800 m disk; reception radii are 1000–1500 m; directional reception covers a closed 180-degree sector. A point within 5 m can still be behind a directional source and return no signal. Clear success depends on distance rather than heading. `no_signal` can mean absent, out of range, or behind the source.

`q4_rl.negative_memory.NegativeObservationMemory` is an independent module, not connected to the controller or trainer. Its immutable, data-independent bank contains 113 positions on a 300 m Cartesian lattice inside the source disk, radii 1000/1250/1500 m, and eight headings at 45-degree intervals plus omni: **3051 existence hypotheses**. It uses no RNG, source identity, case identity, seed, truth, validation statistics or fitted distribution. Domain/radius endpoints are included. Visibility uses the engine's closed radius and distance-scaled `1e-12` dot-product roundoff convention.

Only an explicitly accepted `measure/no_signal` removes finite hypotheses. Repeating the same point/channel does not remove extra mass or increase the unique count; the accepted public history remains replayable. Positive responses mark observed presence but do not narrow this coarse bank using a ±1-degree bearing: precise positive geometry remains G1's responsibility. Actual clear success retires the channel, and later silence cannot pollute its pre-clear memory. Optical misses do not filter this module.

An absent source is a separate logical possibility, not one weighted row. Negative replies never rule it out; actual positive reception or clear success does. An empty existence bank only sets an empty-bank feature. It cannot establish channel absence, source location, scan redundancy, safe clear, or all-clear completion because off-grid positions/headings/radii remain possible. Discrete heading/type counts are also an arbitrary representation measure, not a source prior or calibrated probability.

The query API accepts `(channel, (x, y))` candidate pairs and returns eight numeric features: remaining negative-compatible fraction; candidate visibility among surviving rows; visible surviving fraction of the original bank; empty-bank flag; unique negative count capped/scaled by 512; distance to the nearest negative point capped/scaled by 3600 m; observed-presence flag; and absence-still-compatible flag. All features are in [0,1]. For an empty bank, visibility scores are zero and the explicit flag prevents interpreting zero as evidence of uselessness. Caller policy may learn to combine the scores with G1 positive geometry and route costs; this module neither prunes candidates nor changes a certificate.

```python
from q4_rl.negative_memory import NegativeObservationMemory
memory = NegativeObservationMemory()
memory.observe(action="measure", position=(-900., 0.), channel=1,
               result="no_signal", accepted=True)
features = memory.score_candidates([(1, (-900., 0.)), (1, (900., 0.))])
same_memory = NegativeObservationMemory.replay(memory.history)
```

Sixteen tests pass in 0.49 seconds on one CPU thread. They cover scalar/vectorized physical consistency, radius/sector/domain boundaries and heading wrap, near-backside silence, rejected actions, isolated channels, repeated observations, public replay, order/chunk equality, spatial distinctions at equal counts, empty-bank semantics, and unchanged old schema. No scene or simulator was run.

The numeric fixture benchmark uses 100 declared negative replies and a bounded 128-position visibility cache, then 100 repeated feature calls. With 440 candidates sharing 22 positions, mean CPU/wall per query is **2.50/2.55 ms**; a pessimistic 636-distinct-position cache-thrashing query costs **25.47/25.87 ms**. Raw ten-repeat timings and aggregate counters are in `negative-memory-benchmark.json`. Individual Windows CPU timings have 15.625 ms granularity, so zero individual CPU readings do not mean zero work. At 512 decisions these averages extrapolate to about 1.28 versus 13.04 CPU seconds of additional scoring, not measured whole-episode costs. Masks occupy about 61 KB and the bounded visibility cache at most 391 KB; temporary comparisons are chunked at 32 candidates. No task T/L exists for this array benchmark.

Next step, if selected: an explicitly versioned controller schema (for example 13/58), dedicated network/checkpoint/training wiring, and paired no-memory versus memory comparisons with identical legal actions, certificates, initial BC procedure and fresh training scenes. Count-only versus spatial-memory ablation should distinguish extra feature capacity from actual historical information. Coarse support sensitivity, known-source score quality, behavior after bank exhaustion, total CPU cost and any learned improvement remain untested. The old `q4-micro-g1-v1` 13/50 schema and all safety/completion logic are unchanged.
