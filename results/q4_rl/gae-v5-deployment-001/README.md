# GAE v5 deployment preflight

Authoritative source: df36f0ce43a8b1e23d6ebf1cb1b0bba6f16ef01d, archive r2 SHA256
ab3a7d6e312306700e9a79b9b14092ce53f084c6e5a8e3fb39847e8c6802c464.
All 131 manifest-listed files retained their bytes after server testing. Both
BC256 initialization identities passed their strict G1/G3 loaders. Server tests:
142 passed in 11.67s, process affinity restricted to one CPU, CUDA disabled.

The four jobs are separately bounded to 32 completed PPO batches, 3600 wall seconds,
zero additional BC, five sampler workers and four learner threads each. Combined
declared training compute is36; the concurrent C evaluation declares11. The common
pool is CPU0..49, with the outer supervisor's unchanged eight-core reserve and a
20GiB free-disk stop guard. Preflight's effective affinity-bound capacity50 gives
an initial supervisor allowance42. Raw cgroup quota96 is not the authorized task budget.

The first source archive r1 (48 batches) was superseded before upload or launch and
is retained. Initial prestage rejected equivalent deadline strings (22:00UTC versus
06:00+08:00) before running tests. Versioned prestage-r2 validates the same timestamp;
neither production source nor config was changed by that correction. Both attempts,
all plans/scripts, source archives and object readback receipts are preserved here.

DEPLOYMENT_RECEIPT.json is the portable checked summary; server-PRESTAGE.json and
server-pytest.log are the original object readbacks. RELEASE_MANIFEST.json and
train_gae_v5.json are exact bytes extracted from the authoritative archive.
Files were copied from handoff with per-file SHA256 equality checked. No training
was launched by this preflight; root reviews and records the subsequent live launch.
