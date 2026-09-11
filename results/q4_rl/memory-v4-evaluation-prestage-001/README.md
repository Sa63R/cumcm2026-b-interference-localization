# Isolated memory-v4 evaluation: prepared, not launched

The new server/object task is `q4-rl-memory-v4-eval-20260912`. The completed training task is unchanged. This compact directory preserves the portable source archive, exact compatibility manifest, frozen 13-arm specs, launch controls, predeclared protocol/amendment, and object-readback receipts. Models are preserved under this new task's `models/eval-memory-v4/` object prefix; all ten hashes are in `plan.json`.

The source package preserves all 122 original C release files from commit `57e02aacc933ef600df588ab81141399354261e6` byte for byte and adds only the two qualified R9 production files. The 32 R9-qualified production files that C already contains match exactly. The nine omitted practice/workflow orchestration files are not dependencies of R9, as verified by an isolated no-site import of the combined package. The original C release manifest is preserved inside the archive alongside the combined source manifest.

The server has passed all ten strict model loaders on CPU1, exact schema/dimension/endpoint checks, complete source/model/control hashes, and a no-launch preflight of all thirteen methods. The source archive SHA is `e7da2c7d98bad3eb65c35c02afe20c3ffc89fa9994c55a8b316c01ba99791c69`; frozen plan SHA is `cc4ec421fc9c9f86796cbfc2d6768bfe0c9680f8e97a10401702a5a722ba61a4`. The server receipt was returned only through object storage and is verified by `READBACK_VERIFIED.json`.

After root confirms the completed bundle-v3 raw eligibility audit and rechecks current disk/process resources, launch from the new task root with its read-only shared CPU Python:

```sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /home/dataset-assist-0/usr/lh/ysh/bwc/shumo/q4-deep-rl-20260911/.venv-cpu/bin/python launch/eval-memory-v4/run_panels.py --launch
```

Without `--launch`, the script only checks frozen inputs. The explicit launch runs `eval-memory-v4-fixed` (512 scenarios, 3-hour bound), then `eval-memory-v4-fresh` (128 scenarios, 1-hour bound), in separate new output directories. Both are DEVELOPMENT, and their distributions remain separate. All 13 arms run on each requested scene: 8,320 runs total. The current state-search reference is R9 probe, historical reference R8, and prior eligible RL reference macro-v2 PPO512. Each panel uses 5,000 paired seed-cluster bootstrap replicates; the extra two reference reports reuse exactly the same full rows without new rollouts.

Each panel runs unchanged generic evaluation with ten one-thread workers plus its parent, under the existing adaptive supervisor in the common CPU0–49 pool, reserve8, 20GiB free-space guard, and periodic/final object synchronization. The requested 50-core pool includes sampling/inference/statistics and sync. The next panel requires complete summary/evidence and successful final synchronization. Fully recorded unsuccessful strategy rows are retained and may advance the queue; administrative partials, changed sources, or sync failures stop it. An interrupted output is never silently overwritten or rerun.

No evaluation, training, protected database access, or official/practice simulator run occurred during this preparation. The disk/process snapshot in the plan is preparation evidence only and must be refreshed immediately before launch.
