# Retired experiment versions

`experiment/q4-rl-adaptive-cover` at `350e121124bfbfd602a8892e2d3fdc6906e7888f` is preserved by the pushed annotated tag `archive/q4-adaptive-cover-350e1211`. This retires the small-displacement heuristic prototype and its two unsuccessful four-case development trials, not adaptive coverage as a class. Its exact-certificate module was reused unchanged by the later shared-cover prototype.

Before deletion an independent detached checkout actually restored the archived revision. All 47 result files matched both the original checkout and Git blobs; 32 raw-record hashes, 118 archived-source hashes, both ZIP CRCs, configuration/freeze/summary hashes and the remote tag target passed. The immutable review record is `adaptive-cover-archive-review.json` (SHA256 `c6dd53bf81748ac345112b7a8f747672e90630d677aa8cc72c82f473e771d2df`).

After rechecking both clean worktrees and their exact revision, their resolved paths were verified beneath the workspace and removed with `git worktree remove`. The remote branch was deleted with an exact-commit lease and the local experiment branch removed. The archive tag still resolves to the same commit. No core branch was removed. To revisit the version, create a new worktree from that tag and materialize `results/q4_rl/adaptive-cover-smoke-v1` and `results/q4_rl/adaptive-cover-smoke-v2` if sparse checkout is inherited.
