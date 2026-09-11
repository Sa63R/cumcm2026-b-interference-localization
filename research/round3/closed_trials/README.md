# Closed round3 trials

The bundles retain full trial commits above the protected baseline; source/config/test receipts and completed raw runs remain archived. Deleting a rejected new branch is not a theorem of impossibility.

To restore feedback_envelope in this repository:

```powershell
git fetch research/round3/closed_trials/feedback_envelope.bundle refs/heads/research/q3-r3-feedback-envelope:refs/heads/research/q3-r3-feedback-envelope
git worktree add ../q3-r3-feedback-envelope research/q3-r3-feedback-envelope
```

Required base: `760af8230a8e5bda6050663b19d871a9701c9829`. Bundle SHA and rejection evidence are in `../protocol.json`. Existing immutable batch manifests refer to their original workspace paths; restore worktrees there to rerun, or prepare a new output after updating recipe paths. Never rewrite old evidence to claim a new evaluation.

## Directed localization and incumbent-only repair

Both failed the registered whole-episode gate. The shared `directed_lineage.bundle` contains both full trial refs above the same 760af823 base, including implementations, configurations and tests. The independent pilots, five-arm ablation, physical audits and causal diagnostics remain in the archive. Restore only when needed:

```powershell
git fetch research/round3/closed_trials/directed_lineage.bundle refs/heads/research/q3-r3-directed-localization:refs/heads/research/q3-r3-directed-localization refs/heads/research/q3-r3-directed-incumbent:refs/heads/research/q3-r3-directed-incumbent
git worktree add ../q3-r3-directed-localization research/q3-r3-directed-localization
git worktree add ../q3-r3-directed-incumbent research/q3-r3-directed-incumbent
```

Bundle SHA-256: `ce11353a8cadfd7baf704e2b8c13b0cf8c2f4871ec5398a534eb2dce2e9ba032`. Only the newly-created failed trial branches/worktrees are cleaned up after this archive is pushed. The selected derived baseline and frozen RL reference remain available.
