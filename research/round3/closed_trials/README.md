# Closed round3 trials

The bundles retain full trial commits above the protected baseline; source/config/test receipts and completed raw runs remain archived. Deleting a rejected new branch is not a theorem of impossibility.

To restore feedback_envelope in this repository:

```powershell
git fetch research/round3/closed_trials/feedback_envelope.bundle refs/heads/research/q3-r3-feedback-envelope:refs/heads/research/q3-r3-feedback-envelope
git worktree add ../q3-r3-feedback-envelope research/q3-r3-feedback-envelope
```

Required base: `760af8230a8e5bda6050663b19d871a9701c9829`. Bundle SHA and rejection evidence are in `../protocol.json`. Existing immutable batch manifests refer to their original workspace paths; restore worktrees there to rerun, or prepare a new output after updating recipe paths. Never rewrite old evidence to claim a new evaluation.
