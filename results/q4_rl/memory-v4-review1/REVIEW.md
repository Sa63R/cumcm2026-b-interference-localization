# First memory-v4 progress review

At 2026-09-11 18:16:46 UTC, about 29 minutes after launch, all four jobs had
finished BC256 and entered PPO. Completed batches/episodes were G1-h64 29/464,
G3-h64 26/416, G1-h128 21/336, G3-h128 20/320. Therefore their completed PPO
batch counts were 13, 10, 5 and 4. Each latest sampled batch completed 16/16.
These batches contain different TRAIN requests and stochastic policy samples;
their training T/L values must not be ranked as a paired performance test.

All supervisors and learners remained alive. Load averages were
30.11/26.97/25.46; the shared task volume reported approximately 39 GiB free,
above this run's 20 GiB guard. A corrected affinity check verified all 62
observed threads in both task process groups within CPU0-41. There were 48
processes including wrapper/resource-tracker helpers, versus 43 declared
sampling/learner/evaluator compute processes. The common maximum pool remains
CPU0-49; numerical computation is CPU-only. The first affinity command stopped
because the remote zsh reserves the variable name `status`; it did not change
process settings, and the complete check was rerun with a task-specific name.

Decision: continue the predeclared one-hour allowance, without extending it or
selecting a checkpoint from these diagnostics. The fixed BC-input diagnostic
can investigate underfitting; the separately preregistered fixed/fresh paired
development evaluations determine free-policy performance. Model promotion
still requires unused independent confirmation.

The four immutable progress copies were made inside the approved task and
read back through its approved object-store prefix. They retain completed
batch provenance and measured costs; no trajectory was transferred via SSH.
