# Bounded CPU learner threads

`q4_rl.micro_train` and `q4_rl.memory_train` accept `--learner-threads 1|2|4`.
Only the complete BC/PPO update call temporarily changes Torch intra-op threads;
normal return, administrative stop and exceptions restore one thread. Sampling,
inference, inter-op and other numerical libraries remain single-threaded. CUDA
remains disabled. The existing losses, returns, actions and feature schemas are
unchanged. Update wall/CPU measurements include changing and restoring threads.

The default one-thread configuration retains its old checkpoint shape. Nondefault
values are saved in checkpoint config and update diagnostics. Resume must use the
same setting; changing it requires a separately named experiment. No old optimizer
or warmstart checkpoint is silently migrated to another thread configuration.

With multiple workers, `cpu-budget >= workers + learner-threads`; with one worker,
sampling and updates share one process serially, so `cpu-budget >= learner-threads`.
For example, eight workers and four learner threads reserve twelve CPUs. The bundle
launcher accepts this option only for the two declared trainers and still caps the
sum of all job budgets at fifty. Deployment must count other active task processes.

The local tests compare default model parameters, full Adam state and random streams
exactly against an unwrapped update driver, exercise real BC/PPO updates at two/four
threads, and verify interrupted update rollback and replay. Nondefault parallel
reductions can differ slightly numerically; they are an explicit throughput setting,
not evidence of better policy performance. No scene training or server job was run
while implementing this option.
