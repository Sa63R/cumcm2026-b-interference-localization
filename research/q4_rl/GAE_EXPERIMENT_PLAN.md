# Finite-horizon credit-estimation experiment

Plan recorded 2026-09-11 19:06 UTC, before memory-v4 free-policy evaluation. Implementation and deployment checks must pass before launching. This is a new training experiment, not a silent resume of memory-v4.

Evidence motivating the experiment: completed bundle-v3 summaries still place the frozen macro-v2 PPO ahead of all new micro policies. Attention PPO and SCST improve some micro comparisons but do not beat macro-v2 or historical R8. Several hundred sequential choices and substantial fully charged fallback costs leave the current Monte Carlo actor estimator exposed to high variance. This is a hypothesis, not a demonstrated explanation of the performance gap. Representation, candidate quality and imitation distribution shift remain alternative explanations.

Compare G1/G3 hidden128, each with actor GAE lambda=1 and lambda=0.97, gamma=1 throughout. Initialize both algorithms within each schema from the exact same completed BC256 memory-v4 warmstart, not from a selected PPO checkpoint. Width128 was chosen before its policy result because the fixed teacher-input diagnostic showed better localization classification; this does not establish it as the best acting policy. The two schema initializations differ and are explicitly logged. The old one-thread memory-v4 endpoints remain untouched.

Frozen BC initialization identities:

- G1 h128: a3170afc53ae48d5b03618872400a424ce27d9c1110f7f2b342fdea13837688d
- G3 h128: 61d4ac338cdfc0b62fcbb0cf9c4f2baaf9f832c8d203ac8d33a0f9b308f5ce49

Fresh synthetic TRAIN reservation: 8022000..8031999, balanced eight-family/two-mode existing curriculum. Four jobs share the seed schedule and RNG initialization 424555, batch16, epochs3, minibatch128, Adam learning rate0.0003, entropy0.005, maximum512 policy decisions. Each job targets48 complete PPO batches (768 episodes), with an explicit3600-second wall allowance. If the wall/disk guard prevents reaching48, report actual exposure and do not present unequal-update endpoints as an equal-update ablation. Preserve administrative attempts and the last complete checkpoint; do not pick a favorable earlier endpoint. No additional BC is performed in the new jobs.

Use five one-thread rollout workers and four learner intra-op threads per job, inter-op and other numerical libraries one. Reserve nine CPUs per job,36 for four jobs; at most eleven concurrent evaluation compute processes keep declared training/evaluation compute at47 within the50-core common pool. Supervisor/load/disk limits may lower actual usage. No GPU. Stop this task's new training gracefully below20GiB free space, retaining raw evidence and synced checkpoints. Finite per-run bounds do not set an overall goal deadline.

GAE changes only the actor advantage estimator. The critic still learns the complete undiscounted Monte Carlo cost-to-go, including fallback and retained failure penalty. Each episode has terminal bootstrap zero; no recurrence crosses episode boundaries. Lambda1 follows the exact old tensor path, retaining default checkpoint compatibility. Lambda0.97 can reduce variance by introducing value-estimation bias; it is not proof of improved global planning or a license to discount billed time. No post-termination bound or hidden state enters model features, rewards or advantage estimates.

After all complete endpoints are frozen, evaluate all four on the same fixed512 and fresh128 DEVELOPMENT panels declared for memory-v4, retaining the frozen qualified R9, historical R8 and prior best eligible RL reference. Report the two panels separately. Include the initial BC endpoints and corresponding memory-v4 PPO endpoints as context; cross-wave wall/CPU differences must be explicit. Any recommendation requires an unused confirmation panel after a single model choice. Failed routes retain code, configuration, raw results and diagnosis until recoverable archival is verified.

Reference: Schulman et al., [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438). Its variance/bias tradeoff motivates this estimator test; its benchmark results do not establish efficacy for Q4.
