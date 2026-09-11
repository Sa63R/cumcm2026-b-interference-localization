# Optional actor GAE and G3 BC initialization

G1 and G3 accept `--gae-lambda` in `[0,1]`, default `1`. Gamma remains exactly one;
there is no duration discount. For one complete executed episode, with old rollout
values `V`, the optional actor estimate is `delta_t = reward_t + V_(t+1) - V_t`,
`A_t = delta_t + lambda*A_(t+1)`, with terminal `V=0, A=0`. Reward is negative actual
step cost in the existing fixed 1000-second unit. Fallback is already in step cost;
only the separate terminal failure adjustment is additionally charged. Every episode
is processed independently before journaling, and invalid returned rows are still
preserved. No source identity, seed or hindsight bound enters the estimate.

Only actor advantages change. Shared PPO clipping, centering, entropy, gradients,
optimizer and critic's full undiscounted MC targets remain unchanged. At lambda one
the helper is a no-op and PPO follows the exact original float32 `returns-old_value`
operation. Default config/objective dictionaries retain their legacy shape. Nondefault
config binds `gae_lambda`; objective metadata records that actual lambda and explicitly
names `actor_advantage=gae`, `critic_target=undiscounted_mc`. Load/resume rejects a
mismatch. Raw records retain costs, returns, selected actions and old log probabilities.

G3 alone adds `--initialize-memory-warmstart <checkpoint> --initialize-sha256 <SHA>`.
The exact bytes are hashed before decoding with the strict G3 loader. The source must
have a positive completed BC target, zero PPO batches and no pending batch; architecture,
feature schema, CPU/objective and requested hidden size must agree. This is not a G1
migration. Fresh PPO requires zero additional BC and an explicit training seed interval;
it copies only model weights, creates fresh Adam, resets counters, and resets Python and
Torch action streams to the new experiment's declared RNG seed. `initial.pt` and every
later checkpoint retain the G3-specific source SHA/type binding without its filesystem
path. Resume uses the experiment's own `latest.pt` and embedded binding; omit the source
initialization flags on resume. A changed lambda or training config requires a new run.

The bundle launcher admits GAE only for G1/G3 and the G3 source option only for G3;
existing learner-thread and combined fifty-CPU checks remain in force. This implementation
does not launch the separately preregistered experiment. Numerical reduction differences
from multiple learner threads are independent of the GAE estimator comparison.

The [original GAE paper](https://arxiv.org/abs/1506.02438) motivates exchanging some
estimator bias for lower variance. Whether lambda below one helps this task is an
empirical question, especially with inaccurate value estimates. Preserving the billed
evaluation and MC critic target does not eliminate actor-estimator bias or establish
policy improvement, theoretical optimality, or convergence to a task optimum.

Verification uses tiny public-feature fixtures, no environment or simulator scenarios:
terminal/fallback/failure arithmetic, episode isolation, exact lambda-one model/Adam/RNG,
actual lambda-below-one PPO with unchanged critic targets, retained invalid rows,
interruption rollback/replay, strict metadata, G3 initialization and self-contained resume.

Implementation check: 142 tests passed in 22.42 seconds across the GAE, G3 initialization,
G1/G3 training, learner threads, bundle launcher, micro initialization/attention and journal
test modules. `git diff --check` passed. No training/evaluation scenario or server job ran.
