"""Same actions/returns, explicit v4 geometry, and exactly neutral migration."""

from argparse import Namespace
from copy import deepcopy

import numpy as np
import pytest

from research_rl.action_sets import action_schema, controller_for
from research_rl.controller import FEATURE_DIMS, CONTEXT_DIM, ALGORITHM_VERSIONS, feature_schema
from research_rl.joint_scan import JointScanRLSearch
from research_rl.route_debt import RouteDebtRLSearch, route_debt_features
from simulator_client.state import Position
from simulation import LocalResearchSimulator, random_scenario
from tests.test_strategy import ObservationOnlyClient


def test_debt_is_observed_unknown_obligation_not_a_future_completion():
    sites = tuple(Position(i * 100, 0) for i in range(7))
    pending = {p: set() for p in sites}
    pending[sites[0]] = {1, 2}
    pending[sites[1]] = {1}
    before = deepcopy(pending)
    q = Position(50, 0)
    values = route_debt_features(sites[0], [q, q], sites, pending.__getitem__, {1})
    row = values[q]
    assert len(values) == 1 and len(row) == 16
    assert row[:7] == [.05, 0, 0, 0, 0, 0, 0]
    assert row[7:14] == [.02, 0, 0, 0, 0, 0, 0]
    assert row[14:] == [50 / 3600, 1 / 7]
    assert pending == before  # Features do not mark pending/observed pairs done.
    # No new 16-source action pruning is folded into this representation trial.
    pending[sites[0]] = {17}
    row = route_debt_features(sites[0], [q], sites, pending.__getitem__, set(range(1, 17)))[q]
    assert row[0] == .05


def test_collinear_two_hop_has_zero_extra_and_empty_debt_is_finite():
    sites = tuple(Position(100 + i * 100, 0) for i in range(7))
    q, current = Position(50, 0), Position(0, 0)
    row = route_debt_features(current, [q], sites, lambda _: {1}, set())[q]
    assert row[7:14] == [0.] * 7
    assert row[14] == 50 / 3600
    empty = route_debt_features(current, [q], sites, lambda _: {1}, {1})[q]
    assert empty == [0.] * 16 and np.isfinite(empty).all()
    with pytest.raises(ValueError, match="finite"):
        route_debt_features(Position(float("inf"), 0), [q], sites, lambda _: set(), set())


def test_v4_schema_and_controller_are_explicit_base_only():
    assert FEATURE_DIMS["v4"] == 76
    assert feature_schema("v4")["action_semantics"] == feature_schema("v3")["action_semantics"]
    assert feature_schema("v4")["appended_features"][:36] == feature_schema("v3")["appended_features"]
    assert controller_for("v4", action_schema()) is RouteDebtRLSearch
    with pytest.raises(ValueError, match="base action"):
        controller_for("v4", action_schema("axis_quantiles"))


def capture_teacher(version, seed):
    observations = []
    simulator = LocalResearchSimulator(random_scenario(3, seed))
    cls = JointScanRLSearch if version == "v3" else RouteDebtRLSearch
    controller = cls(ObservationOnlyClient(simulator.client()), lambda features, context, target: target,
        feature_version=version, recorder=lambda f,c,a,t,s,cost: observations.append(
            dict(features=np.asarray(f, dtype=np.float32), context=np.asarray(c, dtype=np.float32),
                 action=a, teacher=t, cost=cost)))
    report = controller.run()
    assert simulator.evaluation()["all_cleared"]
    return report, observations


@pytest.mark.parametrize("seed", [110401, 110402])
def test_same_teacher_candidates_prefix_actions_and_costs(seed):
    old, old_records = capture_teacher("v3", seed)
    new, new_records = capture_teacher("v4", seed)
    assert old.action_history == new.action_history
    assert old.virtual_time_s == new.virtual_time_s
    assert len(old_records) == len(new_records)
    for left, right in zip(old_records, new_records):
        assert np.array_equal(left["features"], right["features"][:, :60])
        assert np.array_equal(left["context"], right["context"])
        assert (left["action"], left["teacher"], left["cost"]) == (right["action"], right["teacher"], right["cost"])
        assert np.isfinite(right["features"]).all()


def test_exact_batch_zero_residual_migration_and_trainable_new_features(tmp_path):
    torch = pytest.importorskip("torch")
    from research_rl.network import CandidateActorCritic, load_policy, pack_observations, architecture_spec
    from research_rl.distributions import distribution_spec
    from research_rl.train import initialize_from, episode, compute_returns, update, validate_resume, source_manifest
    torch.set_num_threads(1)
    torch.manual_seed(9021)
    old = CandidateActorCritic(16, 60).eval()
    new = CandidateActorCritic(16, 76).eval()
    payload = dict(algorithm=ALGORITHM_VERSIONS["v3"], feature_schema=feature_schema("v3"),
                   hidden=16, model=old.state_dict(), source_manifest=source_manifest())
    initialize_from(new, payload)
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())
    assert torch.count_nonzero(new.route_adapter.weight) == 0
    _, records = capture_teacher("v4", 110403)
    tensors = pack_observations(records)
    expected = old(tensors[0][..., :60].contiguous(), tensors[1], tensors[2])
    actual = new(*tensors)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    assert torch.equal(expected[0].argmax(-1), actual[0].argmax(-1))
    policy_path = tmp_path / "v4.pt"
    saved = dict(payload, algorithm=ALGORITHM_VERSIONS["v4"], feature_schema=feature_schema("v4"), model=new.state_dict())
    torch.save(saved, policy_path)
    assert load_policy(policy_path).feature_version == "v4"
    validate_resume(saved, Namespace(feature_version="v4", hidden=16))
    with pytest.raises(ValueError, match="architecture/version"):
        validate_resume(saved, Namespace(feature_version="v3", hidden=16))
    with pytest.raises(ValueError, match="discard"):
        initialize_from(old, saved)
    for kwargs in ({"architecture": architecture_spec("attention")},
                   {"action_distribution": distribution_spec(1)},
                   {"action_schema": action_schema("axis_quantiles")}):
        with pytest.raises(ValueError, match="MLP, flat"):
            CandidateActorCritic(16, 76, **kwargs)
    trajectories, metrics = episode((110404, new.state_dict(), 16, 908, False, 32, None, "v4"))
    assert metrics["success"] and metrics["reward_cost_s"] == pytest.approx(metrics["virtual_time_s"])
    compute_returns(trajectories)
    args = Namespace(bc_epochs=1, epochs=1, minibatch=32, clip=.2, value_coef=.5, entropy_coef=.01,
                     aux_bc_coef=0, max_grad_norm=.5, target_kl=.03)
    result = update(new, torch.optim.Adam(new.parameters(), lr=3e-4), trajectories, args)
    assert result["optimizer_steps"] > 0 and torch.count_nonzero(new.route_adapter.weight) > 0


def test_v4_construction_and_migration_preserve_rng_stream():
    torch = pytest.importorskip("torch")
    from research_rl.network import CandidateActorCritic
    from research_rl.train import initialize_from
    torch.manual_seed(891)
    old = CandidateActorCritic(16, 60)
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(891)
    new = CandidateActorCritic(16, 76)
    assert torch.equal(old_rng, torch.get_rng_state())
    assert all(torch.equal(value, new.state_dict()[key]) for key, value in old.state_dict().items())
    initialize_from(new, dict(algorithm=ALGORITHM_VERSIONS["v3"], feature_schema=feature_schema("v3"), model=old.state_dict()))
    assert torch.equal(old_rng, torch.get_rng_state())


def test_probability_preservation_metadata_does_not_infer_from_schema_names():
    pytest.importorskip("torch")
    from research_rl.network import CandidateActorCritic
    from research_rl.train import preserves_initial_probabilities
    old = dict(algorithm=ALGORITHM_VERSIONS["v3"])
    assert preserves_initial_probabilities(CandidateActorCritic(16, 76), old)
    assert preserves_initial_probabilities(CandidateActorCritic(16, 60), old)
    assert not preserves_initial_probabilities(CandidateActorCritic(16, 60),
                                               dict(algorithm=ALGORITHM_VERSIONS["v2"]))
    assert not preserves_initial_probabilities(CandidateActorCritic(16, 44),
                                               dict(algorithm=ALGORITHM_VERSIONS["v1"]))


def test_neutral_migration_preserves_sampled_and_greedy_trajectories():
    torch = pytest.importorskip("torch")
    from research_rl.network import CandidateActorCritic, TorchPolicy
    from research_rl.train import initialize_from, episode
    torch.set_num_threads(1)
    torch.manual_seed(9081)
    old, new = CandidateActorCritic(16, 60), CandidateActorCritic(16, 76)
    initialize_from(new, dict(algorithm=ALGORITHM_VERSIONS["v3"], feature_schema=feature_schema("v3"),
                              model=old.state_dict()))
    before, old_metrics = episode((110421, old.state_dict(), 16, 997, False, 64, None, "v3"))
    after, new_metrics = episode((110421, new.state_dict(), 16, 997, False, 64, None, "v4"))
    assert old_metrics["virtual_time_s"] == new_metrics["virtual_time_s"]
    assert len(before) == len(after)
    for a, b in zip(before, after):
        assert np.array_equal(a["features"], b["features"][:, :60])
        assert (a["action"], a["reward"], a["log_prob"], a["value"]) == (b["action"], b["reward"], b["log_prob"], b["value"])
    reports = []
    for version, model in (("v3", old), ("v4", new)):
        simulator = LocalResearchSimulator(random_scenario(3, 110422))
        cls = controller_for(version, action_schema())
        reports.append(cls(ObservationOnlyClient(simulator.client()), TorchPolicy(model),
                           feature_version=version, max_decisions=64).run())
        assert simulator.evaluation()["all_cleared"]
    assert reports[0].action_history == reports[1].action_history


def test_v4_cli_initialization_and_actual_resume(tmp_path):
    torch = pytest.importorskip("torch")
    from research_rl.network import CandidateActorCritic
    from research_rl.train import main
    source = tmp_path / "source.pt"
    old = CandidateActorCritic(16, 60)
    torch.save(dict(algorithm=ALGORITHM_VERSIONS["v3"], feature_schema=feature_schema("v3"),
                    hidden=16, model=old.state_dict()), source)
    output = tmp_path / "route"
    args = ["--output", str(output), "--device", "cpu", "--hidden", "16", "--feature-version", "v4",
            "--initialize-from", str(source), "--scenario-start", "110411", "--updates", "1",
            "--episodes-per-update", "2", "--workers", "0", "--epochs", "1", "--max-decisions", "12",
            "--max-wall-s", "60"]
    assert main(args) == 0
    initial = torch.load(output / "initialized.pt", weights_only=False)
    assert initial["state"]["initialization"]["preserves_initial_probabilities"]
    assert torch.count_nonzero(initial["model"]["route_adapter.weight"]) == 0
    assert torch.load(output / "latest.pt", weights_only=False)["state"]["update"] == 1
    index = args.index("--initialize-from")
    args[index:index+2] = ["--resume", str(output / "latest.pt")]
    assert main(args + ["--updates", "2"]) == 0
    assert torch.load(output / "latest.pt", weights_only=False)["state"]["update"] == 2
