"""New physical actions, exact evidence, safe completion, and model identity."""

from argparse import Namespace
from collections import defaultdict
import math

import pytest

from research_rl import run_rl_search
from research_rl.action_sets import action_schema, checkpoint_action_schema, controller_for
from research_rl.anypoint_scan import AnyPointCurrentRLSearch, AnyPointTargetsRLSearch
from research_rl.controller import ALGORITHM_VERSIONS, Candidate, feature_schema
from research_rl.joint_scan import JointScanRLSearch
from simulation import LocalResearchSimulator, Scenario, Source, random_scenario
from simulator_client.state import Position
from strategies.search import _StopSearch
from tests.test_strategy import ObservationOnlyClient


MODES = ("anypoint_current", "anypoint_targets")


def prefer_extra(features, context, teacher):
    return next((i for i, row in enumerate(features) if row[0] == 1 and row[18] > 1), teacher)


def ten_source_case():
    return Scenario("anypoint-ten-source-fixture", 3, 110401,
                    tuple(Source(c, 900 * math.cos(c), 900 * math.sin(c), 1000)
                          for c in range(1, 11)))


@pytest.mark.parametrize("mode", MODES)
def test_old_candidates_features_and_teacher_actions_are_preserved(mode):
    parent = controller_for("v3", action_schema(mode))

    class Checked(parent):
        def _features(self, candidates, remaining):
            old = JointScanRLSearch._candidates(self, remaining)
            assert candidates[:len(old)] == old
            features, context = super()._features(candidates, remaining)
            old_features, old_context = JointScanRLSearch._features(self, old, remaining)
            assert features[:len(old)] == old_features and context == old_context
            assert len(candidates) <= 140 + 16 * 6 + 5 * 20
            assert all(len(row) == 60 and all(math.isfinite(x) for x in row) for row in features)
            for candidate, row in zip(candidates[len(old):], features[len(old):]):
                assert candidate.kind == "cover" and candidate.channel not in self.detected
                assert row[0] == row[44] == 1 and row[18] in (7 / 6, 8 / 6)
                expected = (self.client.state.position.distance_to(candidate.point) / 5 + 5
                            + (candidate.channel != self.client.state.current_channel)) / 1000
                assert row[23] == pytest.approx(expected)
            return features, context

    scenario = random_scenario(3, 110402)
    first, second = LocalResearchSimulator(scenario), LocalResearchSimulator(scenario)
    added = Checked(ObservationOnlyClient(first.client()), lambda f, c, t: t).run()
    old = run_rl_search(ObservationOnlyClient(second.client()), policy=lambda f, c, t: t,
                        feature_version="v3")
    assert added.action_history == old.action_history
    assert added.virtual_time_s == old.virtual_time_s
    assert added.learning["anypoint_measurements"] == 0
    assert added.learning["anypoint_candidates_offered_sum"] > 0


@pytest.mark.parametrize("mode", MODES)
def test_actual_new_actions_are_unique_charged_bounded_and_complete(mode):
    simulator = LocalResearchSimulator(ten_source_case())
    report = run_rl_search(ObservationOnlyClient(simulator.client()), policy=prefer_extra,
                           feature_version="v3", probe_candidates=mode)
    evaluation = simulator.evaluation()
    assert evaluation["all_cleared"] and evaluation["failed_clear_count"] == 0
    assert report.completion_certified_under_model
    assert simulator.observation_history()[-1]["action"] == "/exit"
    assert report.learning["anypoint_measurements"] == 32
    assert report.learning["decisions"] <= 256
    assert sum(report.time_breakdown.values()) == pytest.approx(report.virtual_time_s)
    measured = defaultdict(set)
    newly_measured, extra_cost = [], 0.0
    previous_position, previous_time, previous_channel = Position(0, 0), 0.0, 1
    for action in report.action_history:
        point = Position(*action["position"])
        if action.get("rl_discovery_family"):
            assert action["action"] == "measure"
            assert action["channel"] not in measured[point]
            if action["rl_discovery_family"] == "current":
                assert point == previous_position
            else:
                assert mode == "anypoint_targets" and point != previous_position
            expected = previous_position.distance_to(point) / 5 + 5 + (action["channel"] != previous_channel)
            assert action["virtual_time_s"] - previous_time == pytest.approx(expected, abs=2e-6)
            extra_cost += action["virtual_time_s"] - previous_time
            newly_measured.append(action)
        if action["action"] == "measure":
            measured[point].add(action["channel"])
            previous_channel = action["channel"]
        previous_position, previous_time = point, action["virtual_time_s"]
    assert len(newly_measured) == report.learning["anypoint_measurements"]
    assert extra_cost == pytest.approx(report.learning["anypoint_virtual_time_s"])
    ledger = {Position(*entry["position"]): set(entry["channels"])
              for entry in report.learning["measurement_ledger"]}
    assert ledger == dict(measured)
    # Completeness still needs actual measurements at ALL seven sites on every
    # remaining channel, even though many outside-site observations occurred.
    assert any(action["position"] not in report.coverage_points for action in newly_measured)
    for point in report.coverage_points:
        assert measured[Position(*point)] | set(report.cleared_channels) == set(range(1, 21))


def test_noncover_evidence_is_exact_and_never_forges_coverage():
    simulator = LocalResearchSimulator(ten_source_case())
    controller = AnyPointCurrentRLSearch(ObservationOnlyClient(simulator.client()), prefer_extra)
    controller.client.enter()
    first, second = Position(123, 456), Position(123 + 1e-8, 456)
    initial = {point: set(channels) for point, channels in controller.scan_ledger.items()}
    controller._perform("measure", first, 20, "fixture")
    assert controller.scan_ledger == initial
    assert controller.report.learning["coverage_pairs_remaining"] == 140
    assert not any(c.option == 7 and c.channel == 20 for c in controller._candidates(controller.points))
    controller._perform("measure", second, 19, "fixture")
    assert controller.measurement_ledger[first] == {20}
    assert controller.measurement_ledger[second] == {19}
    # Feature float32/legacy six-decimal rounding is NOT evidence equality.
    extra = [c for c in controller._candidates(controller.points) if c.option == 7]
    assert any(c.point == second and c.channel == 20 for c in extra)
    assert not any(c.channel == 19 for c in extra)
    assert controller.scan_ledger == initial
    controller._perform("measure", controller.points[0], 20, "fixture")
    assert controller.scan_ledger[controller.points[0]] == {20}
    controller.client.exit()


def test_rejection_does_not_create_evidence_or_spend_extension_budget():
    simulator = LocalResearchSimulator(ten_source_case())
    client = simulator.client()
    controller = AnyPointTargetsRLSearch(ObservationOnlyClient(client), prefer_extra)
    client.enter()
    client.measure = lambda position, channel: {"accepted": False}
    with pytest.raises(_StopSearch, match="request_rejected"):
        controller._execute_candidate(Candidate("cover", 20, Position(123, 456), option=8), controller.points)
    assert not controller.measurement_ledger and controller.anypoint_measurements == 0
    assert controller.report.learning["coverage_pairs_remaining"] == 140
    client.exit()


def test_target_pool_is_bounded_and_comes_only_from_legal_source_destinations():
    simulator = LocalResearchSimulator(ten_source_case())
    controller = AnyPointTargetsRLSearch(simulator.client(), prefer_extra)
    controller.client.enter()
    for channel in range(1, 7):
        controller._perform("measure", Position(0, 0), channel, "fixture")
    old = JointScanRLSearch._candidates(controller, controller.points)
    candidates = controller._candidates(controller.points)
    extras = candidates[len(old):]
    source_points = {c.point for c in old if c.kind in ("probe", "clear")}
    target_points = {c.point for c in extras if c.option == 8}
    assert len(target_points) == 4 and target_points <= source_points
    assert all(c.channel not in controller.detected for c in extras)
    assert not any(c.option == 7 for c in extras)  # Origin pairs already exist.
    assert len({(c.point, c.channel) for c in extras}) == len(extras)
    controller.anypoint_measurements = 32
    assert controller._candidates(controller.points) == old
    controller.client.exit()


def test_sixteen_known_proves_only_unknown_channels_absent():
    simulator = LocalResearchSimulator(ten_source_case())
    controller = AnyPointTargetsRLSearch(simulator.client(), prefer_extra)
    controller.client.enter()
    controller._perform("measure", Position(0, 0), 1, "fixture")
    controller.detected.update(range(1, 17))
    before = JointScanRLSearch._candidates(controller, controller.points)
    after = controller._candidates(controller.points)
    assert after == before
    assert any(c.channel == 1 and c.kind == "probe" for c in after)
    assert not controller._unknown_channels()
    controller.client.exit()


def test_completed_channel_cover_proves_no_extra_measurement_needed():
    simulator = LocalResearchSimulator(ten_source_case())
    controller = AnyPointTargetsRLSearch(simulator.client(), prefer_extra)
    controller.client.enter()
    for point in controller.points:
        controller._perform("measure", point, 20, "fixture")
    controller._perform("measure", Position(123, 456), 19, "fixture")
    assert 20 not in controller._unknown_channels()
    assert not any(c.option in (7, 8) and c.channel == 20
                   for c in controller._candidates(controller.points))
    # Removing even one true observation removes that proof; a planned future
    # visit is never sufficient. The current position remains unmeasured on 20.
    controller.scan_ledger[controller.points[0]].remove(20)
    assert 20 in controller._unknown_channels()
    assert any(c.option == 7 and c.channel == 20 for c in controller._candidates(controller.points))
    controller.client.exit()


@pytest.mark.parametrize("decisions", [0, 3])
def test_decision_limit_retains_full_charged_baseline_completion(decisions):
    simulator = LocalResearchSimulator(ten_source_case())
    costs = []
    controller = AnyPointTargetsRLSearch(ObservationOnlyClient(simulator.client()), prefer_extra,
        max_decisions=decisions, recorder=lambda *args: costs.append(args[-1]))
    report = controller.run()
    assert simulator.evaluation()["all_cleared"]
    assert report.completion_certified_under_model
    assert len(costs) == decisions
    assert report.learning["fallback_counts"]["decision_limit"] == 1
    assert sum(costs) + report.learning["fallback_virtual_time_s"] == pytest.approx(report.virtual_time_s)


def test_action_budget_cannot_claim_success():
    simulator = LocalResearchSimulator(ten_source_case())
    result = run_rl_search(ObservationOnlyClient(simulator.client()), policy=prefer_extra,
                          feature_version="v3", probe_candidates="anypoint_targets", max_actions=3)
    assert result.completion_reason == "action_budget"
    assert not result.completion_certified_under_model and not result.coverage_complete
    assert simulator.observation_history()[-1]["action"] == "/exit"


@pytest.mark.parametrize("mode", MODES)
def test_checkpoint_identity_migration_and_cpu_update(mode, tmp_path):
    torch = pytest.importorskip("torch")
    from research_rl.network import CandidateActorCritic, architecture_spec, load_policy
    from research_rl.distributions import distribution_spec
    from research_rl.train import (initialize_from, preserves_initial_probabilities,
                                   validate_resume, source_manifest, episode, compute_returns, update)
    torch.set_num_threads(1)
    torch.manual_seed(4310)
    base = CandidateActorCritic(16, 60)
    model = CandidateActorCritic(16, 60, action_schema=action_schema(mode))
    payload = dict(algorithm=ALGORITHM_VERSIONS["v3"], hidden=16, model=base.state_dict(),
                   feature_schema=feature_schema("v3"), source_manifest=source_manifest())
    with pytest.raises(ValueError, match="action schema change"):
        initialize_from(model, payload)
    initialize_from(model, payload, allow_action_schema_change=True)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in base.state_dict().items())
    assert not preserves_initial_probabilities(model, payload)
    saved = dict(payload, model=model.state_dict(), action_schema=model.action_schema,
                 args={"probe_candidates": mode})
    with pytest.raises(ValueError, match="action schema differs"):
        validate_resume(saved, Namespace(feature_version="v3", hidden=16))
    validate_resume(saved, Namespace(feature_version="v3", hidden=16, probe_candidates=mode))
    with pytest.raises(ValueError, match="contradicts"):
        checkpoint_action_schema(dict(saved, args={"probe_candidates": "base"}))
    with pytest.raises(ValueError, match="unknown action schema"):
        checkpoint_action_schema(dict(saved, action_schema=dict(model.action_schema, max_extra_measurements=33)))
    with pytest.raises(ValueError, match="MLP and flat"):
        CandidateActorCritic(16, 60, architecture=architecture_spec("attention"), action_schema=model.action_schema)
    with pytest.raises(ValueError, match="MLP and flat"):
        CandidateActorCritic(16, 60, action_distribution=distribution_spec(1), action_schema=model.action_schema)
    path = tmp_path / "anypoint.pt"
    torch.save(saved, path)
    policy = load_policy(path)
    assert policy.action_schema == model.action_schema
    with pytest.raises(ValueError, match="action schema differ"):
        run_rl_search(LocalResearchSimulator(ten_source_case()).client(), policy=policy, probe_candidates="base")
    records, metrics = episode((110421, model.state_dict(), 16, 4311, False, 64, None, "v3",
                                model.architecture, model.action_distribution, model.action_schema))
    assert metrics["success"] and metrics["learning"]["action_schema"] == model.action_schema
    assert metrics["reward_cost_s"] == pytest.approx(metrics["virtual_time_s"])
    compute_returns(records, gae_lambda=.95)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    args = Namespace(epochs=1, minibatch=32, clip=.2, value_coef=.5, entropy_coef=.005,
                     aux_bc_coef=0, max_grad_norm=.5, target_kl=.03)
    result = update(model, torch.optim.Adam(model.parameters(), lr=1e-4), records, args)
    assert result["optimizer_steps"] > 0
    assert any(not torch.equal(value, model.state_dict()[key]) for key, value in before.items())


def test_cli_new_trial_and_resume_preserve_mode_and_migration_claim(tmp_path):
    torch = pytest.importorskip("torch")
    from research_rl.network import CandidateActorCritic
    from research_rl.train import main
    model = CandidateActorCritic(16, 60)
    path = tmp_path / "base.pt"
    torch.save(dict(algorithm=ALGORITHM_VERSIONS["v3"], hidden=16, model=model.state_dict(),
                    feature_schema=feature_schema("v3")), path)
    output = tmp_path / "trial"
    args = ["--output", str(output), "--device", "cpu", "--hidden", "16", "--feature-version", "v3",
            "--probe-candidates", "anypoint_targets", "--initialize-from", str(path),
            "--scenario-start", "110431", "--scenario-end", "110432", "--max-attempted-episodes", "2",
            "--bc-episodes", "0", "--updates", "1", "--episodes-per-update", "1", "--workers", "0",
            "--epochs", "1", "--max-wall-s", "90", "--max-decisions", "64"]
    assert main(args) == 0
    initial = torch.load(output / "initialized.pt", map_location="cpu", weights_only=False)
    assert initial["state"]["initialization"]["preserves_initial_probabilities"] is False
    assert initial["state"]["initialization"]["source_action_schema"] == action_schema()
    index = args.index("--initialize-from")
    args[index:index + 2] = ["--resume", str(output / "latest.pt")]
    assert main(args + ["--updates", "2"]) == 0
    latest = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
    assert latest["state"]["update"] == 2
    assert latest["action_schema"] == action_schema("anypoint_targets")
    assert latest["state"]["attempted_episodes"] == 2
