"""Public-only cover certificates, physical ledger safety and trainable migration."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from localization.omni import OmniCandidateRegion
from research_rl.action_sets import action_schema, checkpoint_action_schema, controller_for
from research_rl.certified_cover import CertifiedCoverRLSearch, filter_certified_cover
from research_rl.controller import ALGORITHM_VERSIONS, Candidate, feature_schema
from research_rl.joint_scan import JointScanRLSearch
from simulation import LocalResearchSimulator
from simulation.cases import Scenario, Source
from simulator_client.rules import MAX_SOURCES
from simulator_client.state import ClientState, Position
from tests.test_strategy import ObservationOnlyClient


def public_controller():
    # No simulator or hidden world is needed for these public-state boundaries.
    return CertifiedCoverRLSearch(SimpleNamespace(state=ClientState()), lambda f, c, t: t)


def colocated_case():
    return Scenario("certified-cover-mechanism", 3, 110341,
                    tuple(Source(c, 0., 0., 1000.) for c in range(1, MAX_SOURCES + 1)))


def test_count_cap_is_exact_and_all_noncover_actions_keep_their_order():
    p = Position(0., 0.)
    candidates = [Candidate("cover", c, p) for c in range(1, 21)] + [
        Candidate("probe", 1, p), Candidate("fallback", 2, p), Candidate("clear", 3, p)]
    retained, counts = filter_certified_cover(candidates, set(range(1, MAX_SOURCES + 1)))
    assert retained == [c for c in candidates if c.kind != "cover" or c.channel in set(range(1, 17))]
    assert counts == {"known_count_cap": 4}
    assert Candidate("cover", 3, p) in retained  # A safe clear never removes this choice.
    assert retained[-3:] == candidates[-3:]
    retained, counts = filter_certified_cover(candidates, set(range(1, MAX_SOURCES)))
    assert all(c in retained for c in candidates if c.channel >= MAX_SOURCES)
    assert counts["known_count_cap"] == 0
    for inconsistent in (set(range(1, 18)), {0, *range(1, 16)}):
        assert filter_certified_cover(candidates, inconsistent) == (
            candidates, {"known_count_cap": 0})


@pytest.mark.parametrize("safe_evidence", ["near", "radius_19_9"])
@pytest.mark.parametrize("known_count", [1, MAX_SOURCES])
def test_safe_clear_preserves_known_cover_and_feature_semantics(safe_evidence, known_count):
    search = public_controller()
    search.detected = set(range(1, known_count + 1))
    for channel in search.detected - {1}:
        search.near_points[channel] = search.points[0]
    if safe_evidence == "near":
        search.near_points[1] = search.points[0]
    else:
        # A synthetic public outer region, not a hidden source observation.
        region = OmniCandidateRegion()
        region.vertices = ((-19.9, 0.), (19.9, 0.), (0., 10.))
        region._circle = None
        assert region.enclosing_disk().radius == pytest.approx(19.9)
        search.regions[1] = region
        search.first_bearings[1] = 0.
    search.client.state.current_channel = 1
    search.scan_focus = search.points[0]
    remaining = list(search.points)
    original = JointScanRLSearch._candidates(search, remaining)
    original_features, original_context = search._features(original, remaining)
    before = deepcopy(search.scan_ledger)
    retained = search._candidates(remaining)
    features, context = search._features(retained, remaining)
    indices = [i for i, candidate in enumerate(original) if candidate in retained]
    assert features == [original_features[i] for i in indices]
    assert context == original_context
    if known_count < MAX_SOURCES:
        assert retained == original  # No filtering at all below the count cap.
    else:
        assert retained == [c for c in original if c.kind != "cover" or c.channel in search.detected]
    assert sum(c.kind == "cover" and c.channel == 1 for c in retained) == 7
    assert any(c.kind == "clear" and c.channel == 1 for c in retained)
    chosen = retained[search._teacher(retained, remaining)]
    assert chosen.kind == "cover" and chosen.channel == 1 and chosen.point == search.scan_focus
    assert search.scan_ledger == before and 1 in search._pending(search.scan_focus)
    assert not search.report.coverage_complete
    search.blocked.add(1)
    blocked = search._candidates(remaining)
    assert not any(c.kind == "clear" and c.channel == 1 for c in blocked)
    assert sum(c.kind == "cover" and c.channel == 1 for c in blocked) == 7


@pytest.mark.parametrize("known_scanned", [False, True])
def test_teacher_handles_filtered_current_channel_or_scan_focus_and_live_source(known_scanned):
    search = public_controller()
    search.detected = set(range(1, MAX_SOURCES + 1))
    for channel in search.detected:
        search.near_points[channel] = search.points[0]
    search.scan_focus = search.points[0]
    search.client.state.current_channel = 17  # Unknown at the public count cap.
    if known_scanned:
        for point in search.points:
            search.scan_ledger[point].update(search.detected)
    search._refresh_coverage()
    before = deepcopy(search.scan_ledger)
    remaining = list(search.points)
    candidates = search._candidates(remaining)
    chosen = candidates[search._teacher(candidates, remaining)]
    if known_scanned:
        assert len(candidates) == MAX_SOURCES and all(c.kind == "clear" for c in candidates)
    else:
        assert len(candidates) == 7 * MAX_SOURCES + MAX_SOURCES
        assert chosen.kind == "cover" and chosen.channel == 1
    assert chosen.channel in search.detected
    assert search.scan_ledger == before
    assert search.report.learning["coverage_pairs_remaining"] == (28 if known_scanned else 140)
    assert not search.report.coverage_complete


def test_incomplete_known_count_cannot_filter_all_unknown_cover_choices():
    search = public_controller()
    search.detected = search.cleared = set(range(1, 11))
    search._refresh_coverage()
    candidates = search._candidates(list(search.points))
    assert len(candidates) == 70 and all(c.kind == "cover" for c in candidates)
    assert not search.report.completion_certified_under_model


def test_sixteenth_actual_clear_stops_without_fabricating_coverage():
    def discover_then_clear(features, context, teacher):
        unknown = next((i for i, row in enumerate(features) if row[44] == 1 and row[46] == 0), None)
        return unknown if unknown is not None else next(i for i, row in enumerate(features) if row[2] == 1)

    simulator = LocalResearchSimulator(colocated_case())
    search = CertifiedCoverRLSearch(ObservationOnlyClient(simulator.client()), discover_then_clear)
    report = search.run()
    evaluation = simulator.evaluation()  # Post-exit evaluator only.
    assert evaluation["all_cleared"] and evaluation["failed_clear_count"] == 0
    assert report.completion_reason == "source_count_upper_bound_reached"
    assert report.completion_certified_under_model and not report.coverage_complete
    assert report.coverage_points_visited == 0
    assert search.scan_ledger[search.points[0]] == set(range(1, 17))
    assert all(not search.scan_ledger[p] for p in search.points[1:])
    assert report.learning["coverage_pairs_remaining"] == 28
    assert report.learning["certified_cover_removed_sum"]["known_count_cap"] > 0
    assert set(report.learning["certified_cover_removed_sum"]) == {"known_count_cap"}
    assert simulator.observation_history()[-1]["action"] == "/exit"
    assert report.accepted_actions == evaluation["action_count"] == 34
    assert sum(report.time_breakdown.values()) == pytest.approx(report.virtual_time_s)
    # Pure original physical lower bound L/5+5N: colocated sources have L=0.
    lower_s = 5 * MAX_SOURCES
    assert report.virtual_time_s == 175. and report.virtual_time_s / lower_s == 2.1875


def test_decision_limit_keeps_the_original_finite_fallback_and_full_bill():
    histories = []
    for controller in (JointScanRLSearch, CertifiedCoverRLSearch):
        simulator = LocalResearchSimulator(colocated_case())
        search = controller(ObservationOnlyClient(simulator.client()), lambda f, c, t: t,
                            max_decisions=1)
        report = search.run()
        evaluation = simulator.evaluation()
        assert evaluation["all_cleared"] and evaluation["failed_clear_count"] == 0
        assert report.completion_certified_under_model
        assert report.learning["fallback_counts"]["decision_limit"] == 1
        assert report.learning["decisions"] == 1
        assert report.learning["fallback_virtual_time_s"] + 5 == pytest.approx(report.virtual_time_s)
        assert sum(report.time_breakdown.values()) == pytest.approx(report.virtual_time_s)
        assert report.virtual_time_s / (5 * MAX_SOURCES) >= 1
        histories.append(report.action_history)
    assert histories[0] == histories[1]


def test_base_registration_unchanged_and_new_schema_strict():
    assert controller_for("v3", action_schema()) is JointScanRLSearch
    assert controller_for("v3", action_schema("certified_cover")) is CertifiedCoverRLSearch
    with pytest.raises(ValueError, match="v3 feature"):
        controller_for("v2", action_schema("certified_cover"))
    with pytest.raises(ValueError, match="unchanged base"):
        controller_for("v4", action_schema("certified_cover"))
    with pytest.raises(ValueError, match="missing action schema"):
        checkpoint_action_schema({"args": {"probe_candidates": "certified_cover"}})
    with pytest.raises(ValueError, match="contradicts"):
        checkpoint_action_schema({"action_schema": action_schema(),
                                  "args": {"probe_candidates": "certified_cover"}})


def test_cpu_init_zero_real_ppo_update_resume_and_original_loader(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import research_rl.train as trainer
    from research_rl.network import CandidateActorCritic, load_policy
    torch.set_num_threads(1)
    torch.manual_seed(97121)
    parent = CandidateActorCritic(16, 60)
    path = tmp_path / "parent.pt"
    torch.save(dict(algorithm=ALGORITHM_VERSIONS["v3"], feature_schema=feature_schema("v3"),
                    hidden=16, model=parent.state_dict()), path)
    output = tmp_path / "run"
    args = ["--output", str(output), "--device", "cpu", "--hidden", "16", "--feature-version", "v3",
            "--probe-candidates", "certified_cover", "--initialize-from", str(path),
            "--scenario-start", "110351", "--scenario-end", "110360", "--updates", "0",
            "--episodes-per-update", "1", "--workers", "0", "--num-threads", "1", "--epochs", "1",
            "--bc-episodes", "0", "--max-decisions", "2", "--max-wall-s", "90", "--gae-lambda", "0.95"]
    assert trainer.main(args) == 0
    initial = torch.load(output / "initialized.pt", map_location="cpu", weights_only=False)
    assert all(torch.equal(v, initial["model"][k]) for k, v in parent.state_dict().items())
    assert initial["state"]["initialization"]["source_action_schema"] == action_schema()
    assert initial["state"]["initialization"]["target_action_schema"] == action_schema("certified_cover")
    assert not initial["state"]["initialization"]["preserves_initial_probabilities"]
    assert load_policy(output / "latest.pt").action_schema == action_schema("certified_cover")
    # Capture already-completed training episodes; do not run extra performance worlds.
    completed = []
    real_episode = trainer.episode

    def checked_episode(task):
        records, metrics = real_episode(task)
        assert len(records) == 2 and metrics["success"] and metrics["failed_clear_count"] == 0
        assert -sum(r["reward"] for r in records) * 1000 == pytest.approx(metrics["virtual_time_s"])
        completed.append(task[0])
        return records, metrics

    monkeypatch.setattr(trainer, "episode", checked_episode)
    resume = args[:]
    i = resume.index("--initialize-from")
    resume[i:i + 2] = ["--resume", str(output / "latest.pt")]
    assert trainer.main(resume + ["--updates", "1"]) == 0
    learned = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
    assert learned["state"]["update"] == 1 and learned["state"]["optimizer_steps"] > 0
    assert any(not torch.equal(v, learned["model"][k]) for k, v in initial["model"].items())
    assert trainer.main(resume + ["--updates", "2"]) == 0
    resumed = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
    assert resumed["state"]["update"] == 2 and resumed["state"]["episodes"] == 2
    assert completed == [110351, 110352]
    assert load_policy(output / "latest.pt").action_schema == action_schema("certified_cover")
    with pytest.raises(SystemExit):
        trainer.main(resume + ["--probe-candidates", "base"])
