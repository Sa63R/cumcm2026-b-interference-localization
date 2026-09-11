"""Complete local CPU prefix interventions, leakage guards, and work accounting."""
import copy
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from research_rl import prefix_rollouts as pr
from research_rl.network import CandidateActorCritic


@pytest.fixture(scope="module")
def task():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(871)
        model = CandidateActorCritic(8, feature_dim=60)
    return dict(seed=3100001, weights=copy.deepcopy(model.state_dict()), hidden=8,
                action_seed=73, max_decisions=2)


@pytest.fixture(scope="module")
def complete(task):
    group, metrics = pr.collect_group(dict(task, capture_evidence=True))
    assert metrics["status"] == "complete", metrics
    return group, metrics


def test_complete_groups_include_all_tail_cost_and_exact_public_prefix(complete):
    group, metrics = complete
    assert group["features"].shape == (group["candidate_count"], 60)
    assert group["context"].shape == (12,)
    assert group["old_logits"].shape == (group["candidate_count"],)
    assert group["original_index"] == int(group["old_logits"].argmax())
    assert metrics["trajectories_started"] == metrics["trajectories_completed"] == 3
    assert metrics["trajectories_interrupted"] == 0
    assert metrics["successful_trajectories"] == 3
    labels = group["evaluated_candidates"]
    assert len({row["index"] for row in labels}) == 3
    baseline = metrics["evidence"][0]
    prefix_stop = baseline["decisions"][group["prefix_step"]]["before_public_action_count"]
    expected = pr.public_history(baseline["history"][:prefix_stop])
    assert pr._digest(expected) == group["prefix_sha256"]
    for label, evidence in zip(labels, metrics["evidence"]):
        assert label["cost_to_go_s"] == pytest.approx(label["total_time_s"] - group["prefix_cost_s"])
        assert label["failure_penalty_s"] == 0
        assert label["additional_completion_tail_s"] > 0  # max_decisions=2 forces real fallback
        assert label["cost_accounting_verified"]
        assert evidence["history"][-1]["action"] == "/exit"
        assert evidence["evaluation"]["all_cleared"]
        assert pr.public_history(evidence["history"][:prefix_stop]) == expected
        assert sum(e["cost_s"] for e in evidence["events"]) + label["additional_completion_tail_s"] == pytest.approx(label["total_time_s"])
    for label, evidence in zip(labels[1:], metrics["evidence"][1:]):
        assert label["forced_actions"] == 1 and label["prefix_verified"]
        forced = [d for d in evidence["decisions"] if d["selected_index"] != d["original_index"]]
        assert len(forced) == 1
        assert forced[0]["step"] == group["prefix_step"]
        assert forced[0]["selected_index"] == label["index"]
    actual_actions = sum(pr._physical_count(e["history"]) for e in metrics["evidence"])
    assert metrics["simulator_action_count"] == actual_actions
    assert metrics["accepted_request_count"] == actual_actions + 6  # each /enter and /exit
    assert metrics["replay_prefix_action_count"] == 2 * pr._physical_count(expected)


def test_physical_audit_has_the_original_dp_lower_bound(complete, monkeypatch):
    # Post-termination audit only: this never feeds LB or truth back to the actor.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "research/theory_v1"))
    from audit_eval_bounds import audit_record, physical_bounds
    group, metrics = complete
    bounds = []
    for evidence in metrics["evidence"]:
        sources, _ = audit_record(evidence)
        if not bounds:
            bounds.append(physical_bounds(list(sources.values()))["physical_clairvoyant_lower_s"])
        bound = bounds[0]
        assert evidence["row"]["virtual_time_s"] >= bound > 0
    times = [row["total_time_s"] for row in group["evaluated_candidates"]]
    print({"collector_smoke_seed": group["seed"], "physical_LB_s": bounds[0],
           "T_s": times, "T_over_LB": [value / bounds[0] for value in times]})


def test_deterministic_replay_and_no_large_evidence_by_default(task, complete):
    group, metrics = pr.collect_group(task)
    expected, _ = complete
    assert "evidence" not in metrics
    assert metrics["status"] == "complete"
    assert np.array_equal(group["features"], expected["features"])
    assert np.array_equal(group["old_logits"], expected["old_logits"])
    assert group["prefix_sha256"] == expected["prefix_sha256"]
    assert group["evaluated_candidates"] == expected["evaluated_candidates"]
    # Actor tensors and labels contain no raw world, private engine, or trace.
    assert not {"evaluation", "ground_truth", "history", "sources", "teacher"} & group.keys()


@pytest.mark.parametrize("seed", [6000, 100001, 2200001, 3000101, 5300001, 5400001, 5500001, 5600001])
def test_rejects_every_nontraining_partition_before_engine_construction(task, seed, monkeypatch):
    monkeypatch.setattr(pr, "LocalResearchSimulator", lambda *a, **k: pytest.fail("engine constructed"))
    with pytest.raises(ValueError, match="training ranges"):
        pr.collect_group(dict(task, seed=seed))


def test_expired_task_spends_no_simulator_budget(task, monkeypatch):
    monkeypatch.setattr(pr, "LocalResearchSimulator", lambda *a, **k: pytest.fail("engine constructed"))
    group, metrics = pr.collect_group(dict(task, deadline_epoch=0.0))
    assert group is None and metrics["status"] == "administrative_timeout"
    assert metrics["trajectories_started"] == metrics["simulator_action_count"] == 0


def test_expiration_between_branches_drops_whole_group_but_counts_baseline(task, monkeypatch):
    calls = iter([False, True])
    monkeypatch.setattr(pr, "_expired", lambda deadline: next(calls))
    group, metrics = pr.collect_group(task)
    assert group is None and metrics["status"] == "administrative_timeout"
    assert metrics["trajectories_started"] == metrics["trajectories_completed"] == 1
    assert metrics["baseline_total_time_s"] > 0 and metrics["simulator_action_count"] > 0


@pytest.mark.parametrize("failed_clear,all_cleared", [(1, True), (0, False), (2, False)])
def test_post_terminal_failures_keep_actual_time_and_one_explicit_penalty(task, monkeypatch, failed_clear, all_cleared):
    evaluate = pr.LocalResearchSimulator.evaluation
    def evaluation(simulator):
        result = evaluate(simulator)  # also proves evaluation was after termination
        result.update(failed_clear_count=failed_clear, all_cleared=all_cleared)
        return result
    monkeypatch.setattr(pr.LocalResearchSimulator, "evaluation", evaluation)
    group, metrics = pr.collect_group(dict(task, max_decisions=1, alternatives=1))
    assert metrics["status"] == "complete" and group is not None
    assert metrics["failed_trajectories"] == 2
    for row in group["evaluated_candidates"]:
        assert not row["success"]
        assert row["failure_penalty_s"] == 360000
        assert row["cost_to_go_s"] == pytest.approx(row["total_time_s"] - group["prefix_cost_s"] + 360000)
        assert row["total_time_s"] < 360000
    assert metrics["failed_clear_count"] == failed_clear * 2


def test_replay_mismatch_is_not_a_failure_reward_and_still_exits(task, monkeypatch):
    signature = pr._PrefixPolicy.signature
    def corrupt(self, *args):
        value = signature(self, *args)
        return "different" if self.expected is not None else value
    monkeypatch.setattr(pr._PrefixPolicy, "signature", corrupt)
    group, metrics = pr.collect_group(dict(task, capture_evidence=True))
    assert group is None and metrics["status"] == "invalid_execution"
    assert metrics["trajectories_started"] == 2
    assert metrics["trajectories_completed"] == 1
    assert metrics["trajectory_metrics"][-1]["error_code"] == "ReplayMismatch"
    assert metrics["evidence"][-1]["history"][-1]["action"] == "/exit"
    assert metrics["trajectory_metrics"][-1]["failure_penalty_s"] == 0


def test_rng_neutrality_and_fresh_weights_in_single_worker(task):
    pr._worker_model = None
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    first = pr._model_for(task)
    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    changed = {key: tensor.clone() for key, tensor in task["weights"].items()}
    changed["actor.2.bias"] += 3
    second = pr._model_for(dict(task, weights=changed))
    assert first is second  # bounded one-model cache, with replacement weights
    assert torch.equal(second.state_dict()["actor.2.bias"], changed["actor.2.bias"])


def test_standard_metadata_is_accepted_but_other_schema_rejected(task):
    model = pr._model_for(task)
    pr._validate_task(dict(task, architecture=model.architecture,
        action_schema=model.action_schema, action_distribution=model.action_distribution,
        feature_version="v3"))
    with pytest.raises(ValueError, match="unchanged"):
        pr._validate_task(dict(task, action_schema={"version": 1, "name": "range_probes"}))


def test_public_history_only_ignores_wall_timestamp():
    history = [{"action": "/measure", "response": {"real_timestamp_ms": 9,
        "virtual_time_s": 7.0, "measure_result": "direction", "svd_deg": 12.34}}]
    clean = pr.public_history(history)
    assert "real_timestamp_ms" in history[0]["response"]
    assert clean[0]["response"] == dict(virtual_time_s=7.0, measure_result="direction", svd_deg=12.34)


def test_prefix_sampling_is_uniform_including_one_choice_decisions(monkeypatch):
    class ZeroActor:
        architecture = {"version": 1, "name": "mlp"}
        action_distribution = {"version": 1, "name": "flat"}
        action_schema = {"version": 1, "name": "base"}
        def __call__(self, features, context, mask):
            return torch.zeros(mask.shape), torch.zeros(mask.shape[0])
    monkeypatch.setattr(pr._PrefixPolicy, "signature", lambda *args: "public")
    counts = [0] * 4
    for seed in range(1000):
        policy = pr._PrefixPolicy(ZeroActor(), lambda: [], rng=random.Random(seed))
        policy.control = SimpleNamespace(client=SimpleNamespace(state=SimpleNamespace(virtual_time_s=0.0)))
        for candidates in (1, 3, 2, 5):
            policy.prepare_candidates([None] * candidates, {})
            policy(np.zeros((candidates, 60)), np.zeros(12), object())
        counts[policy.selected_prefix["step"]] += 1
    assert all(190 <= value <= 310 for value in counts), counts


def test_one_choice_prefix_is_not_resampled_or_rerun(task, complete, monkeypatch):
    _, metrics = complete
    row = copy.deepcopy(metrics["trajectory_metrics"][0])
    monkeypatch.setattr(pr, "_run_trajectory", lambda *a, **k: dict(row=row,
        selected_prefix={"candidate_count": 1}, decisions=[], evidence=None))
    group, actual = pr.collect_group(task)
    assert group is None and actual["status"] == "no_choice"
    assert actual["trajectories_started"] == 1


def test_changed_world_is_rejected_even_when_public_prefix_matches(task, monkeypatch):
    evaluate = pr.LocalResearchSimulator.evaluation
    calls = 0
    def evaluation(simulator):
        nonlocal calls
        result = evaluate(simulator)
        calls += 1
        if calls == 2:
            result["ground_truth"]["description"] += "changed-after-prefix"
        return result
    monkeypatch.setattr(pr.LocalResearchSimulator, "evaluation", evaluation)
    group, metrics = pr.collect_group(task)
    assert group is None and metrics["status"] == "invalid_execution"
    assert metrics["trajectory_metrics"][-1]["error_code"] == "WorldIdentityMismatch"
    assert metrics["trajectories_started"] == 2


def test_full_continuation_cost_difference_and_physical_bound(task, monkeypatch):
    group, metrics = pr.collect_group(dict(task, seed=3100002, action_seed=17,
                                          max_decisions=8, capture_evidence=True))
    assert metrics["status"] == "complete"
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "research/theory_v1"))
    from audit_eval_bounds import audit_record, physical_bounds
    baseline = metrics["evidence"][0]
    sources, _ = audit_record(baseline)
    bound = physical_bounds(list(sources.values()))["physical_clairvoyant_lower_s"]
    labels = group["evaluated_candidates"]
    for row, evidence in zip(labels, metrics["evidence"]):
        audit_record(evidence)
        assert row["total_time_s"] >= bound
        assert row["cost_to_go_s"] - labels[0]["cost_to_go_s"] == pytest.approx(
            row["total_time_s"] - labels[0]["total_time_s"])
    assert len({row["total_time_s"] for row in labels}) > 1
    print({"collector_nonzero_difference_seed": group["seed"], "physical_LB_s": bound,
        "T_s": [row["total_time_s"] for row in labels],
        "T_over_LB": [row["total_time_s"]/bound for row in labels]})


def test_spawn_worker_matches_local_group(task, complete):
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        group, metrics = pool.submit(pr.collect_group, task).result(timeout=60)
    expected, _ = complete
    assert metrics["status"] == "complete"
    assert group["evaluated_candidates"] == expected["evaluated_candidates"]
    assert group["prefix_sha256"] == expected["prefix_sha256"]
    assert np.array_equal(group["features"], expected["features"])
