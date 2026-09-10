"""Bounded harness tests with a mocked engine/policy, never rollout benchmarks."""

import argparse
from copy import deepcopy
import gzip
import json
from types import SimpleNamespace

import pytest

from experiments import run_q3_rollout_comparison as driver


class FakeCase:
    def __init__(self, seed):
        self.seed = seed
        self.case_id = f"q3-random-{seed:04d}"

    def evaluation_config(self):
        return {"case_id": self.case_id, "problem": 3, "seed": self.seed,
                "sources": [{"channel": channel, "x": channel * 10, "y": 0} for channel in range(1, 11)]}


class FakeState:
    session = "new"
    cleared_count = 0
    virtual_time_s = 0.0

    def snapshot(self):
        return {"session": self.session, "cleared_count": self.cleared_count,
                "virtual_time_s": self.virtual_time_s}


class FakeClient:
    def __init__(self):
        self.state = FakeState()
        self.pending_request = None
        self.history = []

    def enter(self):
        self.state.session = "active"
        self.history.append({"action": "/enter"})

    def exit(self):
        self.state.session = "exited"
        self.history.append({"action": "/exit"})


class FakeSimulator:
    def __init__(self, case):
        self.case = case
        self._client = FakeClient()

    def client(self):
        return self._client

    def finish_for_evaluation(self):
        if self._client.state.session != "exited":
            self._client.state.session = "harness_finished"

    def evaluation(self):
        state = self._client.state
        assert state.session != "active"
        return {"source_total": 10, "cleared_total": state.cleared_count,
                "cleared_fraction": state.cleared_count / 10, "all_cleared": state.cleared_count == 10,
                "virtual_time_s": state.virtual_time_s,
                "mean_time_per_cleared_s": state.virtual_time_s / state.cleared_count if state.cleared_count else None,
                "measurement_count": 0, "failed_clear_count": 0,
                "action_count": len(self._client.history),
                "time_breakdown_s": {key: state.virtual_time_s if key == "movement_s" else 0 for key in driver.COMPONENTS},
                "ground_truth": self.case.evaluation_config()}

    def observation_history(self):
        return self._client.history


def install_mocks(monkeypatch, *, failure=None):
    calls = []

    def policy(client, *, problem, variant, max_actions, rollout_config=None):
        calls.append({"problem": problem, "variant": variant, "max_actions": max_actions,
                      "rollout_config": rollout_config})
        assert isinstance(client, FakeClient)
        client.enter()
        if variant == "rollout" and failure:
            raise failure("mock interruption/failure")
        client.state.cleared_count = 10
        client.state.virtual_time_s = 120.0 if variant == "efficient" else 100.0
        client.exit()
        planning = {} if variant == "efficient" else {
            "searches": 2, "candidate_evaluations": 6, "changed_decisions": 1,
            "planning_wall_time_s": .25, "fallback_counts": {"budget": 2},
            "decisions": [{"candidate_costs": [100, 120, 140]}],
        }
        result = SimpleNamespace(error=None, exit_error=None, completion_certified_under_model=True,
                                 cleared_count=10, virtual_time_s=client.state.virtual_time_s,
                                 completion_reason="mock_completed")
        result.as_dict = lambda: {"variant": variant, "planning": planning, "fixture_only": True}
        return result

    monkeypatch.setattr(driver, "LocalResearchSimulator", FakeSimulator)
    monkeypatch.setattr(driver, "random_scenario", lambda problem, seed: FakeCase(seed))
    monkeypatch.setattr(driver, "source_hashes", lambda: {"mock_source.py": "unchanged"})
    monkeypatch.setattr(driver, "run_search", policy)
    return calls


def args_for(tmp_path, *, stage="development"):
    configs = tmp_path / "configs.json"
    configs.write_text(json.dumps([{"name": "rollout_test", "rollout_config": {"particles": 2}}]), encoding="utf-8")
    return argparse.Namespace(configs=configs, output=tmp_path / "experiment", random_count=1,
                              random_seed_start=None, include_hard=False, allow_existing=False,
                              stage=stage, max_actions=20000)


def test_baseline_is_default_efficient_and_truth_stays_in_evaluator(monkeypatch):
    calls = install_mocks(monkeypatch)
    case = FakeCase(4000)
    baseline = driver.run_case(case, driver.BASELINE, None, 20000)
    rollout = driver.run_case(case, "rollout_test", {"particles": 2}, 20000)
    assert calls[0]["variant"] == "efficient" and calls[0]["rollout_config"] is None
    assert calls[1]["variant"] == "rollout" and calls[1]["rollout_config"] == {"particles": 2}
    assert set(calls[0]) == {"problem", "variant", "max_actions", "rollout_config"}
    assert baseline["row"]["case_sha256"] == rollout["row"]["case_sha256"]
    assert baseline["evaluation"]["ground_truth"] == rollout["evaluation"]["ground_truth"]
    assert rollout["summary"]["planning"]["decisions"]
    assert rollout["row"]["planning_searches"] == 2
    assert rollout["row"]["planning_fallback_counts"] == {"budget": 2}
    assert rollout["history"][-1]["action"] == "/exit"


def test_execute_archives_complete_records_and_resumes_only_identical_settings(tmp_path, monkeypatch):
    calls = install_mocks(monkeypatch)
    args = args_for(tmp_path)
    assert driver.execute(args) == 0
    assert len(calls) == 2
    manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["random_seed_interval_inclusive"] == [4000, 4000]
    assert manifest["source_sha256_start"] == manifest["source_sha256_end"]
    assert manifest["baseline"]["variant"] == "efficient"
    traces = list((args.output / "traces").glob("*.json.gz"))
    assert len(traces) == 2
    for path in traces:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            record = json.load(stream)
        assert {"summary", "evaluation", "history", "row"} <= record.keys()
    summary = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
    group = next(group for group in summary["groups"] if group["strategy"] == "rollout_test")
    assert group["planning_total_candidate_evaluations"] == 6
    assert group["planning_total_changed_decisions"] == 1
    assert group["planning_fallback_counts"] == {"budget": 2}
    with pytest.raises(ValueError, match="Output exists"):
        driver.execute(args)
    args.allow_existing = True
    assert driver.execute(args) == 0
    assert len(calls) == 2  # No completed strategy reruns or overwritten traces.
    monkeypatch.setattr(driver, "source_hashes", lambda: {"mock_source.py": "changed"})
    with pytest.raises(ValueError, match="Cannot resume"):
        driver.execute(args)


@pytest.mark.parametrize("failure,expected_exit", [(RuntimeError, 1), (KeyboardInterrupt, 130)])
def test_failure_or_interruption_keeps_full_trace_and_returns_nonzero(tmp_path, monkeypatch, failure, expected_exit):
    install_mocks(monkeypatch, failure=failure)
    args = args_for(tmp_path)
    assert driver.execute(args) == expected_exit
    with gzip.open(args.output / driver.trace_path("q3-random-4000", "rollout_test"), "rt", encoding="utf-8") as stream:
        record = json.load(stream)
    assert not record["row"]["successful"]
    assert record["row"]["accepted_exit"]
    assert record["row"]["error"]
    assert record["exception_traceback"]
    assert record["evaluation"]["cleared_total"] == 0
    assert (args.output / "runs.csv").is_file()
    assert (args.output / "summary.json").is_file()


def test_paired_wins_tail_and_planning_statistics_exclude_failed_pairs(monkeypatch):
    install_mocks(monkeypatch)
    template = driver.run_case(FakeCase(4000), "rollout_test", {"particles": 2}, 20000)["row"]
    rows = []
    for index, (base_time, candidate_time, success) in enumerate([(100, 80, True), (200, 240, True), (100, 1, False)]):
        for name, elapsed in [(driver.BASELINE, base_time), ("rollout_test", candidate_time)]:
            row = deepcopy(template)
            row.update(case_id=f"q3-random-{4000+index}", strategy=name, virtual_time_s=elapsed,
                       time_per_source_s=elapsed / row["source_total"],
                       successful=success if name != driver.BASELINE else True)
            rows.append(row)
    group = next(group for group in driver.summaries(rows) if group["strategy"] == "rollout_test")
    assert group["successful_pairs"] == 2
    assert group["paired_mean_seconds_saved"] == -10
    assert group["paired_wins"] == group["paired_losses"] == 1
    assert group["upper_tail_10pct_mean_virtual_time_s"] == 240
    assert not group["all_rows_eligible_for_time_comparison"]
    assert group["planning_fallback_counts"] == {"budget": 6}
    assert group["successful_mean_virtual_time_s"] == 160
    assert group["successful_mean_time_per_source_s"] == 16
    assert group["mean_virtual_time_s"] == 107


def test_holdout_default_seed_is_separate_and_configs_must_be_finite(tmp_path, monkeypatch):
    install_mocks(monkeypatch)
    args = args_for(tmp_path, stage="holdout")
    assert driver.execute(args) == 0
    manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["random_seed_interval_inclusive"] == [5000, 5000]
    args.configs.write_text('[{"name":"bad","rollout_config":{"min_gain_s":NaN}}]', encoding="utf-8")
    with pytest.raises(ValueError):
        driver.read_configs(args.configs)


def test_invalid_rollout_parameters_are_rejected_before_any_run(tmp_path, monkeypatch):
    calls = install_mocks(monkeypatch)
    args = args_for(tmp_path)
    for invalid in ({"x": 1}, {"particles": 0}, {"max_planning_s": -1}, {"candidates": True}):
        args.configs.write_text(json.dumps([{"name": "invalid", "rollout_config": invalid}]), encoding="utf-8")
        with pytest.raises(ValueError):
            driver.execute(args)
        assert not args.output.exists()
    assert not calls
