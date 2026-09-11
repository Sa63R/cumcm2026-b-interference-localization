"""The runtime audit must not hide trajectory or virtual-cost differences."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from experiments.runtime_equivalence_benchmark import (
    BASELINE_COMMIT, COMPARISONS, HARNESS_FILES, SPEC_PATH, SUMMARY_RUNTIME_KEYS,
    compare_pair, digest, parse_seeds, physical_bound, source_identity, without_keys,
)


def test_only_declared_runtime_fields_are_removed_recursively():
    first = {"program_runtime_s": 1, "virtual_time_s": 3500,
             "strategy_parameters": {"planning_log": [{"runtime_s": 2, "score_s": 3}],
                                     "total_runtime_s": 4},
             "action_history": [{"virtual_time_s": 50, "position": [0, 1]}]}
    second = deepcopy(first)
    second["program_runtime_s"] = 999
    second["strategy_parameters"]["planning_log"][0]["runtime_s"] = 999
    second["strategy_parameters"]["total_runtime_s"] = 999
    clean_first = without_keys(first, SUMMARY_RUNTIME_KEYS)
    assert clean_first == without_keys(second, SUMMARY_RUNTIME_KEYS)
    second["action_history"][0]["virtual_time_s"] += 1
    assert clean_first != without_keys(second, SUMMARY_RUNTIME_KEYS)
    assert first["program_runtime_s"] == 1  # Raw evidence is not mutated.


def row_fixture():
    row = {field: 0 for field in COMPARISONS}
    row["action_step_sha256"] = [digest({"action": "measure", "position": [0, 0]}),
                                 digest({"action": "clear", "position": [10, 20]})]
    row["action_history_sha256"] = digest(row["action_step_sha256"])
    return row


def test_pair_rejects_changed_action_even_if_aggregate_cost_is_unchanged():
    baseline = row_fixture()
    candidate = deepcopy(baseline)
    candidate["action_step_sha256"][1] = digest({"action": "clear", "position": [20, 10]})
    comparison = compare_pair(baseline, candidate)
    assert not comparison["identical"]
    assert comparison["first_different_action_index"] == 1
    assert "action_step_sha256" in comparison["different_fields"]


def test_pair_rejects_missing_action_and_bad_certificate():
    baseline = row_fixture()
    candidate = deepcopy(baseline)
    candidate["action_step_sha256"].pop()
    candidate["completion_certified"] = 1
    comparison = compare_pair(baseline, candidate)
    assert comparison["first_different_action_index"] == 1
    assert "completion_certified" in comparison["different_fields"]


def test_seed_ranges_are_inclusive_and_reject_duplicates():
    assert parse_seeds("6000..6002,972000") == [6000, 6001, 6002, 972000]
    with pytest.raises(ValueError):
        parse_seeds("6000..6002,6002")
    with pytest.raises(ValueError):
        parse_seeds("6002..6000")


def test_physical_bound_does_not_add_empty_channel_certification_cost():
    case = SimpleNamespace(case_id="tiny-bound-test", sources=[SimpleNamespace(channel=1, x=120., y=0.)])

    def graph_solver(first, edges):
        assert first == [100. - 1e-6]
        assert edges == [[0.]]
        return first[0], [0]

    result = physical_bound(case, graph_solver)
    assert result["physical_lower_bound_s"] == (100. - 1e-6) / 5 + 5
    assert result["clear_action_lower_bound_s"] == 5


def test_deployed_snapshot_verifies_all_manifest_files_and_critical_coverage(tmp_path):
    names = ["src/strategies/example.py", *HARNESS_FILES, SPEC_PATH, "research/protocol.json"]
    hashes = {}
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {"commit": BASELINE_COMMIT, "source_dirty": False, "source_sha256": hashes}
    identity_path = tmp_path / "SOURCE_SNAPSHOT.json"
    identity_path.write_text(json.dumps(metadata), encoding="utf-8")
    assert source_identity(tmp_path)["git_head"] == BASELINE_COMMIT
    (tmp_path / "research/protocol.json").write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen source snapshot differs"):
        source_identity(tmp_path)
    del metadata["source_sha256"]["research/protocol.json"]
    del metadata["source_sha256"][SPEC_PATH]
    identity_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="omits benchmark-critical"):
        source_identity(tmp_path)
