"""Pure small graphs and constructed saved records; no policy/simulator runs."""

import copy
import gzip
import itertools
import json
import math
from pathlib import Path

import pytest

from experiments.q4_comparison_bounds import (
    ROUNDING_FACTOR, analyze_directory, common_bound, compare_records, digest,
    empty_channel_integer_certificate, self_check, shortest_open_path,
)


def truth(count=10, seed=123, shift=0.):
    return {"problem": 4, "case_id": f"constructed-{seed}", "seed": seed,
            "sources": [{"channel": i+1, "x": shift+100.+20*i, "y": 0.,
                         "reception_radius_m": 1000.,
                         "orientation_deg": 0. if i == 0 else None} for i in range(count)]}


def record(label="baseline", seed=123, successful=True, elapsed=10000.):
    gt = truth(seed=seed)
    row = {"problem": 4, "case_id": gt["case_id"], "seed": seed, "strategy": label,
           "case_sha256": digest(gt), "successful": successful, "all_cleared": successful,
           "completion_certified": successful, "accepted_exit": True,
           "source_total": 10, "cleared_total": 10 if successful else 0,
           "virtual_time_s": elapsed, "penalized_time_s": elapsed if successful else 360000.}
    evaluation = {key: row[key] for key in
                  ("source_total", "cleared_total", "virtual_time_s", "all_cleared")}
    evaluation["ground_truth"] = gt
    return {"row": row, "evaluation": evaluation, "evaluation_phase": "after_policy_termination",
            "spec": {"entrypoint": "constructed:"+label, "kwargs": {}}}


def make_directory(tmp_path, records):
    manifest = {"protocol": {"problem": 4}, "seeds": sorted({r["row"]["seed"] for r in records}),
                "specs": {r["row"]["strategy"]: r["spec"] for r in records}}
    (tmp_path/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path/"summary.json").write_text(json.dumps({"rows": [r["row"] for r in records]}), encoding="utf-8")
    (tmp_path/"records").mkdir()
    for rec in records:
        row = rec["row"]
        with gzip.open(tmp_path/"records"/f"{row['strategy']}-{row['seed']}.json.gz", "wt", encoding="utf-8") as stream:
            json.dump(rec, stream)
    return tmp_path


def test_exact_dp_small_exhaustive_asymmetric_nonmetric():
    assert self_check() == 8
    # Shortest open path is intentionally not a return-to-origin tour.
    first, edges = [1., 100., 100.], [[0., 2., 100.], [99., 0., 3.], [70., 80., 0.]]
    assert shortest_open_path(first, edges) == (6., [0, 1, 2])


@pytest.mark.parametrize("first,edges", [([1], []), ([1], [[1, 2]]),
                                         ([math.nan], [[0]]), ([-1], [[0]]),
                                         ([1], [[math.inf]]), ([0]*17, [[0]*17]*17)])
def test_dp_rejects_invalid_graph(first, edges):
    with pytest.raises(ValueError):
        shortest_open_path(first, edges)


def test_empty_area_integer_relaxation_is_fifty_not_31_stations():
    certificate = empty_channel_integer_certificate()
    assert min(item["cost_s"] for item in certificate) == 50
    assert certificate[-1] == {"measurements": 10, "failed_clears": 0, "cost_s": 50}
    assert certificate[-2] == {"measurements": 9, "failed_clears": 600, "cost_s": 1845}
    for item in certificate:
        assert item["measurements"]*1000**2 + 3*item["failed_clears"]*20**2 >= 3*1800**2


def test_original_formula_and_raw_dataclass_tuple_supported():
    gt = truth()
    gt["sources"] = tuple(gt["sources"])
    bound = common_bound(gt)
    # Constructed source disks overlap consecutively: each relaxed edge is 0.
    # This intentionally demonstrates that DP graph LB need not be attainable.
    assert bound["lower_route_length_m"] == 80.
    assert bound["physical_oracle_lower_bound_s"] == 66.
    assert bound["empty_channel_action_lower_s"] == 500.
    assert bound["common_lower_bound_s"] == 566.
    assert bound["common_lower_bound_rounded_s"] == pytest.approx((566.-1e-6)/ROUNDING_FACTOR)
    assert sorted(bound["lower_graph_order_channels"]) == list(range(1, 11))


def test_public_16_cap_removes_empty_term_despite_four_empty_channels():
    bound = common_bound(truth(16))
    assert bound["empty_channel_count"] == 4
    assert bound["empty_channel_bound_applied"] is False
    assert bound["empty_channel_action_lower_s"] == 0
    assert bound["common_lower_bound_s"] == bound["physical_oracle_lower_bound_s"]
    below_cap = common_bound(truth(15))
    assert below_cap["empty_channel_action_lower_s"] == 250


def test_all_directional_q4_is_legal_even_without_omnidirectional_sources():
    gt = truth(14)
    mixed = common_bound(gt)
    for source in gt["sources"]:
        source["orientation_deg"] = float(source["channel"] * 17)
    directed = common_bound(gt)
    assert directed["source_total"] == 14
    assert directed["empty_channel_action_lower_s"] == 300
    assert directed["common_lower_bound_s"] == mixed["common_lower_bound_s"]
    assert directed["case_sha256"] != mixed["case_sha256"]


@pytest.mark.parametrize("mutation", ["q3", "count", "same_channel", "outside", "nan", "radius", "only_omni", "orientation"])
def test_invalid_truth_rejected(mutation):
    gt = truth()
    if mutation == "q3": gt["problem"] = 3
    if mutation == "count": gt["sources"].pop()
    if mutation == "same_channel": gt["sources"][1]["channel"] = 1
    if mutation == "outside": gt["sources"][0]["x"] = 1801.
    if mutation == "nan": gt["sources"][0]["y"] = math.nan
    if mutation == "radius": gt["sources"][0]["reception_radius_m"] = 999.
    if mutation == "only_omni": gt["sources"][0]["orientation_deg"] = None
    if mutation == "orientation": gt["sources"][0]["orientation_deg"] = 360.
    with pytest.raises(ValueError):
        common_bound(gt)


def test_source_graph_is_lower_than_any_selected_actual_clear_route():
    gt = truth()
    points = [(s["x"], s["y"]) for s in gt["sources"]]
    bound = common_bound(gt)
    # An explicitly feasible physical route, with arbitrary clearance offsets.
    order = [9, 1, 7, 5, 2, 3, 8, 0, 4, 6]
    clears = [(points[i][0]+18*math.cos(i), points[i][1]+18*math.sin(i)) for i in order]
    length = math.hypot(*clears[0])+sum(math.dist(p, q) for p, q in zip(clears, clears[1:]))
    assert bound["lower_route_length_m"] <= length


def test_same_case_computed_once_both_arms_share_denominator_and_external_cache(monkeypatch):
    import experiments.q4_comparison_bounds as module
    original, calls = module.common_bound, []
    def counted(gt):
        calls.append(gt["case_id"])
        return original(gt)
    monkeypatch.setattr(module, "common_bound", counted)
    cache = {}
    records = [record(), record("new", elapsed=9000.)]
    first = compare_records(records, cache=cache)
    second = compare_records(records, cache=cache)
    assert calls == ["constructed-123"]
    assert first["dp_computed_cases"] == 1 and first["dp_cache_hits"] == 1
    assert second["dp_computed_cases"] == 0 and second["dp_cache_hits"] == 2
    assert len({r["common_lower_bound_s"] for r in first["rows"]}) == 1


def test_short_failed_trajectory_keeps_penalty_but_has_no_completion_ratio():
    result = compare_records([record(), record("new", successful=False, elapsed=3.)])
    failed = next(r for r in result["rows"] if not r["successful"])
    assert failed["virtual_time_s"] == 3
    assert failed["time_over_common_lower_bound"] is None
    assert failed["penalized_time_s"] == 360000
    assert failed["penalized_time_over_common_lower_bound"] > 600
    assert result["summaries"]["new"]["all_complete"] is False


@pytest.mark.parametrize("mutation", ["duplicate", "missing_arm", "active", "truth_hash", "paired_truth", "fake_success", "penalty", "row_cost", "too_fast", "cache"])
def test_invalid_comparison_rejected(mutation):
    records, cache = [record(), record("new")], {}
    if mutation == "duplicate": records.append(copy.deepcopy(records[0]))
    if mutation == "missing_arm": records += [record(seed=124)]
    if mutation == "active": records[1]["evaluation_phase"] = "during_policy"
    if mutation == "truth_hash": records[1]["row"]["case_sha256"] = "wrong"
    if mutation == "paired_truth":
        gt = records[1]["evaluation"]["ground_truth"]
        gt["sources"][0]["x"] += 2
        records[1]["row"]["case_sha256"] = digest(gt)
    if mutation == "fake_success": records[1]["row"]["completion_certified"] = False
    if mutation == "penalty":
        records[1] = record("new", successful=False, elapsed=3.)
        records[1]["row"]["penalized_time_s"] = 3.
    if mutation == "row_cost": records[1]["row"]["virtual_time_s"] += 1
    if mutation == "too_fast": records[1] = record("new", elapsed=5.)
    if mutation == "cache":
        sha = records[0]["row"]["case_sha256"]
        cache[sha] = {"version": "stale", "case_sha256": sha}
    with pytest.raises(ValueError):
        compare_records(records, cache=cache)


def test_ratio_of_means_and_mean_ratio_are_distinct_statistics():
    records = [record(seed=123), record(seed=124, elapsed=8000.)]
    gt = records[1]["evaluation"]["ground_truth"]
    for source in gt["sources"]: source["x"] += 1000.
    records[1]["row"]["case_sha256"] = digest(gt)
    summary = compare_records(records)["summaries"]["baseline"]
    assert summary["mean_of_penalized_ratios"] != pytest.approx(summary["ratio_of_mean_penalized_time_to_mean_bound"])


def test_directory_reader_checks_manifest_row_spec_and_hashes(tmp_path):
    directory = make_directory(tmp_path, [record(), record("new")])
    result = analyze_directory(directory)
    assert result["case_count"] == 1 and result["record_count"] == 2
    assert len(result["input_sha256"]) == 4
    manifest = json.loads((directory/"manifest.json").read_text())
    manifest["specs"]["new"]["kwargs"] = {"changed": True}
    (directory/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="spec"):
        analyze_directory(directory)


def test_directory_missing_record_from_summary_is_rejected(tmp_path):
    directory = make_directory(tmp_path, [record(), record("new")])
    summary = json.loads((directory/"summary.json").read_text())
    summary["rows"].pop()
    (directory/"summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="manifest"):
        analyze_directory(directory)


def test_module_has_no_policy_generator_or_network_imports():
    import ast
    import experiments.q4_comparison_bounds as module
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any(name.startswith(("simulation", "strategies", "simulator_client", "socket", "urllib", "requests")) for name in imports)
