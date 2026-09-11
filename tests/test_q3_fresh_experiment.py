import ast
from pathlib import Path

import pytest

from experiments.export_q3_fresh_validation import compatible
from experiments.run_q3_fresh import StressEngine, lower_bound, summarize
from simulation.cases import random_scenario


def event(kind, position, bearing=None):
    import json
    response = {"measure_result": kind}
    if bearing is not None:
        response["svd_deg"] = bearing
    return dict(x_m=position[0], y_m=position[1], response_json=json.dumps(response))


def test_reconstruction_obeys_positive_and_negative_radius_constraints():
    observations = [event("direction", (0, 0), 0), event("no_signal", (2300, 0))]
    assert compatible((1100, 0), observations) == pytest.approx((1100, 1200 - 1e-7))
    assert compatible((1200, 0), observations) is None
    assert compatible((1100, 50), observations) is None


def test_reconstruction_preserves_fixed_recorded_bearing():
    case = random_scenario(3, 941003)
    source = case.sources[0]
    p = (source.x + 100, source.y)
    anchors = [dict(channel=source.channel, position=p,
                    response={"measure_result": "direction", "svd_deg": 180.5})]
    engine = StressEngine(case, anchors)
    client = engine.client()
    client.enter()
    assert client.measure(p, source.channel)["svd_deg"] == 180.5
    assert client.measure(p, source.channel)["svd_deg"] == 180.5
    client.clear((source.x, source.y), source.channel)
    assert client.measure(p, source.channel)["measure_result"] == "no_signal"
    client.exit()


def test_fresh_policy_has_no_truth_or_dataset_dependency():
    path = Path(__file__).resolve().parents[1] / "src/strategies/q3_fresh.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden_modules = {"simulation", "experiments", "sqlite3", "practice_control", "q3_belief"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not any(p in node.module.split(".") for p in forbidden_modules)
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"_sources", "_exchange_fn", "scenario", "evaluation", "ground_truth"}


def test_label_uncertainty_cannot_strengthen_physical_bound():
    case = random_scenario(3, 941003)
    exact = lower_bound(case)
    uncertain = lower_bound(case, {str(s.channel): 5.0 for s in case.sources})
    assert uncertain["physical_lower_bound_s"] <= exact["physical_lower_bound_s"]
