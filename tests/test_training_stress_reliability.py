"""Frozen training stress generation and independent physical ledger checks."""

from copy import deepcopy
import math

from experiments.training_stress_reliability import audit_actions, digest, generate_cases, physical_signature
from simulation import Scenario, Source, difficult_scenarios


def test_training_cases_are_unique_reproducible_valid_and_not_exposed_hard_duplicates():
    cases = generate_cases()
    assert cases == generate_cases()
    assert len(cases) == len({c["case_sha256"] for c in cases}) == 28
    assert [c["scenario"]["seed"] for c in cases] == list(range(113001, 113029))
    old = {physical_signature(c.evaluation_config()) for c in difficult_scenarios(3)}
    for item in cases:
        case = item["scenario"]
        assert item["case_sha256"] == digest(case)
        assert physical_signature(case) not in old
        assert case["case_id"].startswith("q3-training-stress-")
        Scenario(**{**case, "sources": tuple(Source(**s) for s in case["sources"])})
        assert len(case["sources"]) == (10, 12, 14, 16)[item["replicate"]-1]
        assert all(s["reception_radius_m"] == 1000 for s in case["sources"])
        if item["family"] == "boundary":
            assert all(abs(math.hypot(s["x"], s["y"])-1800) < 1e-9 for s in case["sources"])
        if item["family"] == "cluster":
            assert max(math.dist((a["x"], a["y"]), (b["x"], b["y"]))
                       for a in case["sources"] for b in case["sources"]) <= .5+1e-9


def ledger_fixture():
    events = [("/enter", 0, None, {}, 0.),
              ("/measure", 50, 2, {"measure_result": "direction", "svd_deg": 0.}, 16.),
              ("/clear", 50, 2, {"clear_result": "success"}, 21.),
              ("/exit", 50, None, {}, 21.)]
    history = [{"index": i, "action": action, "position": {"x": x, "y": 0.}, "channel": channel,
                "response": {"accepted": True, "virtual_time_s": virtual, **response}}
               for i, (action, x, channel, response, virtual) in enumerate(events)]
    evaluation = {"time_breakdown_s": {"movement_s": 10., "switching_s": 1., "detection_s": 5.,
                                       "optical_s": 3., "removal_s": 2.},
                  "measurement_count": 1, "cleared_total": 1, "failed_clear_count": 0}
    return history, evaluation


def test_independent_cost_ledger_and_nonretuning_clear():
    history, evaluation = ledger_fixture()
    result = audit_actions(history, evaluation)
    assert result["errors"] == [] and result["accepted_exit"]
    assert result["reconstructed_time_s"] == 21.
    # /clear on another channel does not itself add a switching charge.
    history[2]["channel"] = 3
    assert audit_actions(history, evaluation)["errors"] == []


def test_ledger_rejects_wrong_cost_channel_and_measurement_count():
    history, evaluation = ledger_fixture()
    wrong = deepcopy(history)
    wrong[1]["response"]["virtual_time_s"] = 15.
    assert "Per-action cost mismatch 1" in audit_actions(wrong, evaluation)["errors"]
    wrong = deepcopy(history)
    wrong[1]["channel"] = 21
    assert "Illegal channel 1" in audit_actions(wrong, evaluation)["errors"]
    evaluation["measurement_count"] = 2
    assert "Evaluator count mismatch: measurements" in audit_actions(history, evaluation)["errors"]
