"""Construction/guard tests only: no reserved 616xxx case is generated."""
import copy
from dataclasses import replace
import json
import math
import sys
from types import ModuleType

import pytest

from experiments import run_q4_all_directional as m
from simulation.cases import Scenario, Source


@pytest.mark.parametrize("count", [10, 16])
@pytest.mark.parametrize("family", [None, *m.FAMILIES])
def test_constructed_cases_are_all_directional_and_publicly_legal(count, family):
    case = m.build_case(41, count=count, family=family)
    assert isinstance(case, Scenario) and case.problem == 4
    assert len(case.sources) == count and len({s.channel for s in case.sources}) == count
    assert all(s.orientation_deg is not None and 0 <= s.orientation_deg < 360 for s in case.sources)
    assert all(math.hypot(s.x, s.y) <= 1800 and 1000 <= s.reception_radius_m <= 1500 for s in case.sources)
    assert case == m.build_case(41, count=count, family=family)
    if family is not None:
        assert all(s.reception_radius_m == 1000 for s in case.sources)
    if family == "boundary_outward":
        assert all(abs((math.degrees(math.atan2(s.y, s.x))-s.orientation_deg+180) % 360-180) < 1e-8 for s in case.sources)


def test_subclass_does_not_patch_legacy_scenario_or_relax_source_rules():
    case = m.build_case(19, count=10)
    with pytest.raises(ValueError, match="both source types"):
        Scenario(case.case_id, 4, case.seed, case.sources)
    with pytest.raises(ValueError, match="directional"):
        replace(case, sources=(replace(case.sources[0], orientation_deg=None),)+case.sources[1:])
    with pytest.raises(ValueError, match="Exclusive"):
        replace(case, sources=(case.sources[0],)*10)
    with pytest.raises(ValueError):
        Source(1, 1801, 0, 1000, 0)
    with pytest.raises(ValueError):
        replace(case, error_mode="unknown")


def test_freeze_never_generates_cases_and_rejects_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "build_case", lambda *a, **kw: pytest.fail("Freeze generated a scenario"))
    p = tmp_path/"plan.json"
    plan = m.freeze_plan(p)
    assert plan["cases_generated"] is False and set(plan["specs"]) == {m.BASE, m.COMBO}
    assert len(plan["protocol"]["random"]) == 16 and len(plan["protocol"]["stress"]) == 14
    assert plan["protocol"]["stress_assignment"] == [{"family": f, "sources": n} for f in m.FAMILIES for n in (10, 16)]
    assert "experiments/run_q4_all_directional.py" in plan["source_sha256"]
    m.validate_plan(json.loads(p.read_text(encoding="utf-8")))
    with pytest.raises(ValueError, match="Preserve"):
        m.freeze_plan(p)


@pytest.mark.parametrize("field", ["protocol", "source_sha256", "specs", "cases_generated"])
def test_tampered_freeze_blocks_before_any_reserved_generation(tmp_path, monkeypatch, field):
    plan = m.freeze_plan(tmp_path/"plan.json")
    plan[field] = True if field == "cases_generated" else {}
    # Even a recomputed container hash cannot bypass source/protocol/spec checks.
    plan["payload_sha256"] = m.digest({k: v for k, v in plan.items() if k != "payload_sha256"})
    monkeypatch.setattr(m, "make_case", lambda *a, **kw: pytest.fail("Invalid freeze generated a scenario"))
    with pytest.raises(ValueError):
        m.run_stage(tmp_path/"never-created", "random", plan, workers=1)
    assert not (tmp_path/"never-created").exists()


def test_extra_candidate_needs_exact_passing_external_qualification():
    specs = {m.BASE: m.BASE_SPEC, m.COMBO: m.COMBO_SPEC,
             "compact_extra": {"entrypoint": "strategies.example:run", "kwargs": {}}}
    with pytest.raises(ValueError):
        m.validate_specs(specs)
    qualification = {"passed": True, "selected": "compact_extra", "spec": specs["compact_extra"]}
    m.validate_specs(specs, qualification)
    for key, value in [("passed", False), ("selected", "another"), ("spec", {})]:
        with pytest.raises(ValueError):
            m.validate_specs(specs, {**qualification, key: value})
    with pytest.raises(ValueError):
        m.validate_specs({**specs, "compact_fourth": specs["compact_extra"]}, qualification)


@pytest.mark.parametrize("field", ["specs", "source_sha256", "selection_sha256"])
def test_manifest_must_bind_to_plan_before_common_audit(tmp_path, monkeypatch, field):
    plan = m.freeze_plan(tmp_path/"plan.json")
    directory = tmp_path/"audit-only"
    directory.mkdir()
    manifest = {"stage": "random", "seeds": m.PROTOCOL["random"],
        "specs": plan["specs"], "source_sha256": plan["source_sha256"],
        "selection_sha256": plan["payload_sha256"], "all_directional_plan": plan}
    manifest[field] = "0"*64 if field == "selection_sha256" else {}
    m.write_json(directory/"manifest.json", manifest)
    monkeypatch.setattr(m, "common_audit", lambda *a: pytest.fail("Unbound manifest reached common audit"))
    with pytest.raises(ValueError, match="Manifest differs"):
        m.audit_stage(directory)
    assert not (directory/"independent_audit.json").exists()
    assert not (directory/"all_directional_audit.json").exists()


def test_failure_keeps_penalty_and_truth_is_read_only_after_policy_termination(monkeypatch):
    # A deliberately aborted enter-only unit session, not a performance case.
    finished = []
    module = ModuleType("strategies._all_directional_unit")
    def aborted(client, **kwargs):
        for name in ("scenario", "evaluation", "ground_truth"):
            with pytest.raises(AssertionError):
                getattr(client, name)
        client.enter()
        finished.append(True)
        raise RuntimeError("Deliberate unit-test policy failure")
    module.run = aborted
    monkeypatch.setitem(sys.modules, module.__name__, module)
    original = m.LocalResearchSimulator
    class CheckedSimulator(original):
        def evaluation(self):
            assert finished and self._session == "exited"
            return super().evaluation()
    monkeypatch.setattr(m, "LocalResearchSimulator", CheckedSimulator)
    record = m.run_one_case(m.build_case(23, count=10), "compact_unit",
        {"entrypoint": module.__name__+":run", "kwargs": {}}, m.hashes())
    assert record["evaluation_phase"] == "after_policy_termination"
    assert not record["row"]["successful"] and record["row"]["penalized_time_s"] == 360000
    assert record["row"]["common_lower_bound_s"] > 0 and record["row"]["accepted_exit"]
    assert record["history"][-1]["action"] == "/exit"
    assert m.audit_directional_record(record)["passed"]
    bad = copy.deepcopy(record)
    bad["evaluation"]["ground_truth"]["sources"][0]["orientation_deg"] = None
    bad["row"]["case_sha256"] = m.digest(bad["evaluation"]["ground_truth"])
    with pytest.raises(ValueError, match="directional"):
        m.audit_directional_record(bad)
    bad = copy.deepcopy(record)
    bad["evaluation_phase"] = "during_policy"
    with pytest.raises(ValueError, match="termination"):
        m.audit_directional_record(bad)


def test_nonreserved_case_request_is_rejected_without_construction(monkeypatch):
    monkeypatch.setattr(m, "build_case", lambda *a, **kw: pytest.fail("Invalid case generated"))
    with pytest.raises(ValueError):
        m.make_case(1, "random")
    with pytest.raises(ValueError):
        m.make_case(1, "other")
