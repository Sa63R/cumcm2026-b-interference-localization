"""Pure artificial log structures; never open pilot data or create worlds."""

from copy import deepcopy
import json
import hashlib
from types import SimpleNamespace

import pytest

import planning.state_route as parent_helper
from experiments.round3_directed_incumbent_diagnosis import (
    aggregate, physical_costs, price, reconstruct_parent, verify_decision,
    verify_parent_files,
)


def fixture():
    tasks = [{"kind":"source", "channel":i+1, "exit_representative":[10.*i,0.], "service_s":5.} for i in range(2)]
    action = {"action":"clear", "channel":1, "position":[0.,0.],
              "phase":"near_clear", "result":"success", "virtual_time_s":5.}
    selected = {k:v for k,v in tasks[0].items() if k != "service_s"}
    log = {"status":"directed_selected", "after_actual_action_count":0,
        "current_position":[0.,0.], "tasks":tasks, "travel_times_s":[[0.,2.],[2.,0.]],
        "initial_times_s":[0.,2.], "scan_source_s":0., "selected":selected,
        "parent_selected":selected, "parent_planning":{"cost_s":12.},
        "expansions_before_parent":0, "parent_expansions":0, "expansions_after_parent":0,
        "directed_expansions":0, "expansions_after_decision":0,
        "transition_audit":{"complete":True,"arcs":[]}, "uncertain_target_indices":[],
        "route_budget":100, "route":{"order":[0,1],"cost_s":12.,"lower_bound_s":12.,"expanded":0,"exact":True},
        "incumbent_supplied":True, "incumbent_order":[0,1], "parent_order_cost_new_model_s":12.,
        "candidate_minus_parent_order_same_model_s":0., "incumbent_result_order":[0,1],
        "incumbent_result_cost_s":12., "execution_status":"first_physical_action_recorded",
        "first_actual_action_ordinal":1, "first_actual_action":deepcopy(action),
        "selected_task_matches_actual":True, "changed_first_task":False}
    return log, [action]


def test_complete_supplied_incumbent_has_independent_cost_and_action_binding():
    log, actions = fixture()
    original = deepcopy(log)
    result = verify_decision(log, actions, require_incumbent=True)
    assert result["complete"] and result["executed"]
    assert result["cost_s"] == result["supplied_parent_cost_s"] == 12.
    assert log == original
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("field,value", [
    ("incumbent_order",[0,0]), ("incumbent_supplied",False),
    ("parent_order_cost_new_model_s",11.), ("candidate_minus_parent_order_same_model_s",2.),
    ("incumbent_result_order",[1,0]), ("first_actual_action_ordinal",2),
    ("selected_task_matches_actual",False), ("expansions_after_decision",7),
])
def test_mutated_claims_are_rejected(field, value):
    log, actions = fixture()
    log[field] = value
    with pytest.raises(ValueError):
        verify_decision(log, actions, require_incumbent=True)


def test_actual_route_more_expensive_than_supplied_cannot_claim_success():
    log, actions = fixture()
    log["route"].update(order=[1,0], cost_s=14.)
    log["selected"] = {k:v for k,v in log["tasks"][1].items() if k != "service_s"}
    log.update(incumbent_result_order=[1,0], incumbent_result_cost_s=14., candidate_minus_parent_order_same_model_s=2.)
    with pytest.raises(ValueError, match="incumbent contract violated"):
        verify_decision(log, actions, require_incumbent=True)


def test_failed_or_interrupted_execution_is_not_counted_as_executed_model():
    log, _ = fixture()
    log["execution_status"] = "interrupted_before_physical_action"
    log.pop("first_actual_action")
    log.pop("first_actual_action_ordinal")
    result = verify_decision(log, [], require_incumbent=True)
    assert result["complete"] and not result["executed"]
    result["reconstructed_parent_cost_s"] = 12.
    totals = aggregate([result])
    assert totals["supplied_parent_routes"] == 1 and totals["executed_models"] == 0


def test_fallback_never_masquerades_as_a_supplied_complete_model():
    log, actions = fixture()
    log.update(status="parent_fallback", fallback_reason="no_uncertain_source")
    result = verify_decision(log, actions, require_incumbent=True)
    assert not result["complete"]
    assert aggregate([result])["complete_models"] == 0


def test_cost_includes_services_scans_and_directed_orientation():
    log, _ = fixture()
    log["tasks"][0].update(kind="cover", channel=None, service_s=12.)
    log["scan_source_s"] = 6.
    log["travel_times_s"] = [[0.,7.],[2.,0.]]
    assert price(log,[0,1]) == 0+12+6+7+5
    assert price(log,[1,0]) == 2+5+2+12


def test_old_log_without_supply_can_still_measure_incumbent_omission():
    log, actions = fixture()
    for key in tuple(log):
        if key.startswith("incumbent_") or key in ("parent_order_cost_new_model_s","candidate_minus_parent_order_same_model_s"):
            log.pop(key)
    result = verify_decision(log,actions,require_incumbent=False)
    result["reconstructed_parent_cost_s"] = 11.
    totals = aggregate([result])
    assert totals["supplied_parent_routes"] == 0
    assert totals["worse_than_reconstructed_parent"] == 1


def test_first_probe_must_match_real_point_and_never_only_the_exit_representative():
    log, actions = fixture()
    log["uncertain_target_indices"] = [0]
    log["transition_audit"]["arcs"] = [{"target_index":0,"origin":[0.,0.],"first_probe":[0.,1.]}]
    with pytest.raises(ValueError,match="predicted first probe"):
        verify_decision(log,actions,require_incumbent=True)
    actions[0].update(action="measure",phase="active_localization",position=[0.,1.],result="direction")
    log["first_actual_action"] = deepcopy(actions[0])
    assert verify_decision(log,actions,require_incumbent=True)["executed"]


def test_physical_accounting_clear_does_not_tune_channel():
    actions = [
        {"action":"measure","channel":2,"position":[0.,0.],"result":"direction","virtual_time_s":6.},
        {"action":"clear","channel":3,"position":[0.,0.],"result":"success","virtual_time_s":11.},
        {"action":"measure","channel":2,"position":[0.,0.],"result":"direction","virtual_time_s":16.},
    ]
    costs = physical_costs(actions)
    assert costs == {"movement_s":0.,"switching_s":1.,"detection_s":10.,"clear_s":5.}
    actions[-1]["virtual_time_s"] = 17.
    with pytest.raises(ValueError,match="virtual accounting"):
        physical_costs(actions)


def test_original_parent_route_reconstruction_uses_only_frozen_public_tasks():
    log, _ = fixture()
    relocation = {"after_actual_action_count":0, "remaining_before":[],
                  "baseline_proxy_s":12., "evaluated":[]}
    result = reconstruct_parent(log,relocation,parent_helper,
                                {"max_expansions":100,"max_total_expansions":60000})
    assert result == (0,1)
    log["parent_planning"]["cost_s"] = 11.
    with pytest.raises(ValueError,match="selected parent cost"):
        reconstruct_parent(log,relocation,parent_helper,{"max_expansions":100,"max_total_expansions":60000})


def test_original_shared_parent_may_omit_unused_candidate_modules_but_no_source_can_change(tmp_path):
    required = ["src/planning/state_route.py", "src/simulator_client/state.py", "src/simulator_client/rules.py"]
    hashes = {}
    for name in required:
        path = tmp_path/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text("# artificial dependency fixture\n")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes["src/planning/directed_state_route.py"] = "unused archived module"
    identity = {"source_sha256":hashes}
    assert set(verify_parent_files(tmp_path,identity)) == set(required)
    (tmp_path/required[0]).write_text("# changed\n")
    with pytest.raises(ValueError,match="parent source mismatch"):
        verify_parent_files(tmp_path,identity)
