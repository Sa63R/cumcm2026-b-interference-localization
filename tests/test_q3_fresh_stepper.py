from dataclasses import replace

import pytest

from simulation.cases import random_scenario, difficult_scenarios
from simulation.engine import LocalResearchSimulator
from strategies.q3_fresh import run_fresh
from strategies.q3_fresh_stepper import StepperQ3, Action


@pytest.mark.parametrize("case", [random_scenario(3,910100),random_scenario(3,910101),
                                    *difficult_scenarios(3)])
def test_only_baseline_candidate_reproduces_every_action(case):
    original = LocalResearchSimulator(case)
    expected = run_fresh(original.client(), "v1", selective=True)
    resumed = LocalResearchSimulator(case)
    actual = StepperQ3(resumed.client()).run()
    assert actual["action_history"] == expected["action_history"]
    assert resumed.evaluation()["virtual_time_s"] == original.evaluation()["virtual_time_s"]
    assert actual["completion_certified"]


def test_clone_has_independent_controller_and_public_client():
    from simulation.q3_branch import make_q3_branch
    case = random_scenario(3,910102)
    engine=LocalResearchSimulator(case)
    client=engine.client()
    client.enter()
    controller=StepperQ3(client)
    for _ in range(25):
        controller.next_action()
        controller.execute_pending()
    proposed=controller.next_action()
    branch=make_q3_branch(case,client.state,controller.history)
    fork=controller.clone(branch)
    assert fork.pending==proposed
    fork.execute_pending()
    assert len(fork.history)==len(controller.history)+1
    result=fork.run()
    assert result["completion_certified"]
    assert client.state.session=="active"
    assert controller.pending==proposed


def test_observable_tail_returns_to_discovery():
    case=random_scenario(3,910105)
    engine=LocalResearchSimulator(case)
    result=StepperQ3(engine.client(),movable_tail=True).run()
    assert engine.evaluation()["all_cleared"] and result["completion_certified"]


def test_failed_attempt_keeps_channel_and_records_exclusion():
    case=random_scenario(3,910102)
    engine=LocalResearchSimulator(case)
    client=engine.client()
    client.enter()
    p=StepperQ3(client)
    source=case.sources[0]
    q=(source.x+100,source.y)
    p.override((Action("measure",q,source.channel,"test"),))
    p.execute_pending()
    current=client.state.current_channel
    p.override((Action("clear",q,source.channel,"try",True),))
    p.execute_pending()
    assert p.failed_disks[source.channel]==[q]
    assert p.channels[source.channel].status=="detected"
    assert client.state.current_channel==current
