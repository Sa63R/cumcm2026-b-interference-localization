import time

from simulation.cases import random_scenario
from simulation.engine import LocalResearchSimulator
from strategies.q3_fresh_stepper import StepperQ3
from strategies.q3_fresh_rollout import FreshRollout


def prefix():
    engine=LocalResearchSimulator(random_scenario(3,910100))
    client=engine.client()
    client.enter()
    policy=StepperQ3(client)
    for _ in range(22):
        policy.next_action()
        policy.execute_pending()
    policy.next_action()
    return policy


def test_budget_exhaustion_preserves_pending_action_and_certificate():
    policy=prefix()
    action=policy.pending
    planner=FreshRollout(budget_s=0)
    planner.maybe_choose(policy)
    assert policy.pending==action
    assert planner.stats["budget_fallbacks"]==1
    assert policy.run()["completion_certified"]


def test_timeout_is_not_a_low_cost_sample(monkeypatch):
    policy=prefix()
    action=policy.pending
    planner=FreshRollout()
    def expired(*args,**kwargs):
        raise TimeoutError("partial sample")
    monkeypatch.setattr("strategies.q3_fresh_rollout.sample_worlds",expired)
    planner.maybe_choose(policy)
    assert policy.pending==action
    assert planner.stats["overrides"]==0
    assert planner.stats["decisions"][-1]["completed"] is False


def test_common_worlds_and_full_paired_survivor_comparison(monkeypatch):
    policy=prefix()
    planner=FreshRollout()
    worlds=[object() for _ in range(12)]
    monkeypatch.setattr("strategies.q3_fresh_rollout.sample_worlds",lambda *a,**kw:worlds)
    options=planner.candidates(policy)
    baseline=options[0][1]
    seen=[]
    def cost(p,proposal,world,deadline):
        seen.append((proposal,world))
        return 100.0 if proposal==baseline else 70.0
    monkeypatch.setattr(planner,"_evaluate",cost)
    planner.maybe_choose(policy)
    assert planner.stats["overrides"]==1
    assert len(planner.stats["decisions"][0]["paired_delta_s"])==12
    assert all(sum(w is world for _,w in seen)>=2 for world in worlds)
    assert sum(proposal==baseline for proposal,_ in seen)==12


def test_failed_continuation_discards_entire_decision(monkeypatch):
    policy=prefix()
    planner=FreshRollout()
    original=policy.pending
    monkeypatch.setattr("strategies.q3_fresh_rollout.sample_worlds",lambda *a,**kw:[object()]*12)
    monkeypatch.setattr(planner,"_evaluate",lambda *args: (_ for _ in ()).throw(RuntimeError("incomplete")))
    planner.maybe_choose(policy)
    assert policy.pending==original
    assert planner.stats["continuation_failures"]==1


def test_real_remaining_time_reserves_execution_budget():
    policy=prefix()
    planner=FreshRollout()
    policy.client.state.real_deadline=time.monotonic()+20
    planner.maybe_choose(policy)
    assert planner.stats["planning_calls"]==0
    assert planner.stats["budget_fallbacks"]==1


def test_hypothetical_virtual_timeout_falls_back(monkeypatch):
    from simulator_client.errors import DeadlineExceeded
    policy=prefix()
    planner=FreshRollout()
    original=policy.pending
    monkeypatch.setattr("strategies.q3_fresh_rollout.sample_worlds",lambda *a,**kw:[object()]*12)
    monkeypatch.setattr(planner,"_evaluate",lambda *a: (_ for _ in ()).throw(DeadlineExceeded("branch virtual budget")))
    planner.maybe_choose(policy)
    assert policy.pending==original
    assert planner.stats["continuation_failures"]==1


def test_only_baseline_candidate_is_action_identical(monkeypatch):
    from strategies.q3_fresh import run_fresh
    case=random_scenario(3,910100)
    original=LocalResearchSimulator(case)
    reference=run_fresh(original.client(),"v1",selective=True)
    other=LocalResearchSimulator(case)
    planner=FreshRollout()
    monkeypatch.setattr(planner,"candidates",lambda p:[("baseline",(p.pending,))])
    actual=StepperQ3(other.client()).run(planner=planner)
    assert actual["action_history"]==reference["action_history"]
    assert planner.stats["planning_calls"]==0
