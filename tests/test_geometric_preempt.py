from collections import Counter
from types import SimpleNamespace

import pytest

from localization.omni import OmniCandidateRegion
from simulation import LocalResearchSimulator, random_scenario
from simulator_client.state import ClientState, Position
from strategies import geometric_preempt as preempt
from strategies.geometric_probe_cost import run_probe_cost_search
from tests.test_strategy import ObservationOnlyClient


def policy_fixture(max_probes=3):
    client = SimpleNamespace(state=ClientState(session='active', position=Position(0.,0.), current_channel=1))
    policy = preempt.GeometricPreemptSearch(client, 20000, max_probes, None, 120., True)
    policy.regions = {c:OmniCandidateRegion().observe((0.,0.), 0.) for c in (1,2)}
    policy.first_bearings = {1:0., 2:0.}
    policy.detected = {1,2}
    policy._remaining = [Position(-100.,0.)]
    policy._target = lambda c: Position(500.,0.) if c==1 else Position(100.,0.)
    return policy


def test_route_comparison_preserves_entire_task_multiset_and_locked_suffix_is_optimized():
    primary = ('source',1,Position(500.,0.))
    tasks = [primary,('source',2,Position(100.,0.)),('cover',None,Position(-100.,0.))]
    task, decision = preempt.frozen_route_interruption(Position(0.,0.),primary,tasks)
    assert task == tasks[2]
    assert decision['forced_length_m'] == 1100.
    assert decision['interrupted_length_m'] == 700.
    assert decision['score_s'] == 78.
    assert Counter(map(tuple,decision['forced_route'])) == Counter(map(tuple,decision['interrupted_route']))
    # The supplied tail is deliberately reversed. Optimizing it while primary
    # stays first is not evidence that primary should be interrupted.
    primary = ('source',1,Position(100.,0.))
    tasks = [primary,('source',2,Position(500.,0.)),('cover',None,Position(300.,0.))]
    assert preempt.frozen_route_interruption(Position(0.,0.),primary,tasks) is None


def test_source_lifetime_allows_only_one_interruption():
    policy = policy_fixture()
    policy._primary_probe_counts[1] = 1
    policy.actions = 1
    with pytest.raises(preempt._InterruptResolution):
        policy._interruption_after_probe(1)
    policy._primary_probe_counts[1] = 2
    policy.actions = 2
    assert policy._interruption_after_probe(1) is None
    assert len(policy.report.source_interruptions['events']) == 1


def test_probe_budget_persists_across_resume_and_exhaustion_forces_original_fallback(monkeypatch):
    policy = policy_fixture(max_probes=3)
    indices=[]; actions=[]
    def next_probe(channel,index):
        indices.append(index)
        return Position(0.,0.)
    def perform(action,position,channel,phase):
        actions.append((action,channel,phase));policy.actions+=1
    def clear(position,channel,phase):
        actions.append(('clear',channel,phase));policy.actions+=1;policy.cleared.add(channel)
        return True
    policy._next_probe=next_probe;policy._perform=perform;policy._clear=clear
    monkeypatch.setattr(preempt,'clearance_grid',lambda *args,**kwargs:(Position(500.,0.),))
    with pytest.raises(preempt._InterruptResolution):
        policy._resolve(1)
    assert indices == [0]
    assert policy._resolve(1)
    assert indices == [0,1,2]
    assert policy.report.source_interruptions['probe_counts']['1'] == 3
    assert len(policy.report.source_interruptions['events']) == 1
    assert actions[-1] == ('clear',1,'guaranteed_clearance')


def test_new_scheduling_check_occurs_only_after_an_accepted_primary_action(monkeypatch):
    policy = policy_fixture(max_probes=2)
    accepted=[]; checks=[]
    policy._next_probe=lambda channel,index:Position(0.,0.)
    def perform(action,position,channel,phase):
        accepted.append((action,phase));policy.actions+=1
    def check(channel):
        assert accepted and accepted[-1]==('measure','active_localization')
        assert policy._primary_probe_counts[channel]==len(accepted)
        checks.append(len(accepted))
    policy._perform=perform;policy._interruption_after_probe=check
    policy._clear=lambda *args:True
    monkeypatch.setattr(preempt,'clearance_grid',lambda *args,**kwargs:(Position(500.,0.),))
    assert policy._resolve(1)
    assert checks == [1,2]


def test_certified_and_near_sources_are_cleared_without_a_new_scheduler_check(monkeypatch):
    policy = policy_fixture()
    policy.near_points[1]=Position(0.,0.)
    policy._clear=lambda *args:True
    policy._interruption_after_probe=lambda channel:pytest.fail('Ready clear was interrupted')
    assert policy._resolve(1)
    policy.near_points.clear()
    policy.regions[1].vertices=((-1.,-1.),(1.,-1.),(1.,1.),(-1.,1.))
    policy.regions[1]._circle=None
    assert policy._resolve(1)


def test_disabled_is_exactly_the_existing_single_strategy():
    case=random_scenario(3,107001)
    a=run_probe_cost_search(ObservationOnlyClient(LocalResearchSimulator(case).client()))
    b=preempt.run_preempt_search(ObservationOnlyClient(LocalResearchSimulator(case).client()),preempt=False)
    assert a.action_history==b.action_history
    assert a.virtual_time_s==b.virtual_time_s


def test_full_observation_only_run_retains_count_budget_and_physical_ledger():
    sim=LocalResearchSimulator(random_scenario(3,115001))
    report=preempt.run_preempt_search(ObservationOnlyClient(sim.client()))
    result=sim.evaluation()
    assert result['all_cleared'] and result['failed_clear_count']==0
    assert report.completion_certified_under_model
    assert report.virtual_time_s==result['virtual_time_s']
    assert sum(report.time_breakdown.values())==pytest.approx(report.virtual_time_s,abs=1e-4)
    counts=Counter(a['channel'] for a in report.action_history if a['phase']=='active_localization')
    assert max(counts.values())<=6
    assert {str(c):v for c,v in counts.items()}==report.source_interruptions['probe_counts']
    events=report.source_interruptions['events']
    assert len(set(e['channel'] for e in events))==len(events)
    assert all(e['primary_probes']>=1 and e['score_s']>=10 for e in events)
    assert all(a['accepted_actions']<b['accepted_actions'] for a,b in zip(events,events[1:]))


@pytest.mark.parametrize('kwargs',[dict(problem=4),dict(preempt=1),dict(max_active_probes=True)])
def test_invalid_options_fail_before_enter(kwargs):
    sim=LocalResearchSimulator(random_scenario(3,107001))
    with pytest.raises(ValueError):
        preempt.run_preempt_search(ObservationOnlyClient(sim.client()),**kwargs)
    assert sim.observation_history()==[]
