from collections import Counter
from types import SimpleNamespace

import pytest

from localization.omni import OmniCandidateRegion
from simulator_client.state import ClientState,Position
from simulation import LocalResearchSimulator,random_scenario
from strategies import geometric_swap as swap
from strategies.geometric_probe_cost import run_probe_cost_search
from strategies.search import _StopSearch
from tests.test_strategy import ObservationOnlyClient


def fixture():
    client=SimpleNamespace(state=ClientState(session='active',position=Position(0.,0.),current_channel=1))
    p=swap.GeometricSwapSearch(client,20000,3,None,120.,True)
    p.detected={1,2}
    p.regions={c:OmniCandidateRegion().observe((0.,0.),0.) for c in (1,2)}
    p.first_bearings={1:0.,2:0.}
    p._target=lambda c:Position(500.,0.) if c==1 else Position(100.,0.)
    p._remaining=[]
    return p


def test_one_uniform_forced_route_and_exact_cover_measure_surcharge():
    primary=('source',1,Position(500.,0.))
    other=('source',2,Position(100.,0.))
    selected,record=swap.frozen_two_task_swap(Position(0.,0.),primary,[primary,other])
    assert selected==other and record['score_s']==78.
    assert record['forced_length_m']==900. and record['swapped_length_m']==500.
    cover=('cover',None,other[2])
    _,with_cover=swap.frozen_two_task_swap(Position(0.,0.),primary,[primary,cover])
    assert with_cover['score_s']==73. and with_cover['extra_primary_cover_measure_s']==5.
    assert Counter(map(tuple,record['forced_route']))==Counter(map(tuple,record['swapped_route']))


def test_tail_order_is_never_reoptimized_or_changed_and_all_candidates_use_same_baseline():
    p=Position(0.,0.);primary=('source',1,Position(800.,0.))
    tasks=[primary,('source',2,Position(100.,0.)),('cover',None,Position(800.,100.)),
           ('source',3,Position(900.,0.))]
    selected,record=swap.frozen_two_task_swap(p,primary,tasks)
    forced=record['forced_route'];changed=record['swapped_route']
    assert changed[0][:2]==list(selected[:2]) and changed[1][:2]==list(primary[:2])
    assert changed[2:]==[t for t in forced if t[:2] not in (list(selected[:2]),list(primary[:2]))]
    points=lambda r:[Position(t[2],t[3]) for t in r]
    baseline=swap.route_length(points(forced),p)
    scores=[]
    for other in forced[1:]:
        route=[other,forced[0]]+[t for t in forced if t is not other and t is not forced[0]]
        scores.append((baseline-swap.route_length(points(route),p))/5.-2.-(5. if other[0]=='cover' else 0.))
    assert record['score_s']==max(scores)


def test_interruption_only_once_per_primary_and_no_nested_interruption():
    p=fixture();p._primary_probe_counts[1]=1;p.actions=1
    p._locked_execution=True
    assert p._interruption_after_probe(1) is None
    p._locked_execution=False
    with pytest.raises(swap._LockedExchange) as found:
        p._interruption_after_probe(1)
    assert found.value.primary_channel==1
    assert p._interruption_after_probe(1) is None
    assert len(p.report.source_interruptions['events'])==1


@pytest.mark.parametrize('kind',['source','cover'])
def test_execute_one_atomic_other_then_resume_primary_without_global_scheduler(kind):
    p=fixture();calls=[];point=Position(100.,0.)
    p._remaining=[point]
    p._primary_probe_counts[1]=1
    def scan(q):
        assert p._locked_execution
        # Stand-in for the complete inherited scan, not a yield per channel.
        for channel in range(1,21):
            calls.append(('measure',channel));p.actions+=1
        p.regions[1]='fresh-cover-feedback'
    def resolve(c):
        assert p._locked_execution
        if c==1 and kind=='cover':
            assert p.regions[1]=='fresh-cover-feedback'
        calls.append(('resolve',c));p.actions+=1;p.cleared.add(c)
        p._primary_probe_counts[c]+=1
        assert p._interruption_after_probe(c) is None
        return True
    p._scan=scan;p._resolve=resolve
    p._next_task=lambda _:pytest.fail('global scheduler ran inside committed exchange')
    event={}
    p._execute_exchange(swap._LockedExchange((kind,2 if kind=='source' else None,point),1,event))
    assert calls[-1]==('resolve',1)
    assert event['other_end_actions']==event['resume_start_actions']
    assert event['resume_primary_probes_before']==1 and event['resume_primary_probes_after']==2
    assert event['primary_cleared_after_resume'] and not p._locked_execution
    if kind=='cover':
        assert calls[:-1]==[('measure',c) for c in range(1,21)] and not p._remaining
    else:
        assert calls==[('resolve',2),('resolve',1)]


def test_budget_and_unknown_failures_propagate_in_locked_other_without_fake_resume():
    p=fixture();calls=[]
    def resolve(c):
        calls.append(c)
        raise _StopSearch('real_deadline')
    p._resolve=resolve
    with pytest.raises(_StopSearch,match='real_deadline'):
        p._execute_exchange(swap._LockedExchange(('source',2,Position(100.,0.)),1,{}))
    assert calls==[2] and not p._locked_execution


def test_sixteenth_actual_clear_during_resume_preserves_stop_and_final_event_evidence():
    p=fixture();event={};p.cleared=set(range(2,16))
    def resolve(c):
        p.actions+=1;p.cleared.add(c)
        if c==1:
            p.cleared.add(16)
            raise _StopSearch('source_count_upper_bound_reached')
        return True
    p._resolve=resolve
    with pytest.raises(_StopSearch,match='source_count_upper_bound_reached'):
        p._execute_exchange(swap._LockedExchange(('source',16,Position(100.,0.)),1,event))
    assert event['other_end_actions']==event['resume_start_actions']
    assert event['locked_final_actions']==2 and event['primary_cleared_final']
    assert not p._locked_execution


def test_persistent_primary_probe_budget_reaches_existing_optical_fallback(monkeypatch):
    p=fixture();indices=[]
    p._next_probe=lambda c,i:indices.append(i) or Position(0.,0.)
    p._perform=lambda *args:setattr(p,'actions',p.actions+1)
    monkeypatch.setattr('strategies.geometric_preempt.clearance_grid',lambda *args,**kwargs:(Position(500.,0.),))
    p._clear=lambda q,c,phase:p.cleared.add(c) or True
    with pytest.raises(swap._LockedExchange):
        p._resolve(1)
    assert indices==[0]
    p._locked_execution=True
    assert p._resolve(1)
    assert indices==[0,1,2] and p._primary_probe_counts[1]==3


def test_disabled_matches_plain_single_exactly():
    scenario=random_scenario(3,118001)
    old=run_probe_cost_search(ObservationOnlyClient(LocalResearchSimulator(scenario).client()))
    new=swap.run_swap_search(ObservationOnlyClient(LocalResearchSimulator(scenario).client()),enabled=False)
    assert old.action_history==new.action_history and old.virtual_time_s==new.virtual_time_s


def test_observation_only_complete_cost_and_locked_execution_ledger():
    simulator=LocalResearchSimulator(random_scenario(3,118001))
    report=swap.run_swap_search(ObservationOnlyClient(simulator.client()))
    actual=simulator.evaluation()
    assert actual['all_cleared'] and report.completion_certified_under_model
    assert actual['failed_clear_count']==0 and actual['virtual_time_s']==report.virtual_time_s
    assert sum(report.time_breakdown.values())==pytest.approx(report.virtual_time_s,abs=1e-4)
    for event in report.source_interruptions['events']:
        assert event['other_end_actions']==event['resume_start_actions']
        assert event['other_end_actions']>event['other_start_actions']
        assert event['resume_primary_probes_before']>=event['primary_probes']
    assert max(report.source_interruptions['probe_counts'].values(),default=0)<=6


@pytest.mark.parametrize('kwargs',[{'problem':4},{'enabled':1},{'max_active_probes':True}])
def test_invalid_options_before_enter(kwargs):
    simulator=LocalResearchSimulator(random_scenario(3,118001))
    with pytest.raises(ValueError):
        swap.run_swap_search(ObservationOnlyClient(simulator.client()),**kwargs)
    assert not simulator.observation_history()
