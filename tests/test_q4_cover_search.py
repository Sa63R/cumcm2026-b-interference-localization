"""The new cover changes discovery geometry, not feedback or clear permissions."""
import pytest
from planning import coverage_points
from simulation import LocalResearchSimulator, random_scenario
from strategies.q4_state_search import run_q4_state_search
from strategies.q4_cover_search import run_q4_cover_search
from tests.test_strategy import ObservationOnlyClient


def legacy_cover(monkeypatch):
    import planning.q4_directional_cover as cover
    monkeypatch.setattr(cover,'certified_cover_points',lambda profile:
        (coverage_points(4,variant='triangular'),{'passed':True,'test_only':'legacy triangle cover'}))


def test_joint_with_legacy_geometry_preserves_v1_actions(monkeypatch):
    legacy_cover(monkeypatch)
    case=random_scenario(4,394901)
    old=run_q4_state_search(ObservationOnlyClient(LocalResearchSimulator(case).client()),mode='state_pruned')
    new=run_q4_cover_search(ObservationOnlyClient(LocalResearchSimulator(case).client()),schedule='joint')
    assert old.action_history==new.action_history
    assert old.virtual_time_s==new.virtual_time_s


def test_missing_certificate_rejected_before_any_action(monkeypatch):
    import planning.q4_directional_cover as cover
    monkeypatch.setattr(cover,'certified_cover_points',lambda profile: ([(0,0)],{'passed':False}))
    sim=LocalResearchSimulator(random_scenario(4,394901))
    with pytest.raises(ValueError,match='geometric certificate'):
        run_q4_cover_search(ObservationOnlyClient(sim.client()))
    assert not sim.observation_history()


@pytest.mark.parametrize('schedule',['joint','deferred'])
def test_compact_cover_uses_actual_feedback_and_completes(schedule):
    sim=LocalResearchSimulator(random_scenario(4,394902))
    result=run_q4_cover_search(ObservationOnlyClient(sim.client()),schedule=schedule)
    evaluation=sim.evaluation()
    assert result.error is None and result.exit_error is None
    assert evaluation['all_cleared'] and result.completion_certified_under_model
    assert sum(evaluation['time_breakdown_s'].values())==pytest.approx(result.virtual_time_s)
    assert sim.observation_history()[-1]['action']=='/exit'
    assert result.strategy_parameters['directional_cover_certificate']['passed']
    assert all(a['phase']=='guaranteed_clearance' for a in result.action_history
               if a['action']=='clear' and a['result']!='success')


@pytest.mark.parametrize('kwargs',[{'problem':3},{'max_expansions':True},{'max_actions':1},{'schedule':'bad'}])
def test_invalid_before_entry(kwargs):
    sim=LocalResearchSimulator(random_scenario(4,394901))
    with pytest.raises(ValueError): run_q4_cover_search(ObservationOnlyClient(sim.client()),**kwargs)
    assert not sim.observation_history()
