import gzip
import json

import pytest

from simulation import LocalResearchSimulator, random_scenario
from strategies.q3_fresh_stepper import StepperQ3
from strategies.q3_round3 import Round3Stepper, Round3Rollout, Proposal, apply_proposal


def prefix(cls=StepperQ3, movable=False):
    sim = LocalResearchSimulator(random_scenario(3,910100))
    client = sim.client();client.enter()
    p = cls(client, movable_tail=movable) if cls is StepperQ3 else cls(client)
    for _ in range(22):
        p.next_action();p.execute_pending()
    p.next_action()
    return p


def install_costs(monkeypatch, planner, early, confirm):
    worlds = list(range(12))
    monkeypatch.setattr('strategies.q3_round3.sample_worlds',lambda *a,**k:worlds)
    def cost(p,proposal,world,deadline,wi,ci,stage):
        return 2000.0 if ci == 0 else 2000.0+(early if wi<4 else confirm)
    monkeypatch.setattr(planner,'_evaluate_proposal',cost)


def test_lucky_screening_worlds_cannot_override_unfavourable_confirmation(monkeypatch):
    p=prefix();a=p.pending;stack=list(p.stack)
    r=Round3Rollout();install_costs(monkeypatch,r,-1000,50)
    r.maybe_choose(p)
    d=r.stats['decisions'][0]
    assert d['mean_delta_s'] < 0  # The old mixed estimate is optimistic.
    assert d['confirmation']['mean_s']==50
    assert d['selected']=='baseline' and p.pending==a and p.stack==stack


def test_confirmation_uses_eight_common_worlds_and_not_screening(monkeypatch):
    p=prefix();r=Round3Rollout();install_costs(monkeypatch,r,100,-10)
    r.maybe_choose(p)
    d=r.stats['decisions'][0]
    assert d['selected']!='baseline'
    assert d['confirmation_delta_s']==[-10]*8
    assert d['screening']['mean_s']==100
    assert len(d['paired_delta_s'])==12


def find_cover_boundary():
    sim=LocalResearchSimulator(random_scenario(3,910103));p=Round3Stepper(sim.client());p.client.enter()
    for _ in range(800):
        if p.next_action() is None:break
        proposal=p.cover_proposal()
        if proposal is not None:return p,proposal
        p.execute_pending()
    raise AssertionError('Expected a movable cover departure in fixed regression case')


def test_cover_proposal_does_not_mutate_true_baseline():
    p,c=find_cover_boundary();before=(p.pending,list(p.stack),list(p.tail_plan),list(p.history))
    again=p.cover_proposal()
    assert again==c
    assert (p.pending,p.stack,p.tail_plan,p.history)==before
    r=Round3Rollout(evaluate_cover=True,budget_s=0)
    r.maybe_choose(p)
    assert (p.pending,p.stack,p.tail_plan,p.history)==before
    assert r.stats['overrides']==0


def test_adopted_cover_preserves_b_tasks_and_only_changes_current_scan():
    p,c=find_cover_boundary();stack=list(p.stack);old=p.pending
    apply_proposal(p,c)
    assert p.pending.channel==old.channel and p.pending.position!=old.position
    assert len(p.stack)==len(stack)
    for before,after in zip(stack,p.stack):
        if before[0] in {'scan_unknown','scan_known_start','scan_known'} and before[1]==old.position:
            assert after[2:]==before[2:]
        else:assert after==before
    assert not p.movable_tail
    result=p.run()
    assert result['completion_certified']


@pytest.mark.parametrize('seed',[910100,910103])
def test_g2_without_adoption_reproduces_every_b_action(seed):
    a=LocalResearchSimulator(random_scenario(3,seed));b=LocalResearchSimulator(random_scenario(3,seed))
    expected=StepperQ3(a.client()).run()
    actual=Round3Stepper(b.client()).run(planner=Round3Rollout(evaluate_cover=True,budget_s=0))
    assert expected['action_history']==actual['action_history']


def test_sampling_failure_reverts_to_true_b(monkeypatch):
    from strategies.q3_fresh_belief import BeliefSamplingError
    p,c=find_cover_boundary();a=p.pending;stack=list(p.stack)
    r=Round3Rollout(evaluate_cover=True)
    monkeypatch.setattr('strategies.q3_round3.sample_worlds',lambda *a,**k:(_ for _ in ()).throw(BeliefSamplingError('fixture')))
    r.maybe_choose(p)
    assert p.pending==a and p.stack==stack


def test_decision_artifact_preserves_state_proposals_and_group_assignments(monkeypatch,tmp_path):
    p=prefix();r=Round3Rollout(log_dir=tmp_path);install_costs(monkeypatch,r,-10,10)
    r.maybe_choose(p)
    files=list(tmp_path.glob('*.gz'));assert len(files)==1
    d=json.loads(gzip.open(files[0],'rb').read())
    assert d['controller']['stack'] and len(d['public_history'])==22
    assert d['proposals'][0]['kind']=='baseline'
    assert d['decision']['acceptance_world_indices']==list(range(4,12))
    assert d['decision']['selected']=='baseline'


def test_real_hypothetical_branch_logs_until_explicit_exit(tmp_path):
    import time
    from strategies.q3_fresh_belief import sample_worlds
    p=prefix(Round3Stepper);r=Round3Rollout(log_dir=tmp_path)
    world=sample_worlds(p.history,count=1,seed=962000)[0]
    cost=r._evaluate_proposal(p,Proposal('baseline',(p.pending,),'baseline'),world,time.perf_counter()+10,0,0,'confirm')
    branch=r.branch_records[-1]
    assert cost>0 and branch['complete'] and branch['terminal_session']=='exited'
    assert branch['future_action_history']
    assert p.client.state.session=='active' and p.pending is not None


def test_pressure_suite_keeps_whole_common_map_groups():
    from experiments.run_q3_fresh_round3 import build_suite
    cases,metadata=build_suite('pressure',24)
    assert len(cases)==24 and len(set(metadata['original_case_groups'].values()))==6
    for k in range(0,len(cases),4):
        a,b,c,d=cases[k:k+4]
        assert len(a.sources)==len(b.sources)==14
        assert len(c.sources)==len(d.sources)==16
        assert a.sources==c.sources[:14] and b.sources==d.sources[:14]
        assert a.seed==b.seed==c.seed==d.seed


def test_old_logged_planner_keeps_original_mixed_acceptance(monkeypatch,tmp_path):
    from strategies.q3_round3 import OldLoggedRollout
    p=prefix(movable=True);r=OldLoggedRollout(log_dir=tmp_path)
    monkeypatch.setattr('strategies.q3_fresh_rollout.sample_worlds',lambda *a,**k:list(range(12)))
    def cost(p,proposal,world,deadline):
        return 2000 if proposal==(p.pending,) else 2000+(-1000 if world<4 else 50)
    monkeypatch.setattr(r,'_evaluate',cost)
    r.maybe_choose(p)
    assert r.stats['decisions'][0]['selected']!='baseline'
    assert len(list(tmp_path.glob('*.gz')))==1
