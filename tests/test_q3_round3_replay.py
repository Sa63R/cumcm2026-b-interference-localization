from experiments.replay_q3_round3 import actual_continuation
from experiments.audit_q3_round3_journals import audit_branch,canonical
from strategies.q3_round3 import Proposal,Round3Stepper,digest,plain
from simulation import random_scenario
from tests.test_q3_round3 import prefix


def test_offline_actual_branch_is_accounted_and_cannot_mutate_live_state():
    p=prefix(Round3Stepper);world=random_scenario(3,910100)
    state=digest(p.client.state.snapshot());history=digest(p.history);stack=digest(p.stack)
    branch=actual_continuation(p,world,Proposal('baseline',(p.pending,),'baseline'))
    branch['world_sha256']=canonical(plain(world))
    assert branch['complete'] and branch['remaining_s']>0
    assert audit_branch(branch,plain(world),plain(p.history))[0]==[]
    assert state==digest(p.client.state.snapshot()) and history==digest(p.history) and stack==digest(p.stack)


def test_incomplete_actual_branch_has_no_performance_score():
    p=prefix(Round3Stepper)
    branch=actual_continuation(p,random_scenario(3,910100),Proposal('baseline',(p.pending,),'baseline'),timeout_s=-1)
    assert not branch['complete'] and branch['remaining_s'] is None
