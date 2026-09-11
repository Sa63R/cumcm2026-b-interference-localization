import copy
import time

from experiments.audit_q3_round3_journals import audit_branch, run
from strategies.q3_round3 import Round3Rollout, Round3Stepper, Proposal, plain
from strategies.q3_fresh_belief import sample_worlds
from tests.test_q3_round3 import prefix


def logged_branch():
    p=prefix(Round3Stepper);r=Round3Rollout()
    world=sample_worlds(p.history,count=1,seed=963001)[0]
    r._evaluate_proposal(p,Proposal('baseline',(p.pending,),'baseline'),world,time.perf_counter()+10,0,0,'confirm')
    return r.branch_records[0],plain(world),plain(p.history)


def test_independent_full_branch_accounting_and_tamper_detection():
    branch,world,history=logged_branch()
    errors,count=audit_branch(branch,world,history)
    assert not errors and count>0
    corrupt=copy.deepcopy(branch);corrupt['future_action_history'][0]['virtual_time_s']+=2
    assert 'cumulative_action_cost_mismatch' in audit_branch(corrupt,world,history)[0]


def test_complete_requires_explicit_exit_and_all_sources_cleared():
    branch,world,history=logged_branch()
    branch['terminal_session']='active'
    assert 'complete_without_explicit_exit' in audit_branch(branch,world,history)[0]


def test_missing_journals_is_not_a_passing_audit(tmp_path):
    assert run([tmp_path])['passed'] is False
