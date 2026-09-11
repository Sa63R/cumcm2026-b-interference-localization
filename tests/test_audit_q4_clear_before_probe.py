"""Only scripted controller replies and corrupt-record checks; no scenes."""
import copy
import importlib.util
from pathlib import Path

import pytest

from experiments.audit_q4_clear_before_probe import (
    audit_clear_before_probe_prefix, summarize_clear_before_probe_audits)


def fixture_module():
    spec = importlib.util.spec_from_file_location('probe_scripted_fixture',
        Path(__file__).with_name('test_q4_clear_before_probe.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wrap(policy, entered=False, reason=None):
    summary = policy.report.as_dict()
    if reason:
        summary['completion_reason'] = reason
    summary['source_estimates'] = {str(c): {'vertices': [list(p) for p in r.vertices]}
                                   for c, r in policy.regions.items()}
    raw = []
    if entered:
        raw.append({'action':'/enter','response':{'accepted':True,'max_virtual_duration_s':360000.}})
    for a in summary['action_history']:
        response = {'accepted':True,'virtual_time_s':a['virtual_time_s'],
                    'measure_result' if a['action']=='measure' else 'clear_result':a['result']}
        if a['result'] == 'direction':response['svd_deg'] = a['bearing_deg']
        raw.append(dict(action='/'+a['action'],channel=a['channel'],position=a['position'],response=response))
    return dict(summary=summary,history=raw,row={'successful':False},spec={'kwargs':{'max_actions':policy.max_actions}})


def example(monkeypatch, *, success=False, entered=False, skip=False, reject=None):
    f = fixture_module();policy,client = f.make(monkeypatch)
    center = f.prime(policy,client)
    if entered:
        policy.actions += 1  # Actual run() accepted enter before physical prefix.
    if skip:
        policy.max_actions = policy.actions+2
    client.measure_replies = [('no_signal',None)]
    if not success:
        client.clear_replies = ['no_target_in_range']
    client.reject_kind = reject
    if reject:
        with pytest.raises(f._StopSearch):f.pending_measure(policy,center)
    elif success:
        assert policy._resolve(1)
    else:
        f.pending_measure(policy,center)
    r = wrap(policy,entered,reason='request_rejected' if reject else None)
    if reject:
        r['history'].append(dict(action='/'+reject,channel=1,position=[center.x,center.y],response={'accepted':False}))
    return r


@pytest.mark.parametrize('entered', [False,True])
def test_real_failure_resumes_exact_measure_and_counts_enter(monkeypatch,entered):
    r = example(monkeypatch,entered=entered);before = copy.deepcopy(r)
    a = audit_clear_before_probe_prefix(r)
    assert a['attempts'] == a['failures'] == a['failed_then_measured'] == 1
    assert a['successes'] == 0 and r == before
    e = r['summary']['strategy_parameters']['clear_before_probe_log'][0]
    assert e['budget']['policy_action_count'] == 2+int(entered)
    assert a['decision_wall_s'] == e['decision_wall_s']
    assert summarize_clear_before_probe_audits([a,a])['attempts'] == 2


def test_actual_success_ends_source_without_fabricated_measure(monkeypatch):
    r = example(monkeypatch,success=True)
    a = audit_clear_before_probe_prefix(r)
    assert a['successes'] == a['attempts'] == 1 and a['failed_then_measured'] == 0
    assert len(r['summary']['action_history']) == 3


@pytest.mark.parametrize('entered', [False,True])
def test_insufficient_action_continuation_preserves_actual_original_probe(monkeypatch,entered):
    r = example(monkeypatch,entered=entered,skip=True)
    a = audit_clear_before_probe_prefix(r)
    assert a['skipped_budget'] == 1 and a['attempts'] == 0
    assert r['summary']['action_history'][-1]['action'] == 'measure'


@pytest.mark.parametrize('reject', ['clear','measure'])
def test_rejection_is_a_terminal_exception_not_a_fake_measure(monkeypatch,reject):
    r = example(monkeypatch,reject=reject)
    a = audit_clear_before_probe_prefix(r)
    assert a['terminal_interruptions'] == 1
    assert a['attempts'] == int(reject == 'measure')
    r['summary']['completion_reason'] = 'coverage_exhausted_and_all_detected_cleared'
    with pytest.raises(ValueError,match='terminal'):audit_clear_before_probe_prefix(r)


@pytest.mark.parametrize('bad', ['centre','radius','vertices','positive','phase','counter','exit_reserve',
                                'movement','switch_cost','spec_limit','skip_reason','result','fake_measure',
                                'final_C','wire','fee','clock','unlogged'])
def test_false_eligibility_budgets_or_feedback_rejected(monkeypatch,bad):
    r = example(monkeypatch);p = r['summary']['strategy_parameters'];e = p['clear_before_probe_log'][0]
    if bad == 'centre':e['position'][0] += 1
    elif bad == 'radius':e['radius_m'] = 19.9
    elif bad == 'vertices':e['vertices'][0][0] += .1
    elif bad == 'positive':e['positive_observation_count'] += 1
    elif bad == 'phase':e['speculative_phase'] = 'certified_clear'
    elif bad == 'counter':e['budget']['policy_action_count'] += 1
    elif bad == 'exit_reserve':e['budget']['remaining_actions'] += 1
    elif bad == 'movement':e['budget']['movement_ceiling_s'] += 1
    elif bad == 'switch_cost':e['budget']['worst_failure_cost_s'] -= 1
    elif bad == 'spec_limit':r['spec']['kwargs']['max_actions'] -= 1
    elif bad == 'skip_reason':e['budget']['skip_reasons'] = ['action_budget']
    elif bad == 'result':e['clear_result'] = 'success'
    elif bad == 'fake_measure':e['end_actual_action_count'] -= 1
    elif bad == 'final_C':r['summary']['source_estimates']['1']['vertices'][0][0] += .01
    elif bad == 'wire':r['history'][-1]['response']['measure_result'] = 'near'
    elif bad == 'fee':
        r['summary']['action_history'][-1]['virtual_time_s'] += 1
        r['history'][-1]['response']['virtual_time_s'] += 1
    elif bad == 'clock':e['decision_wall_s'] = float('nan')
    else:p['clear_before_probe_log'] = []
    with pytest.raises(ValueError):audit_clear_before_probe_prefix(r)


def test_second_accepted_speculative_attempt_for_same_source_rejected(monkeypatch):
    r = example(monkeypatch);h = r['summary']['action_history'];e = copy.deepcopy(r['summary']['strategy_parameters']['clear_before_probe_log'][0])
    n = len(h);now = h[-1]['virtual_time_s'];pos = h[-1]['position']
    for kind,result,phase,cost in [('clear','no_target_in_range','speculative_clear_before_probe',3),
                                  ('measure','no_signal','active_localization',5)]:
        now += cost
        a = dict(action=kind,result=result,phase=phase,position=pos,channel=1,virtual_time_s=now)
        h.append(a)
        r['history'].append(dict(action='/'+kind,channel=1,position=pos,response={
            'accepted':True,'virtual_time_s':now,'clear_result' if kind=='clear' else 'measure_result':result}))
    e.update(after_actual_action_count=n,end_actual_action_count=n+2,current_position=pos)
    r['summary']['strategy_parameters']['clear_before_probe_log'].append(e)
    with pytest.raises(ValueError,match='Repeated accepted'):audit_clear_before_probe_prefix(r)


def test_clear_success_cannot_be_followed_by_same_source_work(monkeypatch):
    r = example(monkeypatch,success=True);h = r['summary']['action_history'];now = h[-1]['virtual_time_s']+5
    a = dict(action='measure',channel=1,position=h[-1]['position'],result='no_signal',phase='active_localization',virtual_time_s=now)
    h.append(a);r['history'].append(dict(action='/measure',channel=1,position=a['position'],
        response={'accepted':True,'virtual_time_s':now,'measure_result':'no_signal'}))
    with pytest.raises(ValueError,match='stop this resolved source'):audit_clear_before_probe_prefix(r)


def test_observation_only_and_no_policy_reentry(monkeypatch):
    class Guard(dict):
        def __getitem__(self,key):
            assert key in {'summary','history','row','spec'}
            return super().__getitem__(key)
    r = example(monkeypatch)
    import strategies.q4_clear_before_probe as policy
    monkeypatch.setattr(policy.Q4ClearBeforeProbe,'_insertion_budget',lambda *args:pytest.fail('Audit called policy'))
    assert audit_clear_before_probe_prefix(Guard(r))['passed']


@pytest.mark.parametrize('skip', [False,True])
def test_actual_early_service_budget_and_original_measure_continuation(monkeypatch,skip):
    f = fixture_module();policy,client = f.make(monkeypatch)
    client.measure_replies = [('direction',0.),('direction',90.),('no_signal',None),('near',None)]
    policy._perform('measure',f.Position(-1000,0),1,'active_localization')
    policy._perform('measure',f.Position(0,-1000),1,'active_localization')
    current = f.Position(-260,-50) if skip else f.Position(0,-50)
    policy._perform('measure',current,2,'coverage')
    policy.report.coverage_points = [[current.x,current.y],[200.,0.]]
    candidate = policy._early_candidate(f.Position(200,0))
    assert candidate is not None
    policy._early_service(candidate)
    r = wrap(policy);a = audit_clear_before_probe_prefix(r)
    assert a['events'] == 1 and a['skipped_budget'] == int(skip)
    assert a['successes'] == int(not skip)
    e = r['summary']['strategy_parameters']['clear_before_probe_log'][0]
    if skip:
        assert e['budget']['skip_reasons'] == ['service_budget']
        assert r['summary']['action_history'][e['after_actual_action_count']]['action'] == 'measure'
    else:
        assert e['budget']['skip_reasons'] == []
