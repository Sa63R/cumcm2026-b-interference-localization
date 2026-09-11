"""Synthetic physical prefixes; the range helper never reads evaluation truth."""

import copy

import pytest

from experiments.audit_q4_range import audit_range_prefix
from tests.test_audit_q4_state import RecordBuilder


def record():
    b = RecordBuilder(count=14)
    b.sources[0].update(x=1700.,y=0.,orientation_deg=180.)
    b.add('/measure',(1400.,0.),1,'coverage')
    b.parameters['range_skipped_scans']=[{
        'channel':1,'position':[-1800.,0.],'after_actual_action_count':1,
        'reason':'positive_region_beyond_max_reception_radius',
        'distance_lower_bound_m':3000.,'margin_m':1e-5,'positive_observation_count':1}]
    return b.record(successful=False)


def test_valid_far_witness_uses_real_prefix_not_truth_and_adds_no_absence_credit():
    r=record(); del r['evaluation']
    result=audit_range_prefix(r)
    assert result['passed'] and result['range_skips_verified']==1
    assert result['inferred_absence_credits']==result['inferred_observation_count']==0


@pytest.mark.parametrize('mutation',['unknown','future','prefix_before_positive','not_far','no_margin',
                                    'overstated','count','negative_constraint','fake_action','bearing','cleared'])
def test_rejects_bad_range_prefix_or_fabricated_negative(mutation):
    r=record(); event=r['summary']['strategy_parameters']['range_skipped_scans'][0]
    if mutation=='unknown':event['channel']=2
    if mutation=='future':event['after_actual_action_count']=9
    if mutation=='prefix_before_positive':event['after_actual_action_count']=0
    if mutation=='not_far':event['position']=[1500.,0.]
    if mutation=='no_margin':event['distance_lower_bound_m']=1500.000001
    if mutation=='overstated':event['distance_lower_bound_m']=999999.
    if mutation=='count':event['positive_observation_count']=2
    if mutation=='negative_constraint':r['summary']['strategy_parameters']['inferred_no_signal_constraints']=[event]
    if mutation=='fake_action':r['summary']['action_history'].append(copy.deepcopy(r['summary']['action_history'][0]))
    if mutation=='bearing':r['summary']['action_history'][0]['bearing_deg']+=1
    if mutation=='cleared':
        clear={'index':2,'action':'/clear','channel':1,'position':{'x':1700.,'y':0.},
               'response':{'clear_result':'success','accepted':True,'virtual_time_s':1.}}
        r['history'].insert(-1,clear)
        r['summary']['action_history'].append({'action':'clear','channel':1,'position':[1700.,0.],
                                             'result':'success','virtual_time_s':1.})
        event['after_actual_action_count']=2
    with pytest.raises(ValueError):audit_range_prefix(r)


def test_silent_real_feedback_does_not_create_positive_range_evidence():
    r=record()
    r['history'][1]['response']['measure_result']='no_signal'
    r['summary']['action_history'][0]['result']='no_signal'
    with pytest.raises(ValueError,match='positive'):
        audit_range_prefix(r)
