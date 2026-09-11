import pytest
from experiments.analyze_q3_fresh_round3 import regret


def rows(ratios):
    values=[]
    for i,r in enumerate(ratios):
        for name,t in (('B',100),('G12',100*(1+r))):
            values.append(dict(case_id=str(i),strategy=name,virtual_time_s=t,analysis_success=True))
    return values


def test_positive_tail_has_fixed_denominator_and_zero_padding():
    result=regret(rows([.4]+[-.1]*19),'G12')
    assert result['max_positive_relative_regret']==pytest.approx(.4)
    assert result['tail10_all_cases_count']==2
    assert result['tail10_all_cases_positive_mean']==pytest.approx(.2)
    assert result['tail10_positive_cases_only_mean']==pytest.approx(.4)


def test_failed_execution_cannot_look_like_a_good_regret():
    values=rows([-.9]);values[-1]['analysis_success']=False
    assert regret(values,'G12')['complete'] is False


def test_all_improving_has_zero_positive_risk_but_negative_signed_max():
    result=regret(rows([-.2,-.3]),'G12')
    assert result['max_signed_relative_regret']==pytest.approx(-.2)
    assert result['max_positive_relative_regret']==0
    assert result['tail10_all_cases_positive_mean']==0
