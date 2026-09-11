import ast
import inspect
import textwrap

import pytest

from planning.chain_route import ChainSource, solve_chain_route
from strategies.q4_exact_frontier import Q4ExactFrontier, run_q4_exact_frontier
from strategies.q4_joint_continuation import Q4JointContinuation
from strategies.q4_r2_scheduling import Q4R2Scheduling
from tests.test_q4_clear_before_probe import Replies


def test_scheduler_body_is_inherited_except_one_refinement():
    original=ast.parse(textwrap.dedent(inspect.getsource(Q4R2Scheduling._execute_plan)))
    changed=ast.parse(textwrap.dedent(inspect.getsource(Q4ExactFrontier._execute_plan)))
    class RemoveRefinement(ast.NodeTransformer):
        def visit_Assign(self,node):
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='_refine_route':
                return None
            return node
    changed=RemoveRefinement().visit(changed)
    assert ast.dump(original,include_attributes=False)==ast.dump(changed,include_attributes=False)


def test_other_behavior_and_astar_work_unchanged(monkeypatch):
    from simulator_client.state import Position
    monkeypatch.setattr('planning.q4_directional_cover.certified_cover_points',lambda profile:
        ((Position(0,0),Position(100,0)),{'passed':True,'test_only':True}))
    p=Q4ExactFrontier(Replies(),20000,6,max_expansions=200)
    for method in ('_perform','_next_probe','_resolve','_clear','_scan','_early_service','_ready','_target','run'):
        assert getattr(Q4ExactFrontier,method) is getattr(Q4JointContinuation,method)
    sources=[ChainSource((3.,4.))];covers=[(10.,0.)]
    old=solve_chain_route(covers,sources,max_expansions=0,background_scan_s=108.,scan_source_s=0.)
    p.total_expansions=123
    selected=p._refine_route(old,covers,sources,[1],108.)
    assert p.total_expansions==123
    e=p.exact_frontier_log[-1]
    assert e['astar_total_expanded']==123 and e['source_channels']==[1]
    assert e['dp_total_states']==(e['dp_result']['expanded'] if e['called'] else 0)


@pytest.mark.parametrize('kwargs',[{'problem':3},{'config':'bad'},{'max_actions':True},{'max_active_probes':-1},{'max_expansions':10001}])
def test_validation_before_client_access(kwargs):
    with pytest.raises(ValueError):run_q4_exact_frontier(object(),**kwargs)
