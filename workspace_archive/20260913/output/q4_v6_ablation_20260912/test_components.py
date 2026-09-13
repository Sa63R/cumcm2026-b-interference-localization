"""Trace equivalence and checks that the two formerly coupled switches separate."""
import json
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch
import components as c
from components import bench
from jammers_local.core import Scenario,Session,LocalClient


def session(case, solver):
    s=Session(Scenario.from_dict(case['scenario']));client=LocalClient(s.dispatch);client.enter()
    report=solver(bench.Device(client));client.exit();bench.audit(s,report)
    return s,report


def canonical(s):
    return [(r['path'],r['request'],{k:v for k,v in r['response'].items()
        if k not in ('real_timestamp_ms','remaining_real_duration_s')}) for r in s.history]


class ComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bench.verify_sources();bench.initialize(1.005)
        cls.cases=json.loads((c.ROOT/'prior_plan.json').read_text())['cases']

    def test_reference_v4_v5_and_v6_are_exact(self):
        evidence=[]
        for case in [self.cases[0],self.cases[300],self.cases[335]]:
            for name,reference in [('v4',bench.solve_v4),('v5',bench.solve_v5),
                                   ('full',lambda d:bench.solve_v6(d,bench.V6_CONFIG))]:
                with self.subTest(case=case['key'],variant=name):
                    original,_=session(case,reference)
                    split,_=session(case,lambda d:c.solve(d,c.VARIANTS[name]))
                    self.assertEqual(canonical(original),canonical(split))
                    evidence.append(dict(case_key=case['key'],variant=name,
                        full_trace_equal=True,actions=len(original.history)))
        bench.dump(c.ROOT/'reference_equivalence.json',evidence)

    def test_existing_switch_variants_are_exact(self):
        variants={
            'no_route':replace(bench.V6_CONFIG,route_ranking=False),
            'no_transit':replace(bench.V6_CONFIG,transit_sensing=False),
            'no_guard':replace(bench.V6_CONFIG,minimum_stations=0),
            'no_rollout_rb':replace(bench.V6_CONFIG,use_v5_local=False)}
        for name,config in variants.items():
            with self.subTest(variant=name):
                original,_=session(self.cases[0],lambda d:bench.solve_v6(d,config))
                split,_=session(self.cases[0],lambda d:c.solve(d,c.VARIANTS[name]))
                self.assertEqual(canonical(original),canonical(split))

    def test_no_rollout_retains_rb_sharing(self):
        with patch.object(c,'local_candidates',side_effect=AssertionError('rollout called')):
            _,report=session(self.cases[0],lambda d:c.solve(d,c.VARIANTS['no_rollout']))
        self.assertEqual(report['planner_calls'],0)
        self.assertGreater(report['rb_share_checks'],0)

    def test_no_rb_retains_rollout(self):
        with patch.object(c,'RBEngine',side_effect=AssertionError('RB sharing enabled')):
            _,report=session(self.cases[0],lambda d:c.solve(d,c.VARIANTS['no_rb']))
        self.assertEqual(report['rb_share_checks'],0)
        self.assertGreater(report['planner_calls'],0)

    def test_geometry_transit_never_calls_a_learned_model(self):
        config=replace(c.VARIANTS['transit_geometry'],route=False)
        with patch.object(c,'Critic',side_effect=AssertionError('learned model loaded')):
            _,report=session(self.cases[0],lambda d:c.solve(d,config))
        self.assertGreater(report['transit_stops'],0)


if __name__=='__main__':unittest.main(verbosity=2)
