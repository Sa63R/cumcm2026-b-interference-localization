import copy,pickle,random,unittest
from unittest.mock import patch
import common
import q4_baseline as b
from q4_coverage import certify
from online import State,solve_resumable
from planner import Planner,PlanningConfig
from test_online import PublicDevice

class PlannerTests(unittest.TestCase):
    def state(self):
        sim=b.LocalSimulator(b.make_case(106910001,.5),106910001)
        state=State(analytic=True)
        state.execute(PublicDevice(sim),('site',0));state.prepare()
        return state,sim
    def test_planning_is_side_effect_free(self):
        state,sim=self.state();before=pickle.dumps(state);old=sim.summary().copy()
        planner=Planner(401,PlanningConfig(coarse_scenarios=2,verify_scenarios=2))
        action=planner.choose(state,PublicDevice(sim),state.ordered_actions(sim))
        self.assertEqual(before,pickle.dumps(state));self.assertEqual(old,sim.summary())
        self.assertIn(action,state.ordered_actions(sim))
    def test_timeout_and_empty_posterior_fallback(self):
        state,sim=self.state();ordered=state.ordered_actions(sim)
        planner=Planner(402,PlanningConfig(seconds_per_decision=0))
        self.assertEqual(planner.choose(state,PublicDevice(sim),ordered),ordered[0])
        self.assertEqual(planner.stats['timeouts'],1)
        planner=Planner(403)
        with patch('planner.Worlds',side_effect=ValueError('zero mass')):
            self.assertEqual(planner.choose(state,PublicDevice(sim),ordered),ordered[0])
        self.assertEqual(planner.stats['posterior_fallbacks'],1)
    def test_dynamic_final_certificate(self):
        seed=107100010
        sim=b.LocalSimulator(b.make_case(seed,.5),seed)
        cfg=PlanningConfig(coarse_scenarios=8,verify_scenarios=8,max_decisions=8,seconds_per_decision=5,shared=True,dynamic=True)
        planner=Planner(seed+720000000,cfg)
        report=solve_resumable(PublicDevice(sim),True,planner)
        self.assertEqual(sim.clear_count,len(sim._targets))
        self.assertGreater(planner.stats['coverage_passed'],0)
        if report['stop_certificate']=='coverage_complete': self.assertTrue(certify(report['actual_scanpoints'])['ok'])
    def test_failed_clear_never_becomes_success(self):
        class Broken:
            position=(0.,0.);channel=1
            def move(self,p):self.position=p
            def detect(self,c):return b.Observation('strong')
            def clear(self,c):return False
        with self.assertRaises(RuntimeError):solve_resumable(Broken(),True)

if __name__=='__main__':unittest.main(verbosity=2)
