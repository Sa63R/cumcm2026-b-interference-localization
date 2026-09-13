import math,random,unittest
from dataclasses import replace
from unittest.mock import patch
import q4_baseline as b
from q4_v5 import solve_v5,default_config
from q4_local_rollout import solve_local_rollout
from q4_rb_sharing import RBEngine
from q4_belief import PosteriorUnavailable
from q4_state import Engine
from validate_v5 import DeviceOnly,AuditSimulator,make_targets
from q4_coverage import certify

class V5Tests(unittest.TestCase):
    def test_default_parameters_frozen(self):
        c=default_config();self.assertEqual(c.scenes,24);self.assertEqual(c.min_gain,2.)
        self.assertTrue(c.rb_sharing);self.assertFalse(c.dynamic_coverage)
    def test_rb_gain_does_not_mutate_state(self):
        seed=108010001;s=AuditSimulator(b.make_case(seed,.5),seed)
        e=RBEngine(DeviceOnly(s));e.execute(e.ordered_actions()[0]);c=next(iter(e.state.pending))
        before=e.state.clone();report=s.summary();q=(300.,240.)
        g=e.posterior_gain(c,q)
        self.assertGreaterEqual(g,0.);self.assertEqual(before,e.state);self.assertEqual(report,s.summary())
    def test_rb_failure_returns_legacy_ranking(self):
        seed=108010003;s=AuditSimulator(b.make_case(seed,.5),seed)
        e=RBEngine(DeviceOnly(s));e.execute(e.ordered_actions()[0]);s.move((200.,200.))
        with patch.object(e.rb_factory,'pool',side_effect=PosteriorUnavailable('unit')):
            e.shared()
        self.assertGreater(e.rb_share_failures,0);self.assertFalse(e.done())
    def test_local_prior_failure_not_success(self):
        seed=108010005;s=AuditSimulator(b.make_case(seed,.75),seed)
        with patch('q4_local_rollout.SceneFactory.pool',side_effect=PosteriorUnavailable('unit')):
            r=solve_v5(DeviceOnly(s),replace(default_config(),max_decisions=3,rb_sharing=False))
        self.assertEqual(s.clear_count,len(s._targets));self.assertTrue(r['planner_failures'])
    def test_feedback_only_rb_local(self):
        for seed,place,err in [(108010006,'uniform','hash'),(108010007,'r1000','smooth'),(108010008,'cluster','hash')]:
            s=AuditSimulator(make_targets(seed,.5,place),seed,err)
            r=solve_v5(DeviceOnly(s),replace(default_config(),scenes=4,max_decisions=3))
            self.assertEqual(s.clear_count,len(s._targets));self.assertNotEqual(r['stop_certificate'],'in_progress')
            self.assertAlmostEqual(s.virtual_seconds,s.distance_m/5+5*s.detect_count+s.switch_count+3*s.optical_count+2*s.clear_count,places=6)
    def test_smooth_errors_bounded_and_fixed(self):
        s=AuditSimulator(b.make_case(108010008,.5),108010008,'smooth')
        for i in range(200):
            s.position=(i*25-2500,i*31-3000)
            a=s._error(i%20+1);self.assertLessEqual(abs(a),b.DELTA);self.assertEqual(a,s._error(i%20+1))
    def test_coverage_static_certificate(self):
        seed=108010010;s=AuditSimulator(b.make_case(seed,.5),seed)
        e=Engine(DeviceOnly(s));proof=certify(e.state.sites)
        self.assertTrue(proof['ok'])

if __name__=='__main__':unittest.main()
