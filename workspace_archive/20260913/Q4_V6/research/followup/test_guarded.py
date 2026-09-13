import dataclasses,unittest
from unittest.mock import patch
from validate_v5 import DeviceOnly,AuditSimulator,make_targets
from q4_v5 import solve_v5
from q4_v6 import solve_v6,default_config
from q4_v6_unprotected import solve_v6 as unprotected
from q4_count_search import solve_count_search

class GuardedTests(unittest.TestCase):
    def env(self,seed):return AuditSimulator(make_targets(seed,.5,'uniform'),seed,'hash',trace=True)
    def test_zero_guard_matches_unprotected(self):
        for seed in [223010001,223010002]:
            a=self.env(seed);b=self.env(seed)
            unprotected(DeviceOnly(a));solve_v6(DeviceOnly(b),dataclasses.replace(default_config(),minimum_stations=0))
            self.assertEqual(a.trace,b.trace)
    def test_final_matches_development_implementation(self):
        for seed in [223010003,223010004]:
            a=self.env(seed);b=self.env(seed)
            solve_count_search(DeviceOnly(a),max_scans=0,transit=True,route=True,min_visits=4)
            solve_v6(DeviceOnly(b));self.assertEqual(a.trace,b.trace)
    def test_early_prefix_unchanged_and_gate_observation_only(self):
        from q4_guarded_solver import proposals as orig_proposals, route_features as orig_features
        seen=[];snapshot=[]
        def probe(engine,action):
            self.assertGreaterEqual(len(engine.state.visited),4)
            if not snapshot:snapshot.extend(e.trace)
            seen.append(True);return orig_proposals(engine,action)
        def route(engine,actions):
            self.assertGreaterEqual(len(engine.state.visited),4)
            if not snapshot:snapshot.extend(e.trace)
            seen.append(True);return orig_features(engine,actions)
        for seed in [223010005,223010006]:
            e=self.env(seed);base=self.env(seed);snapshot=[]
            with patch('q4_guarded_solver.proposals',side_effect=probe),patch('q4_guarded_solver.route_features',side_effect=route):
                solve_v6(DeviceOnly(e))
            solve_v5(DeviceOnly(base))
            self.assertTrue(snapshot);self.assertEqual(snapshot,base.trace[:len(snapshot)])
        self.assertTrue(seen)
    def test_invalid_guard_rejected(self):
        for n in [-1,22,3.5]:
            with self.assertRaises(ValueError):solve_v6(DeviceOnly(self.env(223010007)),dataclasses.replace(default_config(),minimum_stations=n))
    def test_rejected_scores_keep_baseline_exact(self):
        a=self.env(223010008);b=self.env(223010008)
        solve_v5(DeviceOnly(a))
        with patch('q4_guarded_solver.Critic.predict',return_value=-1e20):r=solve_v6(DeviceOnly(b))
        self.assertEqual(a.trace,b.trace);self.assertEqual(r['transit_stops'],0);self.assertEqual(r['route_accepts'],0)

if __name__=='__main__':unittest.main()
