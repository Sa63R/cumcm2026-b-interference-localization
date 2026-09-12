import copy,json,math,pathlib,random,tempfile,unittest
from unittest.mock import patch
import q4_baseline as b
from q4_state import Engine,Action
from q4_v5 import solve_v5
from q4_v4_solver import V4Config
from q4_transit import TransitEngine
from q4_transit_critic import Critic,proposals,apply_probe
from q4_learned_transit import solve_learned
from q4_route_critic import route_features
from q4_learned_route import solve_learned_route
from validate_v5 import DeviceOnly,AuditSimulator,make_targets

class LearnedTransitTests(unittest.TestCase):
    def environment(self,seed=213010011,trace=False):
        return AuditSimulator(make_targets(seed,.5,'uniform'),seed,'hash',trace=trace)
    def test_disabled_transit_reproduces_v5(self):
        for seed in [213010012,213010013]:
            e0=self.environment(seed,True);e1=self.environment(seed,True)
            solve_v5(DeviceOnly(e0));r=solve_learned(DeviceOnly(e1),threshold=0.,maxstops=0)
            self.assertEqual(e0.trace,e1.trace);self.assertEqual(r['transit_stops'],0)
    def test_reject_all_learned_actions_reproduces_v5(self):
        e0=self.environment(213010014,True);e1=self.environment(213010014,True)
        solve_v5(DeviceOnly(e0))
        with patch('q4_learned_transit.Critic.predict',return_value=-1e20):r=solve_learned(DeviceOnly(e1))
        self.assertEqual(e0.trace,e1.trace);self.assertEqual(r['transit_stops'],0)
    def test_zero_cost_constant_does_not_force_route_change(self):
        e0=self.environment(213010015,True);e1=self.environment(213010015,True)
        solve_v5(DeviceOnly(e0))
        with patch('q4_learned_route.Critic.predict',return_value=0.):r=solve_learned_route(DeviceOnly(e1),threshold=0.)
        self.assertEqual(e0.trace,e1.trace);self.assertEqual(r['route_accepts'],0)
    def test_proposals_lie_on_original_leg_and_preserve_state(self):
        sim=self.environment();engine=Engine(DeviceOnly(sim));n=0
        for _ in range(45):
            if engine.done():break
            a=engine.ordered_actions()[0];before=engine.state.clone();p=engine.device.position;end=TransitEngine.entry(engine,a);summary=sim.summary()
            for prop in proposals(engine,a):
                q=prop['point'];self.assertAlmostEqual(b.dist(p,q)+b.dist(q,end),b.dist(p,end),places=7)
                self.assertEqual(len(prop['features']),57);self.assertTrue(all(math.isfinite(x)for x in prop['features']));n+=1
                self.assertTrue(set(prop['channels']).issubset(engine.state.pending))
            self.assertEqual(before,engine.state);self.assertEqual(summary,sim.summary());engine.execute(a)
        self.assertGreater(n,0)
    def test_route_features_do_not_mutate(self):
        sim=self.environment(213010016);engine=Engine(DeviceOnly(sim));engine.execute(Action('site',0))
        before=engine.state.clone();summary=sim.summary();rr=route_features(engine,engine.ordered_actions())
        self.assertTrue(rr);self.assertTrue(all(len(f)==83 for a,f in rr));self.assertEqual(before,engine.state);self.assertEqual(summary,sim.summary())
    def test_ordinary_negative_probe_is_not_success(self):
        sim=AuditSimulator([b.Target(1,(300.,0.),1000.,None)],213010017)
        engine=Engine(DeviceOnly(sim));engine.execute(Action('site',0));before=len(engine.state.cleared)
        n=apply_probe(engine,dict(point=(2500.,0.),channels=[1]));self.assertEqual(n,1)
        self.assertEqual(sim.clear_count,0);self.assertEqual(len(engine.state.cleared),before);self.assertIn(1,engine.state.pending)
        self.assertEqual(engine.state.rf[1][-1].obs.status,'none');self.assertGreater(sim.virtual_seconds,500.)
    def test_strong_probe_requires_real_clear(self):
        sim=AuditSimulator([b.Target(1,(300.,0.),1000.,None)],213010018)
        engine=Engine(DeviceOnly(sim));engine.execute(Action('site',0))
        apply_probe(engine,dict(point=(300.,0.),channels=[1]));self.assertEqual(sim.clear_count,1);self.assertIn(1,engine.state.cleared)
        self.assertTrue(engine.state.optical[1][-1].success)
    def test_failed_clear_cannot_remove_source(self):
        sim=AuditSimulator([b.Target(1,(300.,0.),1000.,None)],213010019)
        engine=Engine(DeviceOnly(sim));engine.execute(Action('site',0))
        with patch.object(sim,'clear',return_value=False):
            with self.assertRaises(RuntimeError):apply_probe(engine,dict(point=(300.,0.),channels=[1]))
        self.assertIn(1,engine.state.pending);self.assertEqual(len(engine.state.cleared),0)
    def test_finite_stop_budget_and_cost_accounting(self):
        sim=self.environment(213010020)
        with patch('q4_learned_transit.Critic.predict',return_value=1e9):r=solve_learned(DeviceOnly(sim),maxstops=3)
        self.assertLessEqual(r['transit_stops'],3);self.assertEqual(sim.clear_count,len(sim._targets))
        self.assertAlmostEqual(sim.virtual_seconds,sim.distance_m/5+5*sim.detect_count+sim.switch_count+3*sim.optical_count+2*sim.clear_count,places=6)
    def test_selected_v6_feedback_only_and_costs(self):
        from q4_v6 import solve_v6,default_config
        for seed,place,err in [(214010001,'uniform','hash'),(214010002,'outward_boundary','minus')]:
            sim=AuditSimulator(make_targets(seed,.75,place),seed,err)
            r=solve_v6(DeviceOnly(sim));self.assertEqual(sim.clear_count,len(sim._targets))
            self.assertLessEqual(r['transit_stops'],16);self.assertNotEqual(r['stop_certificate'],'in_progress')
            self.assertAlmostEqual(sim.virtual_seconds,sim.distance_m/5+5*sim.detect_count+sim.switch_count+3*sim.optical_count+2*sim.clear_count,places=6)
    def test_invalid_v6_budget_rejected(self):
        from q4_v6 import solve_v6,default_config
        from dataclasses import replace
        with self.assertRaises(ValueError):solve_v6(DeviceOnly(self.environment()),replace(default_config(),max_transit_stops=-1))
    def test_feature_validation(self):
        critic=Critic('transit_critic_huber.json')
        with self.assertRaises(ValueError):critic.predict([0.]*56)
        with self.assertRaises(ValueError):critic.predict([float('nan')]*57)
    def test_model_is_reusable_and_deterministic(self):
        a=Critic('transit_critic_huber.json');b_=Critic('transit_critic_huber.json');rng=random.Random(13)
        for _ in range(30):
            x=[rng.uniform(0,1500)for _ in range(57)]
            self.assertEqual(a.predict(x),b_.predict(x));self.assertTrue(math.isfinite(a.predict(x)))
    def test_no_hidden_data_interface(self):
        d=DeviceOnly(self.environment())
        for name in ['_targets','_seed','clear_count','summary','__dict__']:
            with self.assertRaises(AttributeError):getattr(d,name)

if __name__=='__main__':unittest.main()
