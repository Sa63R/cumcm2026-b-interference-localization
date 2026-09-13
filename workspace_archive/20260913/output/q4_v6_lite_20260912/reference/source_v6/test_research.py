import copy, itertools, math, random, statistics, time, unittest
from unittest.mock import patch
import q4_baseline as b
from q4_v4_solver import solve_v4,V4Config
from q4_v4_local import LocalConfig
from q4_state import Engine,State,Action,RFRecord,OpticalRecord,solve_resumable
from q4_belief import SceneFactory,JointTypesCounts,PosteriorUnavailable,keypoint
from q4_rollout import RolloutConfig,RolloutPlanner,solve_rollout
from q4_local_rollout import solve_local_rollout,LocalRolloutConfig

class DeviceOnly:
    """No metadata, private source list, case seed or summary exposed to solver."""
    __slots__=('__sim',)
    def __init__(self,sim):object.__setattr__(self,'_DeviceOnly__sim',sim)
    @property
    def position(self):return object.__getattribute__(self,'_DeviceOnly__sim').position
    @property
    def channel(self):return object.__getattribute__(self,'_DeviceOnly__sim').channel
    def move(self,p):return object.__getattribute__(self,'_DeviceOnly__sim').move(p)
    def detect(self,c):return object.__getattribute__(self,'_DeviceOnly__sim').detect(c)
    def clear(self,c):return object.__getattribute__(self,'_DeviceOnly__sim').clear(c)
    def __getattribute__(self,name):
        if name in ('position','channel','move','detect','clear'):return object.__getattribute__(self,name)
        raise AttributeError('Only documented device interface is available')

class Resumption(unittest.TestCase):
    def test_original_action_equivalence(self):
        scenarios=[(.25,'uniform','hash'),(.5,'uniform','hash'),(.75,'uniform','hash'),(.75,'outward_boundary','hash'),(.5,'uniform','plus'),(.5,'uniform','minus')]
        for gi,(frac,place,error) in enumerate(scenarios):
            for i in range(3):
                seed=100100000+gi*100000+i
                cfg=V4Config(local=LocalConfig(optical_cover_limit=6 if i%2 else 12))
                sims=[]
                for func in (solve_v4,solve_resumable):
                    sim=b.LocalSimulator(b.make_case(seed,frac,place),seed,error,True)
                    func(DeviceOnly(sim),cfg);sims.append(sim)
                self.assertEqual(sims[0].trace,sims[1].trace,(gi,i))

    def test_resume_from_snapshot_matches_uninterrupted(self):
        seed=100222345;sims=[]
        for resume in [False,True]:
            sim=b.LocalSimulator(b.make_case(seed,.5),seed,trace=True)
            if resume:
                e=Engine(DeviceOnly(sim))
                for _ in range(7):e.execute(e.ordered_actions()[0])
                e=Engine(DeviceOnly(sim),state=e.state.clone());e.run_base()
            else:solve_v4(DeviceOnly(sim))
            sims.append(sim)
        self.assertEqual(sims[0].trace,sims[1].trace)

    def test_snapshot_mutation_isolated(self):
        sim=b.LocalSimulator(b.make_case(104000000,.25),104000000);e=Engine(DeviceOnly(sim))
        e.execute(e.ordered_actions()[0]);old=copy.deepcopy(e.state)
        cloned=e.state.clone();c=next(iter(cloned.pending))
        cloned.pending[c].rf_rounds+=4;cloned.pending[c].poly.clear();cloned.rf[c].clear();cloned.unknown.clear()
        self.assertEqual(e.state,old)

    def test_no_success_without_feedback(self):
        sim=b.LocalSimulator(b.make_case(104000000,.25),104000000);e=Engine(DeviceOnly(sim));e.execute(e.ordered_actions()[0])
        c=next(iter(e.state.pending))
        with self.assertRaises(RuntimeError):e.finish(c,{'certificate':'fake'})
        self.assertIn(c,e.state.pending)

class PosteriorTests(unittest.TestCase):
    def test_joint_dp_matches_enumeration(self):
        # Eight known sources, four uncertain channels, eight known-absent channels.
        opts=[(0.,1.,0.)]*7+[(0.,0.,1.)]+[(1.,.22,.08),(1.,.11,.19),(1.,.08,.23),(1.,.37,.07)]+[(1.,0.,0.)]*8
        dp=JointTypesCounts(opts)
        expected={n:0. for n in range(10,13)}
        for zz in itertools.product(range(3),repeat=4):
            n=8+sum(k>0 for k in zz)
            if n<10:continue
            w=math.prod(opts[8+i][z] for i,z in enumerate(zz))
            expected[n]+=w/(7*math.comb(20,n)*(1-2*.5**n))
        s=sum(expected.values());expected={n:w/s for n,w in expected.items()}
        s=sum(dp.count_weights.values());actual={n:w/s for n,w in dp.count_weights.items()}
        for n in expected:self.assertAlmostEqual(actual[n],expected[n],places=12)
        rng=random.Random(81)
        counts={n:0 for n in expected}
        for _ in range(4000):
            n,k=dp.sample(rng);counts[n]+=1
            self.assertEqual(sum(x>0 for x in k),n)
            self.assertTrue(1 in k and 2 in k)
            self.assertTrue(all(k[j]==0 for j in range(12,20)))
        for n in expected:self.assertLess(abs(counts[n]/4000-expected[n]),.035)

    def test_count_upper_bound(self):
        opts=[(0.,1.,0.)]*15+[(0.,0.,1.)]+[(1.,1.,1.)]*4
        dp=JointTypesCounts(opts)
        for i in range(20):
            n,k=dp.sample(random.Random(i));self.assertEqual(n,16);self.assertEqual(k[16:],[0,0,0,0])

    def test_generated_scenes_and_prefixes(self):
        for gi,(frac,place,err) in enumerate([(.5,'uniform','hash'),(.75,'outward_boundary','hash'),(.5,'uniform','plus'),(.5,'uniform','minus')]):
            seed=104000000+gi*100000
            sim=b.LocalSimulator(b.make_case(seed,frac,place),seed,err)
            e=Engine(DeviceOnly(sim));fac=SceneFactory(draws=128)
            for step in range(5):
                e.execute(e.ordered_actions()[0])
                scenes=fac.sample(e.state,3,random.Random(step))
                for scene in scenes:
                    fac.verify(scene,e.state)
                    self.assertTrue(10<=scene.count<=16)
                # Continuation is legal without exposing hidden sampled sources.
                dev=scenes[0].device(e.device.position,e.device.channel)
                child=Engine(DeviceOnly(dev),state=e.state.clone());child.run_base()
                self.assertTrue(all(t.cleared for t in dev._targets.values()))
                self.assertTrue(child.done())

    def test_same_location_returns_conditioned_observation(self):
        sim=b.LocalSimulator(b.make_case(104000000,.25),104000000);e=Engine(DeviceOnly(sim));e.execute(e.ordered_actions()[0])
        scene=SceneFactory().sample(e.state,1,random.Random(5))[0]
        dev=scene.device(e.device.position,e.device.channel)
        c=next(iter(e.state.pending));old=e.state.rf[c][0].obs
        self.assertEqual(dev.detect(c),old);self.assertEqual(dev.detect(c),old)
        self.assertEqual(dev.detect_count,2)

    def test_duplicate_evidence_not_counted_twice(self):
        sim=b.LocalSimulator(b.make_case(104000000,.25),104000000);e=Engine(sim);e.execute(e.ordered_actions()[0])
        c=next(iter(e.state.pending));f=SceneFactory()
        p=f.pool(e.state,c);s=e.state.clone();s.rf[c].append(s.rf[c][0])
        self.assertIs(f.pool(s,c),p)

    def test_inconsistent_probability_does_not_authorize_exit(self):
        sim=b.LocalSimulator(b.make_case(104000000,.25),104000000);e=Engine(sim);e.execute(e.ordered_actions()[0]);c=next(iter(e.state.pending))
        s=e.state.clone();s.rf[c].append(RFRecord((0.,0.),b.Observation('none')))
        with self.assertRaises(PosteriorUnavailable):SceneFactory().prepare(s)
        self.assertFalse(e.done());self.assertIn(c,e.state.pending)

class RolloutTests(unittest.TestCase):
    def test_no_lookahead_is_exact_v4(self):
        sims=[]
        for f in [solve_v4,lambda d:solve_rollout(d,RolloutConfig(base=V4Config(),max_decisions=0))]:
            sim=b.LocalSimulator(b.make_case(105100001,.5),105100001,trace=True);f(DeviceOnly(sim));sims.append(sim)
        self.assertEqual(sims[0].trace,sims[1].trace)

    def test_search_does_not_mutate_real_device_or_state(self):
        sim=b.LocalSimulator(b.make_case(104000000,.25),104000000);e=Engine(DeviceOnly(sim));e.execute(e.ordered_actions()[0])
        old=e.state.clone();oldsummary=sim.summary()
        p=RolloutPlanner(RolloutConfig(base=V4Config(),scenes=2,validation_scenes=2,position_draws=32,max_decisions=1))
        a=p.choose(e,e.ordered_actions())
        self.assertEqual(e.state,old);self.assertEqual(sim.summary(),oldsummary)
        self.assertIsInstance(a,Action);self.assertGreater(p.rollouts,0)

    def test_probability_failure_returns_base(self):
        sim=b.LocalSimulator(b.make_case(104000000,.25),104000000);e=Engine(sim);e.execute(e.ordered_actions()[0])
        p=RolloutPlanner(RolloutConfig(base=V4Config()));order=e.ordered_actions()
        with patch.object(p.factory,'prepare',side_effect=PosteriorUnavailable('synthetic test')):
            a=p.choose(e,order)
        self.assertEqual(a,order[0]);self.assertFalse(e.done())

    def test_timeout_returns_base(self):
        sim=b.LocalSimulator(b.make_case(104000000,.25),104000000);e=Engine(sim);e.execute(e.ordered_actions()[0])
        p=RolloutPlanner(RolloutConfig(base=V4Config(),decision_seconds=1e-6));order=e.ordered_actions()
        a=p.choose(e,order);self.assertEqual(a,order[0]);self.assertTrue(p.failures)

    def test_full_feedback_only_run(self):
        seed=105232151;sim=b.LocalSimulator(b.make_case(seed,.5),seed)
        report=solve_rollout(DeviceOnly(sim),RolloutConfig(base=V4Config(),scenes=2,validation_scenes=2,max_decisions=2,position_draws=32))
        self.assertEqual(sim.clear_count,len(sim._targets));self.assertNotEqual(report['stop_certificate'],'in_progress')
        self.assertAlmostEqual(sim.virtual_seconds,sim.distance_m/5+5*sim.detect_count+sim.switch_count+3*sim.optical_count+2*sim.clear_count,places=6)

    def test_local_feedback_only_run(self):
        seed=105232152;sim=b.LocalSimulator(b.make_case(seed,.75),seed)
        report=solve_local_rollout(DeviceOnly(sim),LocalRolloutConfig(scenes=8,position_draws=32,max_decisions=3))
        self.assertEqual(sim.clear_count,len(sim._targets));self.assertNotEqual(report['stop_certificate'],'in_progress')

if __name__=='__main__':unittest.main()
