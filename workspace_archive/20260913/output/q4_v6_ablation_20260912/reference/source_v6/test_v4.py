"""V4 unit, feedback, finite-progress and routing-equivalence tests."""
import unittest,random,math
import q4_baseline as b
from q4_v4_solver import Belief,V4Config,solve_v4
from q4_v4_local import LocalConfig,localize_step
from q4_v3_solver import solve_v3,V3Config

class V4Tests(unittest.TestCase):
    def test_cached_routes_are_identical(self):
        from q4_route import multi_route as old
        from q4_route_cached import multi_route as new
        rng=random.Random(91229)
        for _ in range(500):
            n=rng.randrange(0,35);pts=[(rng.uniform(-2000,2000),rng.uniform(-2000,2000)) for i in range(n)]
            here=(rng.uniform(-2000,2000),rng.uniform(-2000,2000));starts=rng.randrange(1,12)
            self.assertEqual(old(here,list(range(n)),pts,starts),new(here,list(range(n)),pts,starts))
    def test_disable_pause_and_cap_six_reproduces_v3(self):
        for seed in range(83312000,83312030):
            sims=[b.LocalSimulator(b.make_case(seed,.5),seed,'hash',trace=True) for _ in range(2)]
            solve_v3(sims[0],V3Config())
            cfg=V4Config(local=LocalConfig(optical_cover_limit=6,pause_after_pair=False))
            solve_v4(sims[1],cfg)
            self.assertEqual(sims[0].trace,sims[1].trace)
    def test_pause_is_not_clear_and_progress_is_persistent(self):
        sim=b.LocalSimulator([b.Target(1,(970.,20.),1200.,None)],9387,'plus')
        ob=sim.detect(1)
        tr=Belief(channel=1,anchor=sim.position,obs=ob,poly=b.initial_polygon(sim.position,ob.theta,1500.),measured=[sim.position],positives=[sim.position])
        cfg=LocalConfig(optical_cover_limit=0)
        seen=[]
        for _ in range(14):
            r=localize_step(sim,tr,cfg)
            if r.get('complete') is False:
                self.assertEqual(sim.clear_count,0);seen.append(tr.rf_rounds)
            else:break
        self.assertEqual(sim.clear_count,1)
        self.assertEqual(seen,list(range(1,len(seen)+1)))
        self.assertLessEqual(len(seen),12)
    def test_exhausted_fast_rounds_enters_fallback(self):
        sim=b.LocalSimulator([b.Target(1,(870.,75.),1200.,None)],2819,'minus')
        ob=sim.detect(1)
        tr=Belief(channel=1,anchor=sim.position,obs=ob,poly=b.initial_polygon(sim.position,ob.theta,1500.),measured=[sim.position],positives=[sim.position],rf_rounds=12)
        r=localize_step(sim,tr,LocalConfig())
        self.assertIn('fallback',r['certificate']);self.assertEqual(sim.clear_count,1)
    def test_exact_direction_boundaries_extreme_errors(self):
        count=0
        for error in ['hash','plus','minus']:
          for r in [5.00001,10.,20.,100.,700.,1499.99]:
            for a in [0.,.3,1.7,3.1,5.7]:
              for turn in [-math.pi/2,math.pi/2]:
                g=b.mul(r,b.unit(a));u=b.unit(a+math.pi+turn)
                sim=b.LocalSimulator([b.Target(1,g,1500.,u)],739,error)
                ob=sim.detect(1);self.assertNotEqual(ob.status,'none')
                poly=b.initial_polygon(sim.position,ob.theta,1500.) if ob.status=='bearing' else []
                tr=Belief(channel=1,anchor=sim.position,obs=ob,poly=poly,measured=[sim.position],positives=[sim.position],negative_enabled=True)
                for _ in range(14):
                    rep=localize_step(sim,tr,LocalConfig())
                    if rep.get('complete') is not False:break
                self.assertEqual(sim.clear_count,1);count+=1
        self.assertEqual(count,180)
    def test_policy_uses_public_feedback_only(self):
        class PublicDevice:
            def __init__(self,d):self.__d=d
            @property
            def position(self):return self.__d.position
            @property
            def channel(self):return self.__d.channel
            def move(self,p):return self.__d.move(p)
            def detect(self,c):return self.__d.detect(c)
            def clear(self,c):return self.__d.clear(c)
        sim=b.LocalSimulator(b.make_case(83322001,.75),83322001)
        result=solve_v4(PublicDevice(sim))
        self.assertEqual(sim.clear_count,len(sim._targets))
        self.assertLessEqual(result['global_actions'],21+16*13)
    def test_failed_certified_clear_is_never_success(self):
        class Broken:
            position=(0.,0.);channel=1
            def move(self,p):self.position=p
            def detect(self,c):return b.Observation('strong')
            def clear(self,c):return False
        tr=Belief(channel=1,anchor=(0.,0.),obs=b.Observation('strong'),poly=[])
        with self.assertRaises(RuntimeError):localize_step(Broken(),tr)

if __name__=='__main__':unittest.main(verbosity=2)
