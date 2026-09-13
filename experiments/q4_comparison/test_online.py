import copy, math, random, unittest
import common
import q4_baseline as b
from q4_v4_solver import solve_v4
from online import State, finish, solve_resumable
from posterior import Worlds, ConditionalSimulator, compatible_position
class PublicDevice:
    def __init__(self,d): self.__d=d
    @property
    def position(self): return self.__d.position
    @property
    def channel(self): return self.__d.channel
    def move(self,p): return self.__d.move(p)
    def detect(self,c): return self.__d.detect(c)
    def clear(self,c): return self.__d.clear(c)
class OnlineTests(unittest.TestCase):
    def test_exact_trace_equivalence(self):
        for group in range(6):
            frac=[.25,.5,.75,.5,.5,.5][group]
            placement='outward_boundary' if group==3 else 'uniform'
            error=['hash']*4+['plus','minus']
            for i in range(3):
                seed=106000000+group*100000+i
                sims=[b.LocalSimulator(b.make_case(seed,frac,placement),seed,error[group],True) for _ in range(2)]
                original=solve_v4(PublicDevice(sims[0]));refactored=solve_resumable(PublicDevice(sims[1]))
                self.assertEqual(sims[0].trace,sims[1].trace,(group,i))
                self.assertEqual(sims[1].clear_count,len(sims[1]._targets))
                for key,value in original.items(): self.assertEqual(value,refactored[key],key)
    def test_saved_state_continuation(self):
        sim=b.LocalSimulator(b.make_case(106900001,.5),106900001,'hash',True);state=State()
        for _ in range(4):
            state.prepare();state.execute(PublicDevice(sim),state.ordered_actions(sim)[0])
        other,other_state=copy.deepcopy(sim),copy.deepcopy(state)
        finish(state,PublicDevice(sim));finish(other_state,PublicDevice(other))
        self.assertEqual(sim.trace,other.trace)
    def test_scenes_obey_history_and_counts(self):
        sim=b.LocalSimulator(b.make_case(106900002,.5),106900002);state=State()
        for _ in range(8):
            state.prepare();state.execute(PublicDevice(sim),state.ordered_actions(sim)[0])
        rng=random.Random(74);worlds=Worlds(state,rng,particles=32,unknown_particles=128)
        for _ in range(12):
            targets,seed=worlds.sample(rng)
            self.assertTrue(10<=len(targets)<=16)
            self.assertEqual(len(targets),len({t.channel for t in targets}))
            self.assertTrue(0<sum(t.direction is not None for t in targets)<len(targets))
            env=ConditionalSimulator(copy.deepcopy(targets),seed,state.history,sim.position,sim.channel)
            for t in targets:
                self.assertTrue(compatible_position(t.position,state.history.get(t.channel,[])))
                env._targets[t.channel].cleared=False
                for kind,p,status,theta in state.history.get(t.channel,[]):
                    env.move(p)
                    if kind=='detect':
                        ob=env.detect(t.channel);self.assertEqual(ob.status,status)
                        if status=='bearing': self.assertAlmostEqual((ob.theta-theta+math.pi)%(2*math.pi)-math.pi,0.)
                    else:
                        self.assertEqual(env.clear(t.channel),status);env._targets[t.channel].cleared=False
    def test_analytic_only_completes(self):
        for mode in ['hash','plus','minus']:
            sim=b.LocalSimulator(b.make_case(106900003,.75),106900003,mode)
            solve_resumable(PublicDevice(sim),analytic=True)
            self.assertEqual(sim.clear_count,len(sim._targets))
if __name__=='__main__': unittest.main(verbosity=2)
