import bootstrap
import math,random,unittest
from bootstrap import b,make_case
from domain_candidate import clip_domain
from unrestricted_posterior import Worlds

class GeometryTests(unittest.TestCase):
    def test_domain_cut_keeps_boundary_and_rounded_bearings(self):
        rng=random.Random(720126)
        for i in range(600):
            theta=rng.random()*2*math.pi
            source=b.mul(1800.,b.unit(theta))
            sensor=b.add(source,b.mul(rng.uniform(20,1490),b.unit(rng.random()*2*math.pi)))
            truth=math.atan2(source[1]-sensor[1],source[0]-sensor[0])
            error=(-1 if i%2 else 1)*math.pi/180
            measured=math.radians(round(math.degrees((truth+error)%(2*math.pi)),2)%360)
            original=b.initial_polygon(sensor,measured,1500.)
            clipped=clip_domain(original)
            self.assertTrue(clipped)
            signed_area=sum(u[0]*v[1]-u[1]*v[0] for u,v in zip(clipped,clipped[1:]+clipped[:1]))
            sign=1 if signed_area>=0 else -1
            # The clipping routine preserves either original winding.
            for u,v in zip(clipped,clipped[1:]+clipped[:1]):
                cross=(v[0]-u[0])*(source[1]-u[1])-(v[1]-u[1])*(source[0]-u[0])
                self.assertGreaterEqual(sign*cross,-1e-5)
    def test_true_single_type_test_fixtures(self):
        self.assertTrue(all(t.direction is None for t in make_case(123,0.,'uniform')))
        self.assertTrue(all(t.direction is not None for t in make_case(123,1.,'uniform')))
    def test_unrestricted_sampler_accepts_single_type(self):
        class Pool:
            def __init__(self,direction):self.direction=direction
            def target(self,c,*args):return b.Target(c,(0.,0.),1200.,self.direction)
        class State:
            pending={i:None for i in range(1,11)}
            cleared=set()
            history={}
        w=Worlds.__new__(Worlds);w.state=State();w.unknown=[]
        w.select=lambda rng:(10,[])
        for direction in (None,(1.,0.)):
            w.pools={c:Pool(direction) for c in w.state.pending}
            targets,seed=w.sample(random.Random(1))
            self.assertEqual(len(targets),10)
            self.assertTrue(all(t.direction==direction for t in targets))

if __name__=='__main__':unittest.main(verbosity=2)
