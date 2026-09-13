"""Geometry/feedback tests; synthetic scenarios, not official tests."""
import unittest,math,random
import q4_baseline as b
from q4_information import tighten_directional
from q4_v3_local import LocalConfig,optical_cover,localize
from q4_v3_solver import Belief,V3Config,solve_v3
from q4_v3_route import matrix_route

def inside(poly,p,tol=1e-5):
    cr=[(z[0]-a[0])*(p[1]-a[1])-(z[1]-a[1])*(p[0]-a[0]) for a,z in zip(poly,poly[1:]+poly[:1])]
    return all(c>=-tol for c in cr) or all(c<=tol for c in cr)

class V3Tests(unittest.TestCase):
    def test_optical_cover_contains_rectangles(self):
        rng=random.Random(15581)
        for _ in range(300):
            length=rng.uniform(10,450);width=rng.uniform(.1,38.)
            u=b.unit(rng.uniform(0,2*math.pi));v=(-u[1],u[0]);o=(rng.uniform(-1200,1200),rng.uniform(-1200,1200))
            point=lambda x,y:b.add(o,b.add(b.mul(x,u),b.mul(y,v)))
            poly=[point(x,y) for x,y in [(-length/2,-width/2),(length/2,-width/2),(length/2,width/2),(-length/2,width/2)]]
            cover=optical_cover(poly)
            self.assertTrue(cover)
            for x in [0.]+[length*(i/100-.5) for i in range(101)]:
                for y in [-width/2,0.,width/2]:
                    self.assertLessEqual(min(b.dist(c,point(x,y)) for c in cover),19.500001)
    def test_optical_failures_are_counted_and_no_false_completion(self):
        target=b.Target(1,(192.,0.),1000.,None)
        sim=b.LocalSimulator([target],7758,'hash')
        poly=[(100.,-2.),(200.,-2.),(200.,2.),(100.,2.)]
        tr=Belief(1,(0.,0.),b.Observation('bearing',0.),poly,[(0.,0.)],[(0.,0.)],[])
        rep=localize(sim,tr,LocalConfig(optical_cover_limit=6))
        self.assertEqual(rep['certificate'],'optical_cover_confirmed')
        self.assertEqual(sim.clear_count,1)
        self.assertEqual(sim.failed_clear_count,2)
        self.assertAlmostEqual(sim.virtual_seconds,sim.distance_m/5+3*3+2)
    def test_sixteen_known_sources_need_no_further_search(self):
        targets=[]
        for i in range(16):
            p=b.mul(100+20*i,b.unit(2*math.pi*i/16))
            targets.append(b.Target(i+1,p,1000.,b.mul(-1.,b.unit(2*math.pi*i/16)) if i else None))
        sim=b.LocalSimulator(targets,115,'plus')
        rep=solve_v3(sim,V3Config(known_bound_stop=True,shared_at_probes=False))
        self.assertEqual(sim.clear_count,16)
        self.assertEqual(rep['visited_stations'],[0])
        self.assertEqual(rep['stop_certificate'],'source_upper_bound')
    def test_negative_outer_constraints_preserve_truth(self):
        rng=random.Random(78961);used=0
        for i in range(350):
            g=b.mul(1600*math.sqrt(rng.random()),b.unit(2*math.pi*rng.random()))
            R=rng.uniform(1000,1500);u=None if i%5==0 else b.unit(2*math.pi*rng.random())
            target=b.Target(1,g,R,u);sim=b.LocalSimulator([target],981+i,'plus' if i%3==0 else 'hash')
            pos=[];neg=[];poly=None
            for j in range(25):
                p=b.add(g,b.mul(rng.uniform(50,1550),b.unit(2*math.pi*rng.random())))
                sim.move(p);ob=sim.detect(1)
                if ob.status=='bearing' and len(pos)<2:
                    pos.append(p)
                    poly=b.initial_polygon(p,ob.theta,1500.) if poly is None else b.clip_bearing(poly,p,ob.theta,1500.)
                elif ob.status=='none':neg.append(p)
            if poly is None:continue
            new,yes=tighten_directional(poly,pos,neg,16);used+=yes
            self.assertTrue(inside(new,g),msg=(g,poly,new,pos,neg))
        self.assertGreater(used,100)
    def test_route_is_a_permutation(self):
        rng=random.Random(771)
        for n in [1,2,3,8,15]:
            p=[(rng.uniform(-2000,2000),rng.uniform(-2000,2000)) for _ in range(n)]
            ent=[[b.add(q,(30.,40.)),b.add(q,(-30.,-40.))] for q in p]
            route=matrix_route((0,0),p,ent,4)
            self.assertEqual(sorted(route),list(range(n)))

class AdditionalV3Tests(unittest.TestCase):
    def test_count_patch_counts_distinct_channels(self):
        from q4_safe_patch import CountBoundDevice
        class Stub:
            position=(0.,0.);channel=1
            def __init__(self):self.calls=[]
            def detect(self,c):
                self.calls.append(c);self.channel=c
                return b.Observation('bearing',0.) if c<=16 else b.Observation('none')
            def move(self,p):self.position=p
            def clear(self,c):return True
        device=Stub();proxy=CountBoundDevice(device)
        proxy.detect(1);proxy.detect(1)
        self.assertEqual(len(proxy.found),1)
        for c in range(2,17):proxy.detect(c)
        old=len(device.calls)
        self.assertEqual(proxy.detect(17).status,'none')
        self.assertEqual(proxy.detect(20).status,'none')
        self.assertEqual(len(device.calls),old)
        self.assertEqual(proxy.skipped_empty_detections,2)
    def test_v3_exact_emission_boundary_and_extreme_errors(self):
        count=0
        for error in ['hash','plus','minus']:
          for r in [5.00001,10.,20.,100.,700.,1499.99]:
            for a in [0.,.3,1.7,3.1,5.7]:
              for rotate in [-math.pi/2,math.pi/2]:
                g=b.mul(r,b.unit(a));u=b.unit(a+math.pi+rotate)
                sim=b.LocalSimulator([b.Target(1,g,1500.,u)],739,error)
                ob=sim.detect(1)
                self.assertNotEqual(ob.status,'none')
                poly=b.initial_polygon((0.,0.),ob.theta,1500.) if ob.status=='bearing' else []
                tr=Belief(1,(0.,0.),ob,poly,[(0.,0.)],[(0.,0.)],[],True,16)
                localize(sim,tr,LocalConfig())
                self.assertEqual(sim.clear_count,1)
                count+=1
        self.assertEqual(count,180)

if __name__=='__main__':unittest.main(verbosity=2)
