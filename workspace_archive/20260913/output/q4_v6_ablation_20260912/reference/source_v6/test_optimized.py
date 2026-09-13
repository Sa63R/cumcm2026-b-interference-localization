import unittest,math,random
import q4_baseline as b
from q4_fast_solver import FastConfig,make_sites,solve_fast
from q4_fast_local import LocalConfig,localize,nearest_clear_point
from q4_coverage import certify

class GeometryTests(unittest.TestCase):
    def test_continuous_coverage(self):
        sites=make_sites(FastConfig())
        self.assertEqual(len(sites),21)
        out=certify(sites)
        self.assertTrue(out['ok'])
        self.assertGreater(out['accepted'],100)
        self.assertEqual(3*out['subdivided']+1,out['accepted']+out['outside'])

    def test_uncertified_layout_fails_closed(self):
        self.assertFalse(certify([(0.,0.)])['ok'])

    def test_clear_projection(self):
        rng=random.Random(56)
        for _ in range(100):
            center=(rng.uniform(-1000,1000),rng.uniform(-1000,1000))
            pts=[b.add(center,b.mul(rng.uniform(0,19),b.unit(2*math.pi*k/5))) for k in range(5)]
            here=(rng.uniform(-2000,2000),rng.uniform(-2000,2000))
            q=nearest_clear_point(pts,here)
            self.assertTrue(all(b.dist(q,p)<=19.500001 for p in pts))
            mec,_=b.enclosing_circle(pts)
            self.assertLessEqual(b.dist(q,here),b.dist(mec,here)+1e-5)

class FeedbackTests(unittest.TestCase):
    def test_boundary_direction_and_extreme_errors(self):
        # Unit tests of one localization, not official full scenarios.
        count=0
        for error in ['hash','plus','minus']:
          for r in [5.00001,10.,20.,100.,700.,1499.99]:
            for angle in [0.,.3,1.7,3.1,5.7]:
              for rotate in [-math.pi/2,math.pi/2]:
                g=b.mul(r,b.unit(angle));u=b.unit(angle+math.pi+rotate)
                t=b.Target(1,g,1500.,u)
                sim=b.LocalSimulator([t],739,error)
                first=sim.detect(1)
                self.assertNotEqual(first.status,'none')
                localize(sim,b.Track(1,sim.position,first))
                self.assertEqual(sim.clear_count,1)
                self.assertEqual(sim.failed_clear_count,0)
                count+=1
        self.assertEqual(count,180)

    def test_same_location_same_error(self):
        sim=b.LocalSimulator([b.Target(1,(500,10),1000,None)],781)
        a=sim.detect(1);sim.move((1.,0.));sim.detect(1);sim.move((0.,0.));c=sim.detect(1)
        self.assertEqual(a,c)

    def test_ten_sources_at_boundary(self):
        targets=[b.Target(k+1,b.mul(1800.,b.unit(k*math.pi/5)),1000.,
                          None if k==0 else b.unit(k*math.pi/5)) for k in range(10)]
        sim=b.LocalSimulator(targets,710,'plus')
        report=solve_fast(sim)
        self.assertEqual(sim.clear_count,10)
        self.assertEqual(report['stop_certificate'],'coverage_complete')
        self.assertEqual(len(report['visited_stations']),21)

if __name__=='__main__':unittest.main(verbosity=2)
