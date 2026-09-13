"""Geometry, no-leakage, and numerical regression tests for Q3.

Truth is accessed ONLY by the independent test assertions, never by the agent.
"""
import math, unittest
import numpy as np
import q3_base as b
import q3_optimized as q


def inside(poly,g,tol=1e-4):
    if len(poly)==1:return b.norm(poly[0]-g)<=tol
    if len(poly)==2:
        u=poly[1]-poly[0];uu=float(u@u)
        p=poly[0] if uu==0 else poly[0]+np.clip(float((g-poly[0])@u)/uu,0,1)*u
        return b.norm(g-p)<=tol
    edges=np.roll(poly,-1,axis=0)-poly;delta=g-poly
    cross=edges[:,0]*delta[:,1]-edges[:,1]*delta[:,0]
    return bool(np.all(cross>=-tol*np.linalg.norm(edges,axis=1)))


class AuditedAgent(q.OptimizedAgent):
    """Independent test observer; its assertions never select an action."""
    def observe(self,*args,**kwargs):
        super().observe(*args,**kwargs)
        for c,t in self.tracks.items():
            g=self.env._sources[c].xy
            assert inside(t.poly,g), ('truth excluded',c,t.poly,g)
            assert b.norm(t.center-g)<=t.radius+1e-4
    def relocate_cover_points(self):
        super().relocate_cover_points()
        if self.unseen:
            assert q.certified_cover(self.full_scans+[self.cover[k] for k in self.remaining_cover])
    def select_probe(self,c):
        point,flag=super().select_probe(c)
        assert np.linalg.norm(self.tracks[c].poly-point,axis=1).max()<1000
        return point,flag


class GeometryTests(unittest.TestCase):
    def test_seven_cover(self):
        self.assertTrue(q.certified_cover(b.COVER))
        a=999.;R=1800.
        d=math.sqrt(R*R+a*a-2*R*a*math.cos(math.pi/7))
        self.assertLess(d,1000)
    def test_six_with_origin(self):
        for radius in (1124,1300,1450,1600,1730):
            pts=[np.zeros(2)]+[radius*np.array([math.cos(k*math.pi/3),math.sin(k*math.pi/3)]) for k in range(6)]
            self.assertTrue(q.certified_cover(pts))
    def test_detect_interior_hole(self):
        pts=[1600*np.array([math.cos(k*math.pi/3),math.sin(k*math.pi/3)]) for k in range(6)]
        self.assertFalse(q.certified_cover(pts))
        self.assertFalse(q.certified_cover(pts+pts))
    def test_incomplete_cover(self):
        self.assertFalse(q.certified_cover([]))
        self.assertFalse(q.certified_cover([np.zeros(2)]))
        for k in range(7):self.assertFalse(q.certified_cover([p for j,p in enumerate(b.COVER) if j!=k]))
    def test_random_cover_certificates(self):
        rng=np.random.default_rng(1877)
        angles=rng.uniform(0,2*math.pi,4000);rad=1800*np.sqrt(rng.random(4000))
        points=np.column_stack([rad*np.cos(angles),rad*np.sin(angles)])
        points=np.vstack([points,1800*np.column_stack([np.cos(np.linspace(0,2*math.pi,4000)),np.sin(np.linspace(0,2*math.pi,4000))])])
        for _ in range(40):
            centers=[np.zeros(2)]+[1400*np.array([math.cos(k*math.pi/3),math.sin(k*math.pi/3)])+rng.normal(0,80,2) for k in range(6)]
            if q.certified_cover(centers):
                self.assertLess(np.linalg.norm(points[:,None,:]-np.array(centers)[None,:,:],axis=2).min(axis=1).max(),1000)
    def test_negative_hull_contains_all_remaining_samples(self):
        rng=np.random.default_rng(241)
        for _ in range(100):
            poly=q.hull(rng.normal(0,800,(8,2)))
            neg=rng.normal(0,700,2)
            weights=rng.random((200,len(poly)));weights/=weights.sum(axis=1)[:,None]
            points=weights@poly
            points=points[np.linalg.norm(points-neg,axis=1)>=1000]
            if not len(points):continue
            new=q.outside_disk_hull(poly,neg)
            for x in points:self.assertTrue(inside(new,x))
            for _ in range(4):new=q.outside_disk_hull(new,neg)
            for x in points:self.assertTrue(inside(new,x))
    def test_range_halfplane(self):
        rng=np.random.default_rng(13)
        for _ in range(1000):
            g=rng.normal(0,700,2);R=rng.uniform(1000,1500)
            a,z=rng.uniform(0,2*math.pi,2)
            pos=g+rng.uniform(0,R)*np.array([math.cos(a),math.sin(a)])
            neg=g+rng.uniform(R+1e-3,R+1000)*np.array([math.cos(z),math.sin(z)])
            self.assertLess(float(2*(neg-pos)@g),float(neg@neg-pos@pos))
    def test_nearest_clear_projection(self):
        rng=np.random.default_rng(19)
        for _ in range(100):
            poly=q.hull(rng.normal(size=(6,2)))
            poly=poly/(np.linalg.norm(poly,axis=1).max())*rng.uniform(2,18)
            center,r=b.mec(poly);p=rng.normal(0,100,2)
            point=q.nearest_clear_point(p,poly,center)
            self.assertLessEqual(np.linalg.norm(poly-point,axis=1).max(),20)
            slack=20-r-1e-5;delta=p-center;dist=b.norm(delta)
            old=p if dist<=slack else center+delta*slack/dist
            self.assertLessEqual(b.norm(point-p),b.norm(old-p)+1e-5)
    def test_fixed_location_error(self):
        env=b.ToySimulator(b.make_case(12),12,'hash')
        env.pos=np.array([100.,200.]);x=env._error(3)
        self.assertEqual(x,env._error(3));self.assertLessEqual(abs(x),b.A)
    def test_base_cover_immutable(self):
        before=np.array(b.COVER).copy()
        q.evaluate(0,'combined')
        np.testing.assert_array_equal(np.array(b.COVER),before)
    def test_regression_near_circle_boundary(self):
        # Previously exposed a numerically inward cut when a retained vertex
        # differed from the 999.999 m exclusion circle by floating-point noise.
        env=b.ToySimulator(b.make_case(1041),1041,'hash')
        agent=AuditedAgent(env,**q.CONFIGS['combined']);agent.run()
        self.assertFalse(env._live);self.assertEqual(env.failed,0)
    def test_audited_boundary_and_adversarial_error(self):
        for noise in ('plus','minus','smooth','alternating'):
            env=b.ToySimulator(b.make_case(30021,True),30021,noise)
            agent=AuditedAgent(env,**q.CONFIGS['combined']);agent.run()
            self.assertFalse(env._live);self.assertEqual(env.failed,0)
    def test_colocated_strong_signals(self):
        sources=[b.Source(c,np.array([0.,0.]) if c<=5 else np.array([999.,0.]),1000.) for c in range(1,11)]
        row,_=q.evaluate(90000,'combined','plus',sources=sources)
        self.assertEqual(row['cleared'],10);self.assertEqual(row['failed_clears'],0)
    def test_backend_does_not_expose_truth(self):
        env=b.ToySimulator(b.make_case(4),4)
        class BackendOnly:
            @property
            def pos(self):return env.pos
            @property
            def channel(self):return env.channel
            def move(self,p):return env.move(p)
            def detect(self,c):return env.detect(c)
            def clear(self,c):return env.clear(c)
        q.OptimizedAgent(BackendOnly(),**q.CONFIGS['combined']).run()
        self.assertFalse(env._live)

if __name__=='__main__':unittest.main(verbosity=2)
