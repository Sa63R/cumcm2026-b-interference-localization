"""Independent truth audits are tests only; they never choose the agent's actions."""
import itertools, math, unittest
import numpy as np
import q3_base as b
import q3_optimized as o
import q3_v3 as v
from probe_score import sample_area, rank_probes
from route_search import search_route, polish
from test_optimized import inside

class AuditAgent(v.FastAgent):
    def __init__(self,sim,**kw):
        self.audit_sim=sim
        self.last_observation=None
        owner=self
        class AuditBackend(v.PublicBackend):
            def detect(self,c):
                ans=super().detect(c)
                owner.last_observation=(c,b.coord_key(self.pos),ans.kind)
                return ans
        super().__init__(AuditBackend(sim),**kw)
    def observe(self,*args,**kw):
        super().observe(*args,**kw)
        for c,t in self.tracks.items():
            truth=self.audit_sim._sources[c].xy
            assert inside(t.poly,truth),('truth exclusion',c,t.poly,truth)
            assert b.norm(t.center-truth)<=t.radius+1e-4
    def select_probe(self,c):
        point,center=super().select_probe(c)
        assert np.linalg.norm(self.tracks[c].poly-point,axis=1).max()<1000
        if center:assert b.norm(point-self.tracks[c].center)<1e-5
        return point,center
    def scan_stop(self,*args,**kw):
        super().scan_stop(*args,**kw)
        if self.unseen:
            assert o.certified_cover(self.full_scans+[self.cover[i] for i in self.remaining_cover])
    def relocate_cover_points(self):
        super().relocate_cover_points()
        if self.unseen:
            assert o.certified_cover(self.full_scans+[self.cover[i] for i in self.remaining_cover])
    def do_clear(self,c):
        strong=self.last_observation==(c,b.coord_key(self.env.pos),'strong')
        if c in self.tracks and not strong:
            assert np.linalg.norm(self.tracks[c].poly-self.env.pos,axis=1).max()<=20+1e-6
        assert b.norm(self.audit_sim._sources[c].xy-self.env.pos)<=20
        super().do_clear(c)

class V3Tests(unittest.TestCase):
    def test_startup_no_unobserved_evidence(self):
        sim=b.ToySimulator(b.make_case(0),0)
        agent=v.FastAgent(v.PublicBackend(sim));agent.scan_stop()
        self.assertEqual(sim.detects,0);self.assertEqual(len(agent.full_scans),0)
        self.assertTrue(o.certified_cover([agent.cover[i] for i in agent.remaining_cover]))
    def test_partial_startup_not_a_full_scan(self):
        sim=b.ToySimulator(b.make_case(0),0)
        agent=v.FastAgent(v.PublicBackend(sim),initial_channels=5);agent.scan_stop()
        self.assertEqual(sim.detects,5);self.assertEqual(len(agent.full_scans),0)
    def test_seven_outer_points_do_not_automatically_cover_centre(self):
        pts=[1050*np.array([math.cos(k*2*math.pi/7),math.sin(k*2*math.pi/7)]) for k in range(7)]
        self.assertFalse(o.certified_cover(pts))
        self.assertTrue(o.certified_cover(pts+[np.zeros(2)]))
    def test_quadrature_weights_and_centroid(self):
        poly=np.array([[0.,0.],[6.,0.],[0.,6.]])
        gs,ws=sample_area(poly)
        self.assertAlmostEqual(ws.sum(),1.)
        np.testing.assert_allclose((gs*ws[:,None]).sum(axis=0),[2,2])
        for g in gs:self.assertTrue(inside(poly,g))
    def test_hypothetical_scores_are_finite(self):
        poly=np.array([[0.,0.],[600.,-10.],[600.,10.]])
        scores=rank_probes(poly,np.zeros(2),np.array([[300.,0.],[300.,80.]]),2,.5,1,np.empty((0,2)),np.empty((0,2)),False)
        self.assertTrue(np.isfinite(scores).all())
    def test_route_permutation_and_square(self):
        coords=np.array([[0.,0.],[100.,0.],[100.,100.],[0.,100.]])
        route=search_route(coords,40)
        self.assertEqual(sorted(route),[0,1,2])
        path=np.vstack([coords[0],coords[route+1]])
        self.assertAlmostEqual(np.linalg.norm(np.diff(path,axis=0),axis=1).sum(),300.)
    def test_polish_never_increases_its_input_route(self):
        rng=np.random.default_rng(912)
        for n in range(3,15):
            coords=rng.normal(size=(n+1,2));d=np.linalg.norm(coords[:,None]-coords[None,:],axis=2)
            rt=rng.permutation(n)+1
            old=sum(d[a,z] for a,z in zip(np.r_[0,rt[:-1]],rt))
            rr=polish(rt.copy(),d)
            new=sum(d[a,z] for a,z in zip(np.r_[0,rr[:-1]],rr))
            self.assertEqual(sorted(rr),list(range(1,n+1)))
            self.assertLessEqual(new,old+1e-7)
    def test_force_centre_after_four_heuristic_probes(self):
        sim=b.ToySimulator(b.make_case(0),0);a=v.FastAgent(v.PublicBackend(sim))
        a.scan_stop();sim.move(np.array([999.,0.]));a.scan_stop(0)
        c=next(iter(a.tracks));a.local_steps[c]=4
        q,flag=a.select_probe(c)
        self.assertTrue(flag);np.testing.assert_array_equal(q,a.tracks[c].center)
    def test_backend_hides_sources_and_seed(self):
        sim=b.ToySimulator(b.make_case(3),3);backend=v.PublicBackend(sim)
        for attr in ('_sources','_live','seed','noise'):
            self.assertFalse(hasattr(backend,attr))
        v.FastAgent(backend).run();self.assertFalse(sim._live)
    def test_audited_random_and_boundary(self):
        for seed,stress in [(0,False),(71,False),(5011,False),(991,True)]:
            for noise in ('hash','plus','minus','smooth','alternating'):
                sim=b.ToySimulator(b.make_case(seed,stress),seed,noise)
                AuditAgent(sim).run();self.assertFalse(sim._live);self.assertEqual(sim.failed,0)
    def test_sixteen_colocated_and_count_certificate(self):
        sources=[b.Source(c,np.array([0.,0.]),1000.) for c in range(1,17)]
        sim=b.ToySimulator(sources,10001,'plus');a=AuditAgent(sim);a.run()
        self.assertEqual(len(a.cleared),16);self.assertEqual(len(a.absent),4)
        self.assertFalse(sim._live);self.assertEqual(sim.failed,0)
    def test_optical_region_through_point(self):
        class LocalBackend:
            pos=np.array([0.,0.]);channel=1
            def move(self,q):self.pos=q.copy()
            def clear(self,c):return True
        backend=LocalBackend();a=v.FastAgent(backend)
        t=b.Track(np.array([[499.,497.],[503.,497.],[503.,503.],[499.,503.]]))
        a.tracks[1]=t;a.unseen.remove(1);a.route_successor=np.array([1000.,0.])
        old=o.nearest_clear_point(backend.pos,t.poly,t.center)
        old_cost=b.norm(old)+b.norm(old-a.route_successor)
        a.go_clear(1);q=backend.pos
        self.assertLessEqual(np.linalg.norm(t.poly-q,axis=1).max(),20)
        self.assertLessEqual(b.norm(q)+b.norm(q-a.route_successor),old_cost+1e-5)
    def test_invalid_configuration(self):
        sim=b.ToySimulator(b.make_case(0),0)
        with self.assertRaises(ValueError):v.FastAgent(sim,initial_channels=21)
        with self.assertRaises(ValueError):v.make_agent(sim,'does-not-exist')
    def test_warmup_does_not_need_backend(self):
        v.warmup()

if __name__=='__main__':unittest.main(verbosity=2)
