"""Analytic geometry and finite inputs; no scenario or network."""
import math
import random

import pytest

from planning.certified_clear_region import choose_certified_clear_point, POLYGON_RADIUS, NEAR_RADIUS
from simulator_client.state import Position


def test_full_vertex_intersection_is_larger_than_old_mec_inball():
    p,log=choose_certified_clear_point((0,0),(0,-100),vertices=[(-19.8,0),(19.8,0)])
    expected=math.sqrt(POLYGON_RADIUS**2-19.8**2)
    assert p.x == pytest.approx(0,abs=1e-12) and p.y == pytest.approx(-expected,abs=1e-10)
    assert -p.y>2.8  # The old MEC-inball radius would be only 0.2 m.
    assert log['method']=='boundary_candidates' and log['certificate']['passed']


def test_single_vertex_radial_projection_is_analytic_nearest_point():
    p,log=choose_certified_clear_point((0,0),(30,40),vertices=[(0,0)])
    assert (p.x,p.y)==pytest.approx((POLYGON_RADIUS*.6,POLYGON_RADIUS*.8))
    assert log['selected_incoming_m']==pytest.approx(50-POLYGON_RADIUS)


@pytest.mark.parametrize('current',[(0,0),(3,4),(-14,0)])
def test_current_position_already_in_near_service_disk_needs_no_movement(current):
    p,log=choose_certified_clear_point((0,0),current,near_point=(0,0))
    assert p==Position(*current) and log['selected_incoming_m']==0
    assert log['certificate']['verification_radius_m']==14.99999


def test_near_disk_reserves_full_five_meter_source_uncertainty():
    p,log=choose_certified_clear_point((10,20),(110,20),near_point=(10,20))
    assert p.x==pytest.approx(10+NEAR_RADIUS)
    assert p.distance_to(Position(5,20))<19.99999
    assert log['certificate']['kind']=='near_disk'


def test_anchored_search_reduces_two_segment_cost_without_incoming_increase():
    current,anchor=Position(-100,0),Position(100,50)
    p,log=choose_certified_clear_point((0,0),current,vertices=[(0,0)],anchor=anchor,config='anchored')
    assert p.y>0 and current.distance_to(p)<=100
    assert current.distance_to(p)+p.distance_to(anchor)<100+math.hypot(100,50)
    assert log['method']=='finite_rays' and 64<=log['ray_count']<=66
    assert log['candidate_count']<=199


def test_anchor_absent_degrades_to_same_incoming_point():
    kwargs={'vertices':[(-12,0),(12,0)]}
    a,la=choose_certified_clear_point((0,0),(30,-20),**kwargs)
    b,lb=choose_certified_clear_point((0,0),(30,-20),config='anchored',**kwargs)
    assert a==b and la['method']==lb['method']=='boundary_candidates'


def test_anchor_on_original_through_segment_can_keep_original():
    p,log=choose_certified_clear_point((0,0),(0,0),vertices=[(-3,0),(3,0)],anchor=(10,30),config='anchored')
    assert p==Position(0,0) and log['proxy_saved_s']==0


@pytest.mark.parametrize('kwargs',[{}, {'vertices':[]},{'vertices':[(float('nan'),0)]},
    {'vertices':[(25,0)]},{'vertices':[(0,0)]*65},{'vertices':[(0,0)],'near_point':(0,0)},
    {'near_point':(0,0),'anchor':(float('nan'),0),'config':'anchored'}])
def test_invalid_or_overbudget_geometry_falls_back_to_original(kwargs):
    p,log=choose_certified_clear_point((0,0),(100,0),**kwargs)
    assert p==Position(0,0) and log['status']=='fallback' and not log['certificate']['passed']


def test_duplicate_and_nearly_tangent_disks_are_finite():
    for vertices in ([(0,0),(0,0)],[(-POLYGON_RADIUS,0),(POLYGON_RADIUS,0)],
                     [(-19.9,0),(19.9,1e-10),(0,1e-12)]):
        p,log=choose_certified_clear_point((0,0),(0,100),vertices=vertices)
        assert math.isfinite(p.x+p.y) and log['certificate']['passed']
        assert max(math.dist((p.x,p.y),v) for v in vertices)<=19.99999


@pytest.mark.parametrize('config',['incoming','anchored'])
def test_polygon_certificate_extends_to_convex_combinations_and_full_incoming(config):
    rng=random.Random(493)
    for trial in range(20):
        vertices=[(rng.uniform(-9,9),rng.uniform(-9,9)) for _ in range(7)]
        current=(rng.uniform(-200,200),rng.uniform(-200,200));anchor=(100.,50.)
        p,log=choose_certified_clear_point((0,0),current,vertices=vertices,anchor=anchor,config=config)
        assert log['certificate']['passed']
        assert math.dist(current,(p.x,p.y))<=math.hypot(*current)
        assert log['selected_objective_m']<=log['original_objective_m']
        for _ in range(12):
            weights=[rng.random() for v in vertices];total=sum(weights)
            x=sum(w*v[0] for w,v in zip(weights,vertices))/total
            y=sum(w*v[1] for w,v in zip(weights,vertices))/total
            assert math.dist((p.x,p.y),(x,y))<20


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):choose_certified_clear_point((0,0),(0,0),config='unbounded')
