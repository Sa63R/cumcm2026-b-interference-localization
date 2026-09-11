"""Pure geometry certificates; no policy, scenario runner, network or database."""
import copy
from fractions import Fraction as F
import math

import pytest

from planning.joint_visibility_region import joint_visibility_outer
from experiments.audit_q4_joint_visibility import (
    audit_joint_visibility_certificate, convex_polygon, exact_intersection_points,
    intersection, orientation_outer, contains_intervals, TAU)


def rect(a=-2.,b=-2.,c=2.,d=2.):
    return ((a,b),(c,b),(c,d),(a,d))


@pytest.fixture(scope='module')
def refined():
    _,e=joint_visibility_outer(rect(990.,-.1,1015.,.1),[(0.,-100.),(0.,100.)],[(1005.,0.)])
    assert e['status']=='outer_refined'
    return e


@pytest.fixture(scope='module')
def omni():
    _,e=joint_visibility_outer(rect(),[(500.,0.),(-250.,400.),(-250.,-400.)],[(1400.,0.)])
    assert e['status']=='unchanged'
    return e


def test_full_refined_certificate_and_unchanged_input(refined):
    before=copy.deepcopy(refined)
    result=audit_joint_visibility_certificate(refined)
    assert result['cells']==256 and result['deleted_cells']==88 and result['retained_cells']==168
    assert refined==before


def test_no_forced_negative_retains_omnidirectional_branch(omni):
    result=audit_joint_visibility_certificate(omni)
    assert result['deleted_cells']==0 and result['retained_cells']==256
    bad=copy.deepcopy(omni);bad['cells'][0].update(removed=True,forced_negative_indices=[0])
    with pytest.raises(ValueError,match='forced-negative'):audit_joint_visibility_certificate(bad)


@pytest.mark.parametrize('bad', ['missing_cell','duplicate_cell','bbox','edge','delete_retained','omit_forced',
                                'shrink_hull','roundbox','fake_outside','radius','negative_upper','angles','intersection'])
def test_tampered_whole_cell_and_final_hull_evidence_rejected(refined,bad):
    e=copy.deepcopy(refined)
    kept=next(c for c in e['cells'] if c['retained_intersection_boxes'])
    removed=next(c for c in e['cells'] if c['removed'])
    if bad=='missing_cell':e['cells'].pop()
    elif bad=='duplicate_cell':e['cells'][1]=copy.deepcopy(e['cells'][0])
    elif bad=='bbox':e['bbox'][0]+=.1
    elif bad=='edge':e['x_edges'][3]=e['x_edges'][2]
    elif bad=='delete_retained':kept.update(removed=True,retained_intersection_boxes=[])
    elif bad=='omit_forced':removed['forced_negative_indices']=[]
    elif bad=='shrink_hull':
        v=e['output_vertices'];cx=sum(p[0] for p in v)/len(v);cy=sum(p[1] for p in v)/len(v)
        e['output_vertices']=[[(p[0]+cx)/2.,(p[1]+cy)/2.] for p in v]
    elif bad=='roundbox':kept['retained_intersection_boxes']=[kept['retained_intersection_boxes'][0]]
    elif bad=='fake_outside':kept.update(reason='outside_canonical_region_exact',retained_intersection_boxes=[])
    elif bad=='radius':removed['R_min_lower_m']+=100.
    elif bad=='negative_upper':removed['negative_max_distances_upper_m'][0]=0.
    elif bad=='angles':kept['constraints'][0]['allowed_intervals']=[[0.,0.]]
    else:kept['orientation_intersection']=[]
    with pytest.raises(ValueError):audit_joint_visibility_certificate(e)


def test_exact_intersections_include_nonbinary_edge_points_and_closed_touch():
    p=convex_polygon(((0.,0.),(3.,1.),(0.,2.)))
    result=exact_intersection_points(p,(.5,0.,1.,2.))
    assert (F(1,2),F(1,6)) in result and (F(1),F(1,3)) in result
    assert exact_intersection_points(p,(3.,1.,4.,2.))=={(F(3),F(1))}


def test_redundant_collinear_original_vertex_is_not_required_as_roundbox():
    canonical=((-2.,-2.),(0.,-2.),(2.,-2.),(2.,2.),(-2.,2.))
    _,e=joint_visibility_outer(canonical,[(20.,0.)],[(3000.,0.)])
    assert audit_joint_visibility_certificate(e)['deleted_cells']==0


def test_clockwise_input_and_convex_hull_filling_are_safe():
    _,e=joint_visibility_outer(rect(990.,-1.,1015.,1.)[::-1],[(0.,-100.),(0.,100.)],[(1005.,0.)])
    assert e['deleted_intersecting_cells']>0 and not e['old_vertices_excluded']
    assert audit_joint_visibility_certificate(e)['passed']


def test_closed_singletons_cross_zero_and_small_numeric_gap_retained():
    shared=intersection(((0.,math.pi),),((math.pi,TAU),))
    assert contains_intervals(shared,((math.pi,math.pi),))
    assert contains_intervals(intersection(((0.,.1),),((6.,TAU),)),((0.,0.),))
    assert intersection(((0.,1.),),((1.+1e-13,2.),))
    assert not intersection(((0.,1.),),((1.+1e-6,2.),))
    arc,_,_=orientation_outer((100.,0.),(-1.,-1.,1.,1.),True,1e-7)
    assert contains_intervals(arc,((0.,0.),(TAU-.001,TAU)))
    assert not contains_intervals(arc,((math.pi,math.pi),))


@pytest.mark.parametrize('observer', [(0.,0.),(1.,0.),(1.,1.),(1.+1e-9,0.)])
def test_inside_or_near_zero_observer_requires_full_circle(observer):
    for positive in (True,False):
        result,_,_=orientation_outer(observer,(-1.,-1.,1.,1.),positive,1e-7)
        assert result==((0.,TAU),)


def test_produced_inside_observer_and_engine_edge_tolerance_certificates():
    for c,p,n in [(rect(-1.,-1.,1.,1.),[(0.,0.)],[(.5,0.)]),
                  (rect(-.01,-.01,.01,.01),[(-5e-10,1000.)],[(-500.,0.)])]:
        _,e=joint_visibility_outer(c,p,n)
        assert audit_joint_visibility_certificate(e)['passed']


def test_actual_union_must_cover_all_location_corners_not_only_centre():
    intervals,_,_=orientation_outer((2.,0.),(-1.,-1.,1.,1.),True,1e-7)
    # A centre-only positive half-circle misses orientations permitted at the
    # upper/lower corners, even though their source positions share this cell.
    centre_only=((0.,math.pi/2.),(3.*math.pi/2.,TAU))
    assert not contains_intervals(centre_only,intervals)
    for location in ((-1.,-1.),(1.,-1.),(1.,1.),(-1.,1.),(0.,0.)):
        for k in range(72):
            theta=k*TAU/72.
            dot=(2.-location[0])*math.cos(theta)-location[1]*math.sin(theta)
            if dot>=0.:
                assert contains_intervals(intervals,((theta,theta),))


@pytest.mark.parametrize('canonical,positive,negative', [
    ([(0.,0.)],[(1.,0.)],[]),
    (rect(),[],[(1.,0.)]),
    (rect(),[(1.,0.)]*257,[]),
    (rect(-.01,-.01,.01,.01),[(100.,0.),(-50.,86.),(-50.,-86.)],[(500.,0.)])])
def test_fallback_preserves_original_even_if_partial_grid_exists(canonical,positive,negative):
    _,e=joint_visibility_outer(canonical,positive,negative)
    assert e['status']=='fallback' and audit_joint_visibility_certificate(e)['fallback']
    bad=copy.deepcopy(e);bad['output_vertices'][0][0]+=.01
    with pytest.raises(ValueError,match='Fallback changed'):audit_joint_visibility_certificate(bad)


def test_no_planner_reentry_and_no_hidden_fields(refined,monkeypatch):
    import planning.joint_visibility_region as planner
    monkeypatch.setattr(planner,'joint_visibility_outer',lambda *a,**k:pytest.fail('Audit reentered planner'))
    monkeypatch.setattr(planner,'_clip_box_exact',lambda *a,**k:pytest.fail('Audit reused production clip'))
    monkeypatch.setattr(planner,'_allowed_orientation',lambda *a,**k:pytest.fail('Audit reused angular proof'))
    class Guard(dict):
        def __getitem__(self,key):
            assert key not in {'truth','ground_truth','source','scenario','client'}
            return super().__getitem__(key)
    assert audit_joint_visibility_certificate(Guard(refined))['passed']
