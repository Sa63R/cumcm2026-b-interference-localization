"""Pure synthetic optimizer/accounting checks, no coverage-search invocation."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('cover21', ROOT/'experiments/research_q4_continuous_cover21.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_optimizer_quadratic_and_bound_are_real_objectives():
    target = np.array([.2, .7, 1.2])
    point, log = m.nelder_mead(lambda x: float(np.sum((x-target)**2)), [.4, .4, .4], .1, 1500)
    assert point == pytest.approx([.2, .7, 1.], abs=2e-6)
    assert log['evaluations'] <= 1500
    assert all(0. <= x <= 1. for x in point)


def test_joint_margin_requires_same_station_for_range_and_orientation():
    compiled = (np.array([[0., 0.]]), np.array([[1., 0.]]))
    # One station is in front but out of range; the other in range but behind.
    assert m.joint_margin([[1100., 0.], [-50., 0.]], compiled) == -50.
    assert m.joint_margin([[900., 0.], [-50., 0.]], compiled) == 100.


def test_validated_counterexample_rejects_a_covering_station():
    witness = dict(source=[0., 0.], normal=[1., 0.])
    assert m.independent_counter([[-10., 0.], [1100., 0.]], witness)['passed']
    with pytest.raises(ValueError, match='every station'):
        m.independent_counter([[-10., 0.], [900., 0.]], witness)


def test_unresolved_constraints_are_not_counterexamples():
    w = m.WorkingSet()
    w.absorb(dict(counterexample=None, unresolved_cell=dict(box=[-1., 1., -1., 1.])),
             [[0., 10.], [-10., -10.], [10., -10.]], 'synthetic-unknown')
    assert len(w.positions) == 5
    assert not any(r['evidence_kind']=='verified_strict_counterexample' for r in w.rows)
    before = len(w.rows)
    w.refresh_normals([[0., 10.], [-10., -10.], [10., -10.]])
    assert len(w.rows) == before


def test_exact_frozen_input_and_layout_order_identity():
    pts, old, _ = m.identities()
    assert len(pts) == 21 and pts[0] == [0., 0.]
    values = np.array([999., old['outer_radius_m'], old['outer_phase_deg']])
    bounds = np.array([[970., 1020.], [1850., 1900.], [10., 20.]])
    built, _ = m.build_points((values-bounds[:, 0])/(bounds[:, 1]-bounds[:, 0]),
                             bounds, old['native_station_route_ids'], [])
    assert np.max(np.abs(np.array(pts)-np.array(built))) < 1e-9


def test_local_coordinates_preserve_origin_and_point_count():
    bounds = np.array([[999., 1000.], [1858., 1859.], [13., 14.], [-10., 10.], [-10., 10.]])
    pts, values = m.build_points(np.array([0., 0., 0., 1., 0.]), bounds, list(range(21)), [1])
    assert len(pts) == 21 and pts[0] == [0., 0.]
    assert pts[1] == pytest.approx([1009., -10.])
