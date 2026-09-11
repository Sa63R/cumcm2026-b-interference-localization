"""Two fixed observation-oriented rings with complete directional coverage.

This reproduces GEOMETRY_PROTOCOL's angle-zero inner-CCW/outer-CW route.
No scene, feedback, NN, 2-opt or route search enters its construction.
"""
import copy
from functools import lru_cache
import hashlib
import json
import math

from .q4_directional_cover import (
    concentric_stations, certify_directional_cover, verify_directional_cover_certificate,
)


CONFIGS = {"ring_28": (9, 1920.), "ring_31": (10, 1900.)}
CERTIFICATE_BUDGET = dict(arena_radius=1800., reception_radius=1000.,
                          max_depth=16, max_cells=200000,
                          range_margin_m=1e-5, orientation_margin_m=1e-7)


def _boundary(k, rho):
    """Exact-model angular count, restricted to radial-outward boundary sources."""
    step = math.pi/k
    heading = math.acos(1800./rho)
    range_angle = math.acos((rho*rho+1800.**2-1000.**2)/(2*rho*1800.))
    alpha = min(heading, range_angle)
    ratio = 2*alpha/step
    return dict(angular_step_deg=math.degrees(step), halfwidth_deg=math.degrees(alpha),
        heading_halfwidth_deg=math.degrees(heading), range_halfwidth_deg=math.degrees(range_angle),
        receiving_arc_over_step=ratio, minimum_receiving_outer_stations=math.floor(ratio),
        angle_fraction_at_least_two=min(1., max(0., ratio-1.)),
        heading_two_station_threshold_m=1800./math.cos(step),
        range_at_one_step_m=math.sqrt(rho*rho+1800.**2-2*rho*1800.*math.cos(step)),
        min_source_station_distance_m=rho-1800.)


@lru_cache(maxsize=2)
def _observation_cover_route(config):
    k, rho = CONFIGS[config]
    native = concentric_stations(999., rho, inner_count=k, outer_count=2*k,
                                 inner_phase_deg=0., outer_phase_deg=0.)
    ids = [0]+list(range(1,k+1))+[k+1+((2*k-2-j)%(2*k)) for j in range(2*k)]
    if sorted(ids) != list(range(1+3*k)) or len(set(native)) != 1+3*k:
        raise ValueError("Ring route must visit each proposed station exactly once")
    points = tuple(native[i] for i in ids)
    length = math.fsum(a.distance_to(b) for a,b in zip(points,points[1:]))
    formula = rho+2*(k-1)*999.*math.sin(math.pi/k)+2*(2*k-1)*rho*math.sin(math.pi/(2*k))
    if not math.isfinite(length) or abs(length-formula) > 1e-7:
        raise ValueError("Fixed ring-route length disagrees with its closed form")
    full = certify_directional_cover(points, **CERTIFICATE_BUDGET)
    # An inconclusive computation is never a coverage guarantee. Replay every
    # leaf and the full partition before any client can be entered.
    verification = verify_directional_cover_certificate(points, full)
    leaf_sha = hashlib.sha256(json.dumps(full["leaves"],sort_keys=True,separators=(",",":"),
                                         allow_nan=False).encode()).hexdigest()
    boundary = _boundary(k, rho)
    if boundary["minimum_receiving_outer_stations"] < 2:
        raise ValueError("The fixed radial-boundary two-receiver property was lost")
    metadata = dict(config=config, inner_count=k, outer_count=2*k,
        inner_radius_m=999., outer_radius_m=rho, inner_phase_deg=0., outer_phase_deg=0.,
        native_station_route_ids=ids, route_length_m=length, route_closed_form_m=formula,
        pure_movement_s=length/5., station_sha256=full["station_sha256"],
        full_leaf_certificate_sha256=leaf_sha, boundary_radial_outward=boundary,
        route_scope="Fixed origin then angle-zero inner CCW and outer CW from index 2k-2; no shortest-route claim",
        boundary_scope="Only source radius 1800m, exactly radial-outward emission normal, reception radius 1000m; neither two receivers for all sources/orientations nor localization/task-time guarantee")
    certificate = {key:value for key,value in full.items() if key != "leaves"}
    certificate.update(profile="observation_"+config, route_length_m=length,
        route_kind="fixed_angle_zero_inner_ccw_outer_cw",
        full_leaf_certificate_sha256=leaf_sha, independent_leaf_verification=verification,
        proposal=dict(inner_count=k, outer_count=2*k, inner_radius_m=999., outer_radius_m=rho,
                      inner_phase_deg=0., outer_phase_deg=0.))
    return points, certificate, metadata


def observation_cover_route(config):
    """Return fixed points, a genuine new-cover summary and fresh route metadata.

    Complete proof construction/replay is cached per process. The summary's
    runtime_s records that construction, not the cost of each cache lookup.
    """
    if not isinstance(config,str) or config not in CONFIGS:
        raise ValueError("Only ring_28 and ring_31 are fixed production candidates")
    points, certificate, metadata = _observation_cover_route(config)
    return points, copy.deepcopy(certificate), copy.deepcopy(metadata)
