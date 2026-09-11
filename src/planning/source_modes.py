"""Observation-only nominal single-source service cost and exit prediction.

This is equal-weight quadrature over 3 (or 5) feasible, inward geometry nodes,
not a posterior distribution, an executable plan, or a Q3 value bound. Each
node follows a *complete* zero-error, centidegree-rounded nominal continuation
on independent continuous outer polygons. No finite-node singleton authorizes
clearance: only near feedback or the ordinary enclosing-disk certificate does.

The first probe is compulsory. Its arrival travel is excluded; a fixed one
second entry channel-switch surrogate is included. Later measurements use the
same channel. No cross-source information, inference, optical search fallback,
or value of future coverage is credited here. An unfinished support makes the
whole mode invalid, rather than receiving a fabricated cheap terminal cost.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from localization import CandidateRegion
from planning.probe_candidates import geometry_candidates
from planning.radius_probe import choose_radius_probe
from simulator_client.state import Position


@dataclass(frozen=True)
class NominalAction:
    action: str
    position: Position
    result: str
    bearing_deg: float | None
    movement_s: float
    measurement_s: float
    switching_s: float
    optical_s: float
    removal_s: float

    @property
    def cost_s(self):
        return (self.movement_s + self.measurement_s + self.switching_s
                + self.optical_s + self.removal_s)


@dataclass(frozen=True)
class SourceModeSupport:
    source_position: Position
    radius_interval_m: tuple[float, float]
    cost_s: float
    exit_position: Position | None
    measurements: int
    valid: bool
    reason: str
    actions: tuple[NominalAction, ...]


@dataclass(frozen=True)
class SourceModePrediction:
    cost_s: float
    exit_position: Position | None
    hypotheses: int
    measurements_mean: float
    valid: bool
    reason: str
    support_results: tuple[SourceModeSupport, ...] = ()


def _invalid(reason, results=()):
    return SourceModePrediction(math.inf, None, len(results),
                                sum(r.measurements for r in results) / len(results)
                                if results else 0., False, reason, tuple(results))


def _key(position):
    return round(position.x, 6), round(position.y, 6)


def _compatible_radius_interval(region, source):
    """One R fits every sign; positive endpoints closed, negative upper open.

    The interval endpoints alone do not encode openness. The returned interval
    is a compatibility witness for scoring; no R is sampled and it is never
    a probability weight. A collapsed interval is retained only if all strict
    negative constraints still hold (e.g. R=1500 with no conflicting silence).
    """
    if not region.contains((source.x, source.y)):
        return None
    if math.hypot(source.x, source.y) > min(1800., region.prior_radius):
        return None
    lower = 1000.
    for observation in region.observations:
        observer = Position.coerce(observation.position)
        distance = source.distance_to(observer)
        if distance <= 5:
            return None  # A direction report excludes near feedback.
        angle = math.degrees(math.atan2(source.y-observer.y, source.x-observer.x)) % 360
        error = abs((angle-observation.bearing_deg+180) % 360-180)
        if error > observation.error_deg + 1e-8:
            return None
        lower = max(lower, distance)
    negatives = [source.distance_to(Position.coerce(p))
                 for p in getattr(region, "no_signal_positions", ())]
    upper = min([1500., float(region.reception_radius)] + negatives)
    if lower > upper or any(lower >= distance for distance in negatives):
        return None
    return lower, upper


def _supports(region, count):
    center = Position.coerce(region.enclosing_disk().center)
    vertices = tuple(Position.coerce(p) for p in region.vertices)
    if count == 3:
        ends = max(((a, b) for a in vertices for b in vertices),
                   key=lambda pair: pair[0].distance_to(pair[1]))
    else:
        ends = tuple(vertices[i*len(vertices)//min(4, len(vertices))]
                     for i in range(min(4, len(vertices))))
    candidates = (center,) + tuple(Position(.75*p.x+.25*center.x,
                                           .75*p.y+.25*center.y) for p in ends)
    seen, result = set(), []
    for source in candidates:
        if (source.x, source.y) in seen:
            continue
        seen.add((source.x, source.y))
        interval = _compatible_radius_interval(region, source)
        if interval is not None:
            result.append((source, interval))
    return tuple(result)


def _guaranteed(region, position):
    return bool(region.vertices) and all(
        position.distance_to(Position.coerce(v)) <= 1000. for v in region.vertices)


def _safe_exit(region, current):
    circle = region.enclosing_disk()
    if circle.radius > 19.9:
        return None
    center = Position.coerce(circle.center)
    distance = center.distance_to(current)
    fraction = min(1., max(0., 19.9-circle.radius)/distance) if distance else 0.
    return Position(center.x + fraction*(current.x-center.x),
                    center.y + fraction*(current.y-center.y))


def _continue(region, source, interval, first_probe, bearing, observed, limit, weight):
    # Even enclosing_disk() fills a cache: *all* work happens on this copy.
    region = region.copy()
    current, destination = first_probe, first_probe
    actions = []
    for step in range(limit):
        if not _guaranteed(region, destination):
            reason = "continuation_not_guaranteed_reception"
            break
        travel = current.distance_to(destination)/5 if step else 0.
        current = destination
        observed = observed | {_key(current)}
        near = source.distance_to(current) <= 5.
        nominal_bearing = None if near else round(math.degrees(math.atan2(
            source.y-current.y, source.x-current.x)) % 360, 2) % 360
        actions.append(NominalAction("measure", current, "near" if near else "direction",
                                     nominal_bearing, travel, 5., 1. if not step else 0., 0., 0.))
        if near:
            endpoint = current
        else:
            region.observe(current, nominal_bearing)
            if not region.vertices or not region.contains((source.x, source.y)):
                reason = "nominal_observation_inconsistent"
                break
            endpoint = _safe_exit(region, current)
        if endpoint is not None:
            actions.append(NominalAction("clear", endpoint, "success", None,
                                         current.distance_to(endpoint)/5, 0., 0., 3., 2.))
            return SourceModeSupport(source, interval, sum(a.cost_s for a in actions),
                                     endpoint, step+1, True, "near" if near else "enclosing_disk",
                                     tuple(actions))
        if step+1 == limit:
            reason = "probe_budget_exhausted_without_certificate"
            break
        extras, _ = geometry_candidates(region, mode="axis_quantile", old_best=None)
        destination, _ = choose_radius_probe(region, current, bearing, observed,
                                              weight, extra_points=extras)
        if destination is None:
            reason = "no_fresh_guaranteed_continuation"
            break
    return SourceModeSupport(source, interval, math.inf, None,
                             sum(a.action == "measure" for a in actions),
                             False, reason, tuple(actions))


def predict_source_mode(region, first_probe, first_bearing_deg,
                        observed_positions=frozenset(), max_probes=6,
                        uncertainty_weight=.5, *, supports=3):
    """Predict nominal complete local cost and mean *clearing* endpoint.

    ``max_probes`` includes the supplied first probe. ``supports`` is 3 or 5;
    after exact sign/radius and bearing checks fewer nodes may survive, all
    equally weighted. ``hypotheses`` reports that actual retained count.
    The mean endpoint is a scheduling representative, never a clearance or
    reception certificate and generally not a point visited by any support.
    Invalid input/state returns ``valid=False, cost_s=inf, exit_position=None``.
    Programming/configuration errors in scalar limits raise ValueError.
    """
    if type(max_probes) is not int or not 0 <= max_probes <= 30:
        raise ValueError("max_probes must be integer in [0,30]")
    if type(supports) is not int or supports not in (3, 5):
        raise ValueError("supports must be 3 or 5")
    if (isinstance(uncertainty_weight, bool) or not isinstance(uncertainty_weight, (int, float))
            or not math.isfinite(uncertainty_weight) or uncertainty_weight < 0):
        raise ValueError("uncertainty_weight must be nonnegative and finite")
    if not isinstance(region, CandidateRegion):
        return _invalid("unsupported_region")
    if not region.vertices:
        return _invalid("empty_region")
    if not region.observations:
        return _invalid("positive_bearing_history_required")
    if not max_probes:
        return _invalid("no_probe_budget")
    try:
        first_probe = Position.coerce(first_probe)
        if isinstance(first_bearing_deg, bool) or not math.isfinite(first_bearing_deg):
            return _invalid("invalid_first_bearing")
        observed = frozenset(_key(Position.coerce(p)) for p in observed_positions)
        observed |= frozenset(_key(Position.coerce(o.position)) for o in region.observations)
        observed |= frozenset(_key(Position.coerce(p))
                              for p in getattr(region, "no_signal_positions", ()))
        # The live MEC cache, lists and negative set must remain unchanged.
        snapshot = region.copy()
        if not _guaranteed(snapshot, first_probe):
            return _invalid("first_probe_not_guaranteed_reception")
        if _key(first_probe) in observed:
            return _invalid("first_probe_already_observed")
        nodes = _supports(snapshot, supports)
    except (ValueError, TypeError, OverflowError):
        return _invalid("invalid_observation_geometry")
    if not nodes:
        return _invalid("no_radius_and_history_compatible_support")
    results = tuple(_continue(snapshot, source, interval, first_probe,
                              first_bearing_deg, observed, max_probes, uncertainty_weight)
                    for source, interval in nodes)
    if any(not result.valid for result in results):
        reasons = ",".join(sorted({r.reason for r in results if not r.valid}))
        return _invalid("unfinished_nominal_support:" + reasons, results)
    return SourceModePrediction(sum(r.cost_s for r in results)/len(results),
                                Position(sum(r.exit_position.x for r in results)/len(results),
                                         sum(r.exit_position.y for r in results)/len(results)),
                                len(results), sum(r.measurements for r in results)/len(results),
                                True, "equal_weight_feasible_geometry_nodes_zero_error_not_posterior",
                                results)


__all__ = ["SourceModePrediction", "SourceModeSupport", "NominalAction", "predict_source_mode"]
