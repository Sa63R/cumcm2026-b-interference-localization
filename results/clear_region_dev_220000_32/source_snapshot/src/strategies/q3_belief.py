"""Approximate, observation-conditioned Q3 worlds for research rollouts.

Assumed prior (NOT an official distribution): N uniform on 10..16, a uniform
N-subset of the twenty channels, independent uniform-area positions in the
1800 m disk, and independent radii uniform on [1000,1500]. Bearings use a flat
1.005-degree compatibility band, an approximation to the unknown quantized
error likelihood. Identical measurements are deduplicated, not independent.

Positions are proposed uniformly in conservative geometry polygons and
weighted by the length of their feasible radius interval. Negative-only
channel existence evidence is estimated by bounded prior Monte Carlo; the
source-count/subset sampler includes the prior 1 / choose(20,N) factor. These
finite weighted pools are an approximate posterior, never a completeness
certificate. Pool exhaustion raises BeliefSamplingError unless coverage
proves the channel cannot exist. Callers must fall back to their safe policy.

Returned Scenarios retain sources already cleared in history. Their new seed
models FUTURE errors; it does not reproduce past numerical bearings. A rollout
adapter must initialize the observed cleared state and cache/reuse historical
measurements at repeated coordinates. Geometric compatibility alone is not
an exact conditioning of the simulator's hashed error seed.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import random
import time

from geometry import HalfPlane, bearing_halfplanes, clip_polygon, disk_halfplanes, disk_polygon
from simulation.cases import Scenario, Source


class BeliefSamplingError(ValueError):
    """Invalid/inconsistent history or insufficient bounded sampling support."""


class BeliefSamplingTimeout(BeliefSamplingError):
    """The cooperative wall-clock sampling deadline was reached."""


def _check_deadline(deadline):
    if deadline is not None and time.perf_counter() >= deadline:
        raise BeliefSamplingTimeout("Belief sampling deadline exhausted; use the deterministic fallback")


@dataclass(frozen=True)
class _Event:
    action: str
    position: tuple[float, float]
    result: str
    bearing: float | None = None


@dataclass(frozen=True)
class _Pool:
    # A None pool represents the exact unconditional source prior.
    candidates: tuple[tuple[float, float, float, float], ...] | None
    weights: tuple[float, ...]
    evidence: float


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise BeliefSamplingError(f"{name} must be a finite number")
    return float(value)


def _history_by_channel(history, deadline=None):
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        raise BeliefSamplingError("history must be a sequence of action records")
    events = {channel: [] for channel in range(1, 21)}
    mandatory, cleared, measured = set(), set(), {}
    for index, record in enumerate(history):
        if index % 32 == 0:
            _check_deadline(deadline)
        if not isinstance(record, Mapping):
            raise BeliefSamplingError("Each history record must be a mapping")
        action, channel = record.get("action"), record.get("channel")
        if action not in ("measure", "clear"):
            raise BeliefSamplingError("History may contain only measure/clear actions")
        if isinstance(channel, bool) or not isinstance(channel, int) or not 1 <= channel <= 20:
            raise BeliefSamplingError("History channel must be an integer in 1..20")
        position = record.get("position")
        if not isinstance(position, (tuple, list)) or len(position) != 2:
            raise BeliefSamplingError("History position needs two coordinates")
        position = (_number(position[0], "x"), _number(position[1], "y"))
        result = record.get("result")
        allowed = ("no_signal", "near", "direction") if action == "measure" else (
            "success", "no_target_in_range")
        if result not in allowed:
            raise BeliefSamplingError("Unknown observation result")
        bearing = None
        if result == "direction":
            bearing = _number(record.get("bearing_deg"), "bearing_deg")
            if not 0 <= bearing < 360:
                raise BeliefSamplingError("bearing_deg must be in [0,360)")
        if channel in cleared:
            expected = "no_signal" if action == "measure" else "no_target_in_range"
            if result != expected:
                raise BeliefSamplingError("A successfully cleared channel cannot emit or clear again")
            continue  # Silence after removal says nothing about original position/R.
        event = _Event(action, position, result, bearing)
        if action == "measure":
            key = (channel, position)
            if key in measured:
                if measured[key] != event:
                    raise BeliefSamplingError("Repeated fixed-coordinate measurements disagree")
                continue
            measured[key] = event
        if event not in events[channel]:
            events[channel].append(event)
        if result in ("direction", "near", "success"):
            mandatory.add(channel)
        if result == "success":
            cleared.add(channel)
    if len(mandatory) > 16:
        raise BeliefSamplingError("Observations require more than sixteen sources")
    return {channel: tuple(items) for channel, items in events.items()}, mandatory


def _radius_interval(position, events):
    if math.hypot(*position) > 1800.0:
        return None
    lower, upper = 1000.0, 1500.0
    for event in events:
        distance = math.dist(position, event.position)
        if event.result == "direction":
            if distance <= 5.0:
                return None
            angle = math.degrees(math.atan2(position[1] - event.position[1],
                                           position[0] - event.position[0])) % 360
            if abs((angle - event.bearing + 180) % 360 - 180) > 1.005 + 1e-10:
                return None
            lower = max(lower, distance)
        elif event.result == "near":
            if distance > 5.0:
                return None
            lower = max(lower, distance)
        elif event.result == "no_signal":
            upper = min(upper, distance)
        elif event.result == "success":
            if distance > 20.0:
                return None
        elif distance <= 20.0:  # failed /clear of an existing source
            return None
    return (lower, upper) if upper > lower else None


def _proposal_polygon(events, deadline=None):
    _check_deadline(deadline)
    vertices = disk_polygon((0.0, 0.0), 1800.0, 32, outer=True)
    positive, negative = [], []
    for event in events:
        constraints = ()
        if event.result == "direction":
            constraints = bearing_halfplanes(event.position, event.bearing, 1.005)
            constraints += disk_halfplanes(event.position, 1500.0, 32, outer=True)
            positive.append(event.position)
        elif event.result in ("near", "success"):
            radius = 5.0 if event.result == "near" else 20.0
            constraints = disk_halfplanes(event.position, radius, 32, outer=True)
            positive.append(event.position)
        elif event.result == "no_signal":
            negative.append(event.position)
        for constraint in constraints:
            _check_deadline(deadline)
            vertices = clip_polygon(vertices, constraint)
    # Every positive point is within R, including successful clear points
    # (20 m < the minimum R). These linear comparisons tighten the proposal.
    for p in positive:
        for n in negative:
            _check_deadline(deadline)
            dx, dy = n[0] - p[0], n[1] - p[1]
            length = math.hypot(dx, dy)
            if length == 0:
                return ()
            nx, ny = dx / length, dy / length
            midpoint = (p[0] + dx / 2, p[1] + dy / 2)
            vertices = clip_polygon(vertices, HalfPlane(nx, ny, nx * midpoint[0] + ny * midpoint[1]))
    return vertices


def _polygon_sampler(vertices, rng, deadline=None):
    triangles, cumulative, total = [], [], 0.0
    if len(vertices) >= 3:
        a = vertices[0]
        for index, (b, c) in enumerate(zip(vertices[1:-1], vertices[2:])):
            if index % 32 == 0:
                _check_deadline(deadline)
            area = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2
            if area > 0:
                total += area
                cumulative.append(total)
                triangles.append((a, b, c))
    if not total:
        raise BeliefSamplingError("No positive-area source proposal; use the deterministic fallback")

    def sample():
        a, b, c = triangles[bisect_left(cumulative, rng.random() * total)]
        u, v = math.sqrt(rng.random()), rng.random()
        return (a[0] + u * ((1 - v) * (b[0] - a[0]) + v * (c[0] - a[0])),
                a[1] + u * ((1 - v) * (b[1] - a[1]) + v * (c[1] - a[1])))

    return sample


def _prior_position(rng):
    angle, radius = rng.uniform(0, 2 * math.pi), 1800 * math.sqrt(rng.random())
    return radius * math.cos(angle), radius * math.sin(angle)


def _absence_proven(events, deadline=None):
    """Sufficient rectangle-cover proof, never a sample-based absence claim."""
    disks = [(event.position, 1000.0 if event.result == "no_signal" else 20.0)
             for event in events if event.result in ("no_signal", "no_target_in_range")]
    stack = [(-1800.0, -1800.0, 1800.0, 1800.0, 0)]
    visited = 0
    while stack:
        if visited % 32 == 0:
            _check_deadline(deadline)
        x0, y0, x1, y1, depth = stack.pop()
        visited += 1
        if visited > 20_000:
            return False
        nearest_x = max(x0, min(0.0, x1))
        nearest_y = max(y0, min(0.0, y1))
        if math.hypot(nearest_x, nearest_y) > 1800.0:
            continue
        corners = ((x0, y0), (x0, y1), (x1, y0), (x1, y1))
        if any(all(math.dist(center, corner) <= radius - 1e-7 for corner in corners)
               for center, radius in disks):
            continue
        if depth == 10:
            return False
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        stack.extend(((x0, y0, xm, ym, depth + 1), (xm, y0, x1, ym, depth + 1),
                      (x0, ym, xm, y1, depth + 1), (xm, ym, x1, y1, depth + 1)))
    return True


def _pool(events, required, rng, deadline=None):
    _check_deadline(deadline)
    if not events:
        return _Pool(None, (), 1.0)
    sample = (_polygon_sampler(_proposal_polygon(events, deadline), rng, deadline)
              if required else lambda: _prior_position(rng))
    candidates, weights = [], []
    trials = 0
    # A bounded larger second batch avoids abandoning narrow feasible regions
    # immediately. Evidence uses every attempted point, including rejections.
    for batch in (256 if required else 512, 3584):
        for _ in range(batch):
            if trials % 32 == 0:
                _check_deadline(deadline)
            position = sample()
            interval = _radius_interval(position, events)
            trials += 1
            if interval is not None:
                lower, upper = interval
                candidates.append((*position, lower, upper))
                weights.append((upper - lower) / 500.0)
        if candidates:
            break
        if not required and _absence_proven(events, deadline):
            return _Pool((), (), 0.0)
    if not candidates:
        raise BeliefSamplingError("Finite proposal pool exhausted; absence is not established")
    return _Pool(tuple(candidates), tuple(weights), sum(weights) / trials)


def _draw_source(channel, pool, rng):
    if pool.candidates is None:
        x, y = _prior_position(rng)
        return Source(channel, x, y, rng.uniform(1000.0, 1500.0))
    x, y, lower, upper = rng.choices(pool.candidates, weights=pool.weights, k=1)[0]
    # no_signal requires R < distance; never draw the upper endpoint.
    radius = min(math.nextafter(upper, lower), lower + rng.random() * (upper - lower))
    return Source(channel, x, y, radius)


def sample_worlds(history: Sequence[dict], *, count: int, seed: int,
                  deadline: float | None = None) -> list[Scenario]:
    """Sample approximate Q3 worlds; fail safely without weakening observations.

    Histories have SearchResult.action_history's schema. No simulator state or
    hidden source data is accepted. See the module documentation for the prior,
    finite-pool approximation, radius weighting and historical-error cache.
    ``deadline`` is an absolute time.perf_counter() deadline; cancellation
    raises BeliefSamplingTimeout rather than returning a partial world list.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise BeliefSamplingError("count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise BeliefSamplingError("seed must be an integer")
    if deadline is not None:
        deadline = _number(deadline, "deadline")
    _check_deadline(deadline)
    histories, mandatory = _history_by_channel(history, deadline)
    rng, pools, cache = random.Random(seed), {}, {}
    for channel, events in histories.items():
        _check_deadline(deadline)
        if len(mandatory) == 16 and channel not in mandatory:
            pools[channel] = _Pool((), (), 0.0)
            continue
        key = (events, channel in mandatory)
        if key not in cache:
            cache[key] = _pool(events, channel in mandatory, rng, deadline)
        pools[channel] = cache[key]
    optional = sorted(set(range(1, 21)) - mandatory)
    size, known = len(optional), len(mandatory)
    # suffix[i][j] is the elementary symmetric sum of order j of q[i:].
    suffix = [[0.0] * (size + 1) for _ in range(size + 1)]
    suffix[size][0] = 1.0
    for i in range(size - 1, -1, -1):
        suffix[i][0] = 1.0
        q = pools[optional[i]].evidence
        for j in range(1, size - i + 1):
            suffix[i][j] = suffix[i + 1][j] + q * suffix[i + 1][j - 1]
    totals = list(range(max(10, known), 17))
    weights = [suffix[0][n - known] / math.comb(20, n) for n in totals]
    if not sum(weights) > 0:
        raise BeliefSamplingError("No sampled channel set satisfies the public 10..16 source bound")
    worlds = []
    for index in range(count):
        _check_deadline(deadline)
        total = rng.choices(totals, weights=weights, k=1)[0]
        remaining, selected = total - known, set(mandatory)
        for i, channel in enumerate(optional):
            if not remaining:
                break
            probability = pools[channel].evidence * suffix[i + 1][remaining - 1] / suffix[i][remaining]
            if rng.random() < probability:
                selected.add(channel)
                remaining -= 1
        if remaining:
            raise BeliefSamplingError("Numerically degenerate channel-subset sample")
        sources = tuple(_draw_source(channel, pools[channel], rng) for channel in sorted(selected))
        worlds.append(Scenario(
            f"q3-belief-{seed}-{index:04d}", 3, rng.randrange(2**31), sources, "uniform",
            "Approximate conditional research prior: uniform N/subset/disk/radius; "
            "finite radius-weighted pools, 1.005-degree compatibility; cache past errors",
        ))
    _check_deadline(deadline)
    return worlds


__all__ = ["BeliefSamplingError", "BeliefSamplingTimeout", "sample_worlds"]
