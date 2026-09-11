"""Public negative-measurement memory; finite scores, never physical certificates.

An absent source is a separate logical possibility, not one of the weighted
existence rows. A finite bank becoming empty cannot prove channel absence.
Positive geometry remains the controller's responsibility; this bank represents
compatibility with negative observations only, not a posterior distribution.
"""
from collections import OrderedDict
from functools import lru_cache
import math
import time

import numpy as np

from simulator_client.rules import (ARENA_RADIUS_M, MIN_RECEPTION_RADIUS_M,
    MAX_RECEPTION_RADIUS_M, CHANNELS, MAX_COORDINATE_M)


VERSION = "q4-public-negative-bank-v1"
FEATURE_NAMES = (
    "negative_compatible_fraction", "candidate_hit_compatible_fraction",
    "candidate_hit_full_bank_fraction", "negative_bank_empty",
    "unique_negative_count_fraction", "nearest_negative_distance_fraction",
    "public_presence_observed", "absence_still_compatible",
)
FEATURE_DIM = len(FEATURE_NAMES)
BANK_SIZE = 3051
GRID_SPACING_M = 300.
CACHE_POINTS = 128
SCORE_CHUNK_SIZE = 32


def _point(position):
    if len(position) != 2:
        raise ValueError("a public position needs two coordinates")
    point = tuple(float(v) for v in position)
    if any(not math.isfinite(v) or abs(v) > MAX_COORDINATE_M for v in point):
        raise ValueError("public position outside protocol bounds")
    return point


def _channel(channel):
    if type(channel) is not int or channel not in CHANNELS:
        raise ValueError("public channel must be an integer in 1..20")
    return channel - 1


@lru_cache(maxsize=1)
def fixed_bank():
    """113 positions x three radii x (eight headings plus omni), no RNG/data."""
    rows = []
    for ix in range(-6, 7):
        for iy in range(-6, 7):
            x, y = ix * GRID_SPACING_M, iy * GRID_SPACING_M
            if math.hypot(x, y) > ARENA_RADIUS_M:
                continue
            for radius in (MIN_RECEPTION_RADIUS_M, 1250., MAX_RECEPTION_RADIUS_M):
                for heading in (None, *range(0, 360, 45)):
                    angle = 0. if heading is None else math.radians(heading)
                    rows.append((x, y, radius, math.cos(angle), math.sin(angle),
                                 heading is None))
    bank = np.asarray(rows, dtype=np.float64)
    assert bank.shape == (BANK_SIZE, 6)
    bank.flags.writeable = False
    return bank


def visibility(position):
    """Visibility of every fixed existence row, matching the research engine.

    Source-to-robot heading, closed radius and 180-degree sector. The 1e-12
    distance-scaled dot tolerance matches engine._visible's trig roundoff guard.
    A near point behind a directional source can still return no_signal.
    """
    x, y = _point(position)
    bank = fixed_bank()
    dx, dy = x-bank[:, 0], y-bank[:, 1]
    distance = np.hypot(dx, dy)
    dot = bank[:, 3]*dx + bank[:, 4]*dy
    return ((distance <= bank[:, 2]) &
            ((bank[:, 5] == 1.) | (dot >= -1e-12*np.maximum(1., distance))))


class NegativeObservationMemory:
    """Bounded per-channel masks updated only from accepted public replies.

    The full raw reply log is retained by the caller. This memory retains just
    the unique public facts needed for deterministic replay and query features.
    Cleared channels are ignored after actual success and must not be scored.
    """
    def __init__(self):
        self.compatible = np.ones((20, BANK_SIZE), dtype=np.bool_)
        self.negative_points = [set() for _ in CHANNELS]
        self.presence_observed = np.zeros(20, dtype=np.bool_)
        self.cleared = np.zeros(20, dtype=np.bool_)
        self.history = []
        self._visibility_cache = OrderedDict()
        self.update_wall_s = 0.
        self.score_wall_s = 0.
        self.visibility_computations = 0

    def _visible(self, point):
        result = self._visibility_cache.get(point)
        if result is None:
            result = visibility(point)
            result.flags.writeable = False
            self._visibility_cache[point] = result
            self.visibility_computations += 1
            if len(self._visibility_cache) > CACHE_POINTS:
                self._visibility_cache.popitem(last=False)
        else:
            self._visibility_cache.move_to_end(point)
        return result

    def observe(self, *, action, position, channel, result, accepted):
        """No receipt means no update; only measure/no_signal filters the bank."""
        if type(accepted) is not bool:
            raise ValueError("accepted must be an explicit boolean")
        if not accepted:
            return False
        index, point = _channel(channel), _point(position)
        if (action == "measure" and result not in ("no_signal", "near", "direction")
                or action == "clear" and result not in ("success", "no_target_in_range")
                or action not in ("measure", "clear")):
            raise ValueError("unsupported accepted public action/result")
        started = time.perf_counter()
        self.history.append(dict(action=action, position=list(point), channel=channel,
                                 result=result, accepted=True))
        changed = False
        if not self.cleared[index]:
            if action == "clear" and result == "success":
                self.cleared[index] = True
                self.presence_observed[index] = True
                changed = True
            elif action == "measure" and result in ("near", "direction"):
                changed = not bool(self.presence_observed[index])
                self.presence_observed[index] = True
            elif action == "measure" and point not in self.negative_points[index]:
                self.compatible[index] &= ~self._visible(point)
                self.negative_points[index].add(point)
                changed = True
        self.update_wall_s += time.perf_counter()-started
        return changed

    def score_candidates(self, candidates):
        """Return finite feature rows for explicit (channel, (x,y)) candidates.

        Counts are unweighted finite-library scores, not calibrated hit/absence
        probabilities. Empty existence rows produce zero hit scores plus a flag;
        they never authorize skipping a candidate or stopping the controller.
        """
        started = time.perf_counter()
        queries = [(_channel(c), _point(p)) for c, p in candidates]
        if any(self.cleared[c] for c, _ in queries):
            raise ValueError("memory scores apply only to live channels")
        output = np.zeros((len(queries), FEATURE_DIM), dtype=np.float64)
        counts = self.compatible.sum(axis=1)
        for start in range(0, len(queries), SCORE_CHUNK_SIZE):
            chunk = queries[start:start+SCORE_CHUNK_SIZE]
            channels = np.asarray([c for c, _ in chunk], dtype=np.intp)
            visible = np.stack([self._visible(p) for _, p in chunk])
            hits = np.count_nonzero(visible & self.compatible[channels], axis=1)
            alive = counts[channels]
            out = output[start:start+len(chunk)]
            out[:, 0] = alive/BANK_SIZE
            out[:, 1] = np.divide(hits, alive, out=np.zeros(len(chunk)), where=alive > 0)
            out[:, 2] = hits/BANK_SIZE
            out[:, 3] = alive == 0
            out[:, 6] = self.presence_observed[channels]
            out[:, 7] = ~self.presence_observed[channels]
            for row, (channel, point) in zip(out, chunk):
                negatives = self.negative_points[channel]
                row[4] = min(len(negatives), 512)/512.
                row[5] = (min(1., min(math.hypot(point[0]-x, point[1]-y)
                    for x, y in negatives)/3600.) if negatives else 1.)
        self.score_wall_s += time.perf_counter()-started
        return output

    def state_summary(self):
        return dict(version=VERSION, bank_size=BANK_SIZE, feature_names=list(FEATURE_NAMES),
            compatible_counts=self.compatible.sum(axis=1).tolist(),
            unique_negative_counts=[len(points) for points in self.negative_points],
            presence_observed=self.presence_observed.tolist(), cleared=self.cleared.tolist(),
            update_wall_s=self.update_wall_s, score_wall_s=self.score_wall_s,
            visibility_computations=self.visibility_computations,
            cached_points=len(self._visibility_cache),
            semantics="Finite negative-only compatibility scores; never probability or safety/completion proof")

    @classmethod
    def replay(cls, public_history):
        memory = cls()
        for row in public_history:
            memory.observe(**row)
        return memory
