"""Analytically marginalize source direction and receiver radius at fixed position.

Scope: planning probabilities only, NOT a continuous-geometry safety certificate.
Position is an input candidate, not assumed known to the real agent. Bearings,
optical outcomes, and candidate-position weights are handled outside this module.
Assumed planning prior: uniform direction, radius prior specified below, and an
explicit prior probability of a directional source. This is NOT an official
simulator distribution. All same-position RF observations concern a static source.

The full rolling-horizon planner is intentionally NOT implemented here.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import random
from typing import Iterable

Point = tuple[float, float]
TAU = 2.0 * math.pi


def _point(p: Point) -> Point:
    if len(p) != 2 or not all(math.isfinite(float(v)) for v in p):
        raise ValueError('A position must have two finite coordinates')
    return (float(p[0]), float(p[1]))


@dataclass(frozen=True)
class RadiusPrior:
    """Mixture of Uniform[1000,1500] and optional atoms at legal radii.

    The interval queried is [positive_lower, negative_upper). A missing negative
    upper bound is +infinity, so the upper endpoint 1500 remains inclusive for
    an atom there. Atoms make boundary stress models possible without pretending
    continuous priors assign positive mass to endpoints.
    """
    uniform_weight: float = 1.0
    atoms: tuple[tuple[float, float], ...] = ()  # (radius, probability mass)

    def __post_init__(self) -> None:
        if not math.isfinite(self.uniform_weight) or self.uniform_weight < 0:
            raise ValueError('Invalid uniform weight')
        for radius, weight in self.atoms:
            if not 1000 <= radius <= 1500 or not math.isfinite(weight) or weight < 0:
                raise ValueError('Invalid radius atom')
        if not math.isclose(self.uniform_weight + sum(w for _, w in self.atoms),
                            1.0, abs_tol=1e-12, rel_tol=0):
            raise ValueError('Prior weights must sum to one')

    def mass(self, lower: float, upper: float = math.inf) -> float:
        if math.isnan(lower) or math.isnan(upper):
            raise ValueError('NaN interval bound')
        continuous = max(0., min(1500., upper) - max(1000., lower)) / 500.
        return (self.uniform_weight * continuous +
                sum(w for r, w in self.atoms if lower <= r < upper))


@dataclass(frozen=True)
class AngularPiece:
    begin: float
    end: float
    radius_lower: float
    radius_upper_exclusive: float
    mass: float  # includes uniform-direction arc probability


@dataclass(frozen=True)
class Marginal:
    omni_evidence: float
    directional_evidence: float
    mixture_evidence: float
    posterior_directional: float | None
    pieces: tuple[AngularPiece, ...]


def marginalize(position: Point, positive: Iterable[Point] = (),
                negative: Iterable[Point] = (), *, directional_prior: float = .5,
                radius_prior: RadiusPrior | None = None) -> Marginal:
    """P(RF visibility history | position, source present), integrated over R,u.

    Only the continuous orientation integral is exact up to floating point.
    It ignores probability-zero angle endpoints; NEVER use zero returned mass
    as a proof that no legal geometric hypothesis exists.
    """
    if not math.isfinite(directional_prior) or not 0 <= directional_prior <= 1:
        raise ValueError('directional_prior must lie in [0,1]')
    prior = radius_prior or RadiusPrior()
    g = _point(position)
    positives = tuple(dict.fromkeys(_point(p) for p in positive))
    negatives = tuple(dict.fromkeys(_point(p) for p in negative))
    pos = [(p[0]-g[0], p[1]-g[1]) for p in positives]
    neg = [(p[0]-g[0], p[1]-g[1]) for p in negatives]
    pd = [math.hypot(*v) for v in pos]
    nd = [math.hypot(*v) for v in neg]
    lower = max([1000.] + pd)
    empty = Marginal(0., 0., 0., None, ())
    if lower > 1500 or set(positives).intersection(negatives):
        return empty
    # At the source, a negative RF result is impossible for either type.
    if any(d == 0 for d in nd):
        return empty
    omni = prior.mass(lower, min(nd, default=math.inf))
    # Illumination membership only changes at these angular boundaries.
    boundaries = {0., TAU}
    for x, y in pos + neg:
        if x == 0 and y == 0:
            continue
        a = math.atan2(y, x)
        boundaries.add((a-math.pi/2) % TAU)
        boundaries.add((a+math.pi/2) % TAU)
    cut = sorted(boundaries)
    pieces = []
    for a, z in zip(cut, cut[1:]):
        if z-a <= 1e-14:
            continue
        angle = (a+z)/2
        ux, uy = math.cos(angle), math.sin(angle)
        if any(ux*x + uy*y < 0 for x, y in pos):
            continue
        upper = min((d for (x, y), d in zip(neg, nd)
                     if ux*x + uy*y >= 0), default=math.inf)
        mass = (z-a)/TAU * prior.mass(lower, upper)
        if mass > 0:
            pieces.append(AngularPiece(a, z, lower, upper, mass))
    directional = math.fsum(p.mass for p in pieces)
    mixture = (1-directional_prior)*omni + directional_prior*directional
    posterior = directional_prior*directional/mixture if mixture > 0 else None
    return Marginal(omni, directional, mixture, posterior, tuple(pieces))


def visibility_probability(position: Point, positive: Iterable[Point],
                           negative: Iterable[Point], probe: Point, *,
                           directional_prior: float = .5,
                           radius_prior: RadiusPrior | None = None) -> float:
    """P(next RF result is positive | candidate position and previous history)."""
    positive, negative = tuple(positive), tuple(negative)
    kw = dict(directional_prior=directional_prior, radius_prior=radius_prior)
    denominator = marginalize(position, positive, negative, **kw).mixture_evidence
    if denominator <= 0:
        raise ValueError('History has zero prior mass at this candidate; this is not an absence certificate')
    numerator = marginalize(position, positive+(_point(probe),), negative, **kw).mixture_evidence
    return min(1., max(0., numerator/denominator))


def _logadd(a: float, b: float) -> float:
    if a == -math.inf: return b
    if b == -math.inf: return a
    m = max(a, b)
    return m + math.log1p(math.exp(min(a, b)-m))


def count_and_subset_sampler(present_likelihoods: list[float], known_count: int,
                             *, total_channels: int = 20, min_count: int = 10,
                             max_count: int = 16,
                             count_prior: dict[int, float] | None = None):
    """Posterior source count + exact weighted subset sampler under stated prior.

    Prior: choose N from count_prior, then uniformly choose N channels out of
    total_channels. Known channels have positive evidence; list items correspond
    to as-yet unknown channels. A known-absent channel can be omitted. Each item
    is P(its all-negative history | source present), with absent likelihood=1.
    This function enforces count constraints, not the separate 'both types exist'
    constraint. A full scenario sampler must condition that constraint as well.
    Returns (count_probability_dict, sample_subset_callable).
    """
    if not 0 <= known_count <= total_channels or not 0 <= min_count <= max_count <= total_channels:
        raise ValueError('Invalid counts')
    m = len(present_likelihoods)
    if m+known_count > total_channels:
        raise ValueError('Too many channels')
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in present_likelihoods):
        raise ValueError('Present likelihoods must lie in [0,1]')
    cp = count_prior or {n: 1./(max_count-min_count+1)
                         for n in range(min_count, max_count+1)}
    if any(not math.isfinite(w) or w < 0 for w in cp.values()):
        raise ValueError('Invalid count prior')
    if any(n < min_count or n > max_count for n, w in cp.items() if w):
        raise ValueError('Count prior outside allowed range')
    lp = [math.log(x) if x else -math.inf for x in present_likelihoods]
    # Suffix elementary symmetric polynomials, stored in log space.
    table = [[-math.inf]*(m+1) for _ in range(m+1)]
    table[m][0] = 0.
    for i in range(m-1, -1, -1):
        table[i][0] = 0.
        for r in range(1, m-i+1):
            table[i][r] = _logadd(table[i+1][r], lp[i]+table[i+1][r-1])
    logs = {}
    for r in range(m+1):
        n = known_count+r
        if cp.get(n, 0) > 0 and table[0][r] > -math.inf:
            logs[n] = math.log(cp[n])-math.log(math.comb(total_channels, n))+table[0][r]
    if not logs:
        raise ValueError('No count-consistent posterior mass; do not declare success')
    pivot = max(logs.values())
    raw = {n: math.exp(v-pivot) for n, v in logs.items()}
    normalizer = math.fsum(raw.values())
    probabilities = {n: p/normalizer for n, p in raw.items()}

    def sample_subset(rng: random.Random) -> tuple[int, tuple[int, ...]]:
        n = rng.choices(list(probabilities), weights=list(probabilities.values()))[0]
        r = n-known_count
        selected = []
        for i in range(m):
            if r == 0: break
            inc = lp[i]+table[i+1][r-1]
            prob = 0. if inc == -math.inf else math.exp(inc-table[i][r])
            if rng.random() < min(1., prob):
                selected.append(i); r -= 1
        if r != 0:
            raise RuntimeError('Internal weighted-subset error')
        return n, tuple(selected)

    return probabilities, sample_subset
