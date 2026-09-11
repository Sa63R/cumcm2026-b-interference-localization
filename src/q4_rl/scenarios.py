"""Predeclared synthetic Q4 scenes; never constructed from audit data.

Scenario metadata belongs to the environment/evaluator, never the policy.
This generator intentionally includes legal all-directional scenes.
"""
from dataclasses import dataclass
import math
import random

from simulation.cases import Scenario, Source

SPLIT_BASES = {"train": 8000000, "development": 8100000,
               "confirmation": 8200000, "final": 8300000}
SPLIT_SIZE = 100000
FAMILIES = ("random", "minimum_radius", "boundary_outward", "cluster",
            "positive_error", "negative_error", "alternating_error", "narrow_strip")
SOURCE_MODES = ("mixed", "all_directional")


@dataclass(frozen=True)
class Q4Scenario(Scenario):
    def __post_init__(self):
        if type(self.problem) is not int or self.problem != 4:
            raise ValueError("Q4 only")
        if type(self.seed) is not int or not self.case_id:
            raise ValueError("Explicit reproducible scene identity required")
        if not isinstance(self.sources, tuple) or not 10 <= len(self.sources) <= 16:
            raise ValueError("Q4 requires 10..16 sources")
        if any(not isinstance(s, Source) for s in self.sources):
            raise ValueError("Validated Source instances required")
        if len({s.channel for s in self.sources}) != len(self.sources):
            raise ValueError("Exclusive channels required")
        if not any(s.orientation_deg is not None for s in self.sources):
            raise ValueError("Q4 requires a directional source")
        if self.error_mode not in {"uniform", "zero", "positive_extreme",
                                   "negative_extreme", "alternating_extreme"}:
            raise ValueError("Unknown error mode")


def split_of_seed(seed):
    if type(seed) is not int:
        raise ValueError("Seed must be an integer")
    for split, base in SPLIT_BASES.items():
        if base <= seed < base + SPLIT_SIZE:
            return split
    raise ValueError("Seed outside the isolated Q4 RL namespaces")


def build_case(seed, *, split=None, family="random", source_mode="mixed", count=None):
    actual_split = split_of_seed(seed)
    if split is not None and split != actual_split:
        raise ValueError("Scene belongs to a different data split")
    if family not in FAMILIES or source_mode not in SOURCE_MODES:
        raise ValueError("Unknown predeclared family/source mode")
    # An identical seed gives the same primitive scene before stress transforms.
    rng = random.Random(seed + 4 * 1000003)
    n = rng.randint(10, 16) if count is None else count
    if type(n) is not int or not 10 <= n <= 16:
        raise ValueError("Source count must be in 10..16")
    channels = rng.sample(list(range(1, 21)), n)
    directed = set(channels if source_mode == "all_directional" else
                   rng.sample(channels, rng.randint(1, n - 1)))
    phase = rng.uniform(0, 2 * math.pi)
    sources = []
    for i, channel in enumerate(channels):
        angle, radius = rng.uniform(0, 2 * math.pi), 1800 * math.sqrt(rng.random())
        x, y = radius * math.cos(angle), radius * math.sin(angle)
        reception = rng.uniform(1000, 1500) if family == "random" else 1000.
        orientation = rng.uniform(0, 360) if channel in directed else None
        if family == "boundary_outward":
            theta = phase + 2 * math.pi * i / n
            x, y = 1799.9 * math.cos(theta), 1799.9 * math.sin(theta)
            orientation = math.degrees(theta) % 360 if channel in directed else None
        elif family == "cluster":
            x, y = 1100 * math.cos(phase) + rng.uniform(-.25, .25), 1100 * math.sin(phase) + rng.uniform(-.25, .25)
        elif family == "narrow_strip":
            along, across = 900 + 30 * i, (-1)**i * .2
            x, y = along * math.cos(phase) - across * math.sin(phase), along * math.sin(phase) + across * math.cos(phase)
        sources.append(Source(channel, x, y, reception, orientation))
    error = {"positive_error": "positive_extreme", "negative_error": "negative_extreme",
             "alternating_error": "alternating_extreme", "cluster": "alternating_extreme",
             "narrow_strip": "alternating_extreme"}.get(family, "uniform")
    identity = f"q4-rl-{actual_split}-{source_mode}-{family}-n{n}-{seed}"
    return Q4Scenario(identity, 4, seed, tuple(sources), error,
                      "Predeclared synthetic assumption; not fitted to official/audit cases")


def case_requests(split="development", *, start=None, count=8, families=("random",),
                  source_modes=("mixed", "all_directional")):
    if split not in SPLIT_BASES or type(count) is not int or count < 1:
        raise ValueError("Invalid split/count")
    start = SPLIT_BASES[split] if start is None else start
    if len(set(families)) != len(families) or len(set(source_modes)) != len(source_modes):
        raise ValueError("Duplicate scene strata")
    requests = []
    for seed in range(start, start + count):
        if split_of_seed(seed) != split:
            raise ValueError("Requested range crosses data splits")
        for source_mode in source_modes:
            for family in families:
                if source_mode not in SOURCE_MODES or family not in FAMILIES:
                    raise ValueError("Unknown scene stratum")
                requests.append(dict(seed=seed, split=split, family=family, source_mode=source_mode))
    if not requests:
        raise ValueError("No scenes requested")
    return requests
