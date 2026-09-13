"""Seeded research cases. The generator is an assumption, not the official law."""

from dataclasses import asdict, dataclass, replace
import math
import random

from simulator_client.rules import ARENA_RADIUS_M, CHANNELS


@dataclass(frozen=True)
class Source:
    channel: int
    x: float
    y: float
    reception_radius_m: float
    orientation_deg: float | None = None

    def __post_init__(self):
        if isinstance(self.channel, bool) or self.channel not in CHANNELS:
            raise ValueError("A source needs an integer channel in 1..20")
        if not isinstance(self.channel, int):
            raise ValueError("A source needs an integer channel in 1..20")
        if not all(math.isfinite(v) for v in (self.x, self.y, self.reception_radius_m)):
            raise ValueError("Source coordinates and radius must be finite")
        if math.hypot(self.x, self.y) > ARENA_RADIUS_M + 1e-9:
            raise ValueError("Sources must be inside the 1800 m target disk")
        if not 1000 <= self.reception_radius_m <= 1500:
            raise ValueError("Reception radius must be in [1000, 1500]")
        if self.orientation_deg is not None and (
            not math.isfinite(self.orientation_deg) or not 0 <= self.orientation_deg < 360
        ):
            raise ValueError("Orientation must be None or an angle in [0, 360)")


@dataclass(frozen=True)
class Scenario:
    case_id: str
    problem: int
    seed: int
    sources: tuple[Source, ...]
    error_mode: str = "uniform"
    description: str = ""

    def __post_init__(self):
        if self.problem not in (3, 4):
            raise ValueError("Problem must be 3 or 4")
        if not 10 <= len(self.sources) <= 16:
            raise ValueError("Research competition cases must have 10..16 sources")
        if len({s.channel for s in self.sources}) != len(self.sources):
            raise ValueError("Every source must have an exclusive channel")
        if self.problem == 3 and any(s.orientation_deg is not None for s in self.sources):
            raise ValueError("Question 3 has omnidirectional sources only")
        if self.problem == 4 and not (
            any(s.orientation_deg is None for s in self.sources)
            and any(s.orientation_deg is not None for s in self.sources)
        ):
            raise ValueError("Question 4 research cases contain both source types")
        if self.error_mode not in {
            "uniform", "zero", "positive_extreme", "negative_extreme", "alternating_extreme"
        }:
            raise ValueError("Unknown error mode")

    def evaluation_config(self) -> dict:
        """For the experiment harness only; never passed to search policies."""
        return asdict(self)


def random_scenario(problem: int, seed: int) -> Scenario:
    """Uniform disk locations, uniform radii, and sampled exclusive channels."""
    rng = random.Random(seed + problem * 1_000_003)
    count = rng.randint(10, 16)
    channels = rng.sample(list(CHANNELS), count)
    directed = set(rng.sample(channels, rng.randint(1, count - 1))) if problem == 4 else set()
    sources = []
    for channel in channels:
        theta = rng.uniform(0, 2 * math.pi)
        radius = ARENA_RADIUS_M * math.sqrt(rng.random())
        sources.append(Source(
            channel, radius * math.cos(theta), radius * math.sin(theta),
            rng.uniform(1000, 1500),
            rng.uniform(0, 360) if channel in directed else None,
        ))
    return Scenario(f"q{problem}-random-{seed:04d}", problem, seed, tuple(sources),
                    description="Uniform-area positions and uniform reception radii; assumed research distribution")


def difficult_scenarios(problem: int) -> list[Scenario]:
    """Deterministic boundary, radius, error and geometry stress cases."""
    base = random_scenario(problem, 100_000)
    minimum = tuple(replace(s, reception_radius_m=1000.0) for s in base.sources)
    boundary = tuple(Source(
        i + 1, 1800 * math.cos(2 * math.pi * i / 12),
        1800 * math.sin(2 * math.pi * i / 12), 1000.0,
        (360 * i / 12) if problem == 4 and i != 0 else None,
    ) for i in range(12))
    close = tuple(replace(
        s, x=850 + 0.25 * math.cos(2 * math.pi * i / len(base.sources)),
        y=350 + 0.25 * math.sin(2 * math.pi * i / len(base.sources)),
        reception_radius_m=1000.0,
    ) for i, s in enumerate(base.sources))
    parallel = tuple(replace(s, x=1000 + i * 24, y=(-1) ** i * 0.2,
                             reception_radius_m=1000.0)
                     for i, s in enumerate(base.sources))
    specifications = [
        ("boundary_outward", boundary, "uniform", "Boundary sources; Q4 directional sources face outside the disk"),
        ("minimum_radius", minimum, "uniform", "All reception radii equal the guaranteed minimum 1000 m"),
        ("positive_error", minimum, "positive_extreme", "All latent angular errors +1 degree before bounded quantization"),
        ("negative_error", minimum, "negative_extreme", "All latent angular errors -1 degree before bounded quantization"),
        ("alternating_error", minimum, "alternating_extreme", "Location-hashed angular errors alternate between -1 and +1 degree"),
        ("close_targets", close, "alternating_extreme", "Sources on distinct channels within a 0.5 m cluster"),
        ("near_parallel", parallel, "alternating_extreme", "Sources in a narrow eastward strip: nearly parallel initial bearings"),
    ]
    return [Scenario(f"q{problem}-hard-{name}", problem, 200_000 + i, sources, mode, description)
            for i, (name, sources, mode, description) in enumerate(specifications)]
