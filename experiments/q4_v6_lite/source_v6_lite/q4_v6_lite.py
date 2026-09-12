"""V6 Lite: V4 ordering/sharing, local rollout and guarded learned transit.

Only route model ranking and posterior shared-sensing scoring are removed.
The posterior used by local rollout is retained. Geometry, coverage and real
clear feedback remain the completion criteria. Inference uses the stdlib.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import random
import time

import q4_baseline as b
from q4_v4_solver import V4Config
from q4_state import Engine
from q4_belief import SceneFactory, PosteriorUnavailable
from q4_proposals import local_candidates
from q4_transit_critic import Critic, proposals, apply_probe


@dataclass(frozen=True)
class LiteConfig:
    base: V4Config = field(default_factory=V4Config)
    transit_model: str = str(Path(__file__).with_name('transit_critic_big_extra.json'))
    transit_threshold: float = 0.
    max_transit_stops: int = 16
    minimum_stations: int = 4
    scenes: int = 24
    position_draws: int = 128
    min_gain: float = 2.
    max_decisions: int = 40
    max_free_probes: int = 40
    seed: int = 67541
    total_planning_seconds: float = 120.


def default_config() -> LiteConfig:
    return LiteConfig()


def solve_v6_lite(device, config: LiteConfig | None = None) -> dict:
    """Use only position/channel/move/detect/clear on a fresh device session."""
    cfg = config or default_config()
    for name, low, high in [('max_transit_stops', 0, 64), ('minimum_stations', 0, 21),
                            ('scenes', 1, None), ('position_draws', 1, None),
                            ('max_decisions', 0, None), ('max_free_probes', 0, None)]:
        value = getattr(cfg, name)
        if type(value) is not int or value < low or (high is not None and value > high):
            raise ValueError(f'Invalid {name}')
    if not all(math.isfinite(value) and value >= 0 for value in
               [cfg.transit_threshold, cfg.min_gain, cfg.total_planning_seconds]):
        raise ValueError('Thresholds and planning budget must be finite and nonnegative')

    engine = Engine(device, cfg.base)
    transit = Critic(cfg.transit_model)
    factory = SceneFactory(cfg.position_draws, cfg.seed)
    rng = random.Random(cfg.seed)
    calls = accepts = stops = tests = guarded_steps = 0
    elapsed = 0.
    last = activation = None
    chosen = []
    failures = {}
    planner_budget_reached = False
    while not engine.done():
        order = engine.ordered_actions()
        active = len(engine.state.visited) >= cfg.minimum_stations
        if not active:
            guarded_steps += 1
        elif activation is None:
            activation = dict(visited_stations=len(engine.state.visited),
                              known_sources=len(engine.state.pending) + len(engine.state.cleared))
        base = action = order[0]
        if active and stops < cfg.max_transit_stops and last != (base.kind, base.key):
            candidates = proposals(engine, base)
            if candidates:
                values = [transit.predict(p['features']) for p in candidates]
                selected = max(range(len(candidates)), key=lambda j: values[j])
                if values[selected] > cfg.transit_threshold:
                    tests += apply_probe(engine, candidates[selected])
                    stops += 1
                    last = (base.kind, base.key)
                    chosen.append(values[selected])
                    continue
        if elapsed >= cfg.total_planning_seconds:
            planner_budget_reached = True
        if base.kind == 'target' and calls < cfg.max_decisions and elapsed < cfg.total_planning_seconds:
            track = engine.state.pending[base.key]
            if track.obs.status != 'strong' and b.enclosing_circle(track.poly)[1] > b.CLEAR_CERT_RADIUS:
                started = time.perf_counter()
                calls += 1
                try:
                    pool = factory.pool(engine.state, base.key)
                    candidates = local_candidates(engine, base.key, pool, order, rng,
                                                  samples=cfg.scenes, top=1)
                    if candidates and candidates[0][1] > cfg.min_gain:
                        candidate = candidates[0][0]
                        if candidate.kind != 'probe' or engine.state.free_probes < cfg.max_free_probes:
                            action = candidate
                            accepts += 1
                except PosteriorUnavailable as error:
                    failures[str(error)] = failures.get(str(error), 0) + 1
                elapsed += time.perf_counter() - started
        engine.execute(action)
        last = None
    return dict(**engine.report(), algorithm='Q4_V6_LITE', configuration=asdict(cfg),
                transit_stops=stops, transit_detections=tests, critic_values=chosen,
                planner_calls=calls, planner_accepts=accepts, planner_seconds=elapsed,
                guarded_steps=guarded_steps, activation=activation,
                minimum_stations=cfg.minimum_stations, planner_failures=failures,
                planner_budget_reached=planner_budget_reached)
