"""Q4 V6: guarded transit sensing and learned completion-cost action ranking.

Production inference uses only Python's standard library. The supplied tree
weights were trained OFFLINE on separate synthetic scenarios. The online
Device interface never exposes truth, scenario seed, or remaining source count.
No official HTTP endpoints or response fields are assumed.
"""
from __future__ import annotations
from dataclasses import dataclass,replace,asdict
import math
from q4_guarded_solver import solve_guarded
from q4_learned_transit import solve_learned
from q4_v5 import solve_v5

@dataclass(frozen=True)
class V6Config:
    route_model: str = 'route_critic_extra.json'
    transit_model: str = 'transit_critic_big_extra.json'
    route_threshold: float = 0.
    transit_threshold: float = 0.
    max_transit_stops: int = 16
    minimum_stations: int = 4
    route_ranking: bool = True
    transit_sensing: bool = True
    use_v5_local: bool = True


def default_config() -> V6Config:
    return V6Config()


def solve_v6(device, config: V6Config | None = None) -> dict:
    """Run one fresh session. Correctness comes from geometry and feedback,
    not from either learned score. Model scoring only selects legal actions.
    """
    c=config or default_config()
    if not isinstance(c.max_transit_stops,int) or not 0<=c.max_transit_stops<=64:
        raise ValueError('max_transit_stops must be an integer in [0,64]')
    if not isinstance(c.minimum_stations,int) or not 0<=c.minimum_stations<=21:
        raise ValueError('minimum_stations must be an integer in [0,21]')
    if not all(math.isfinite(z) and z>=0 for z in [c.route_threshold,c.transit_threshold]):
        raise ValueError('Action thresholds must be finite and nonnegative')
    if c.route_ranking or c.transit_sensing:
        r=solve_guarded(device,c.route_model if c.route_ranking else None,c.route_threshold,
              v5=c.use_v5_local,transit=c.transit_model if c.transit_sensing else None,
              transit_threshold=c.transit_threshold,maxstops=c.max_transit_stops,
              minimum_stations=c.minimum_stations)
    else:
        if c.use_v5_local:r=solve_v5(device)
        else:
            from q4_state import Engine
            r=Engine(device).run_base()
    r['algorithm']='Q4_V6_GUARDED';r['configuration']=asdict(c)
    return r
