"""Default V3 behaviour with cached route distances, no virtual-time change.

Equivalent action traces are tested against original V3. This convenience
wrapper does not expose V3's nondefault experimental routing configurations.
"""
from q4_v4_solver import solve_v4,V4Config
from q4_v4_local import LocalConfig

def solve_v3_cached(device):
    return solve_v4(device,V4Config(local=LocalConfig(optical_cover_limit=6,pause_after_pair=False)))
