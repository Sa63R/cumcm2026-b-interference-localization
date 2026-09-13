"""Q3 strategy bridge to the attachment 2 official protocol. No import-time I/O."""
from __future__ import annotations
import ast
import math
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
COMPARISON = HERE.parent / 'q3_comparison'
sys.path[:0] = [str(CODE / 'src'), str(COMPARISON / 'vendor' / 'B_Q3_optimized_v3'), str(COMPARISON)]
import q3_base as b
import q3_v3 as v
from simulator_client import SimulatorClient

METHODS = ('phased', 'joint', 'v2', 'v3', 'v3_origin20', 'optical',
           'scenario', 'future_cover', 'scenario_future')


class OfficialDevice:
    def __init__(self, client, progress=None):
        self.__client = client
        self._pos = np.array([client.state.position.x, client.state.position.y])
        self.progress = progress
        self.last_success_virtual_s = None
        self.request_wall_s = 0.

    @property
    def pos(self):
        return self._pos.copy()

    @property
    def channel(self):
        return self.__client.state.current_channel

    def move(self, p):
        q = np.asarray(p, dtype=float)
        if q.shape != (2,) or not np.all(np.isfinite(q)) or np.any(np.abs(q) > 2_000_000):
            raise ValueError('Invalid destination')
        # The protocol combines movement with the next measure or clear.
        self._pos = q.copy()

    def _request(self, method, channel):
        start = time.perf_counter()
        try:
            response = method(tuple(map(float, self._pos)), int(channel))
        finally:
            self.request_wall_s += time.perf_counter() - start
        if self.progress:
            self.progress(self.__client)
        return response

    def detect(self, c):
        response = self._request(self.__client.measure, c)
        kind = response['measure_result']
        if kind == 'direction':
            return b.Observation('bearing', math.radians(response['svd_deg']))
        if kind == 'near':
            return b.Observation('strong')
        if kind == 'no_signal':
            return b.Observation('none')
        raise ValueError(kind)

    def clear(self, c):
        response = self._request(self.__client.clear, c)
        success = response['clear_result'] == 'success'
        if success:
            self.last_success_virtual_s = response['virtual_time_s']
        return success


def make_agent(device, method):
    b.A = math.radians(1.005)
    public = v.PublicBackend(device)
    if method in ('phased', 'joint'):
        return b.Agent(public, mode=method)
    if method in ('scenario', 'future_cover', 'scenario_future'):
        from new_methods import PlanningAgent
        return PlanningAgent(public, scenario=method != 'future_cover', future_cover=method != 'scenario')
    if method == 'optical':
        # The original module imports pandas solely for its standalone CSV
        # benchmark. Execute its unchanged policy definitions without that CLI.
        source = COMPARISON / 'vendor' / 'avg200_diagnostics' / 'optical_experiment.py'
        tree = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
        tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name != 'main']
        namespace = {'np': np, 'b': b, 'v': v}
        exec(compile(tree, str(source), 'exec'), namespace)
        return namespace['ExperimentalOpticalAgent'](public)
    return v.make_agent(device, method)


def run_policy(client, method, progress=None):
    device = OfficialDevice(client, progress)
    start = time.perf_counter()
    agent = make_agent(device, method)
    agent.run()
    wall_s = time.perf_counter() - start
    if agent.unseen or agent.tracks or len(agent.cleared | agent.absent) != 20:
        raise RuntimeError('Incomplete channel certificate')
    if len(agent.cleared) != client.state.cleared_count:
        raise RuntimeError('Clear count differs from accepted API feedback')
    return dict(cleared=sorted(agent.cleared), absent=sorted(agent.absent),
                complete_channel_certificate=True, bearing_bound_deg=1.005,
                last_success_virtual_s=device.last_success_virtual_s,
                policy_wall_s=wall_s, request_wall_s=device.request_wall_s,
                nonrequest_wall_s=wall_s-device.request_wall_s,
                speculative_success=getattr(agent, 'speculative_success', 0),
                speculative_failure=getattr(agent, 'speculative_failure', 0),
                scenario_decisions=getattr(agent, 'scenario_decisions', 0),
                information_stops=getattr(agent, 'information_stops', 0),
                plan_changes=getattr(agent, 'plan_changes', 0))
