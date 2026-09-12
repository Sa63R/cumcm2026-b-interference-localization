"""Q4 bridge for attachment 2. Importing this module never contacts a simulator."""
from __future__ import annotations

import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
sys.path[:0] = [str(CODE / 'src'), str(HERE.parent / 'q4_comparison')]
import common
import q4_baseline as b
from online import State
from q4_coverage import certify
from simulator_client import SimulatorClient

METHODS = ('v4',)


def configure_geometry():
    # Attachment 2 rounds degrees to 0.01. Add the maximum rounding error
    # to the physical 1 degree bound. Keep the archived source files intact.
    b.DELTA = math.radians(1.005)
    b.COS_D, b.TAN_D = math.cos(b.DELTA), math.tan(b.DELTA)
    contraction_bound = math.sqrt(21 / 16 - b.COS_D + math.sin(b.DELTA) / 2)
    if contraction_bound >= b.KAPPA:
        raise RuntimeError('Paired-probe contraction constant is insufficient')
    return {'bearing_bound_deg': 1.005, 'contraction_bound': contraction_bound,
            'kappa': b.KAPPA, 'clear_certificate_radius_m': b.CLEAR_CERT_RADIUS}


class OfficialDevice:
    """Only accepted public feedback reaches the strategy.

    Device.move records a destination: the official protocol combines the move
    with the next measure/clear. Radio channel changes only on measure.
    """
    def __init__(self, client):
        self.__client = client
        self._position = (client.state.position.x, client.state.position.y)
        self.last_success_virtual_s = None

    @property
    def position(self):
        return self._position

    @property
    def channel(self):
        return self.__client.state.current_channel

    def move(self, position):
        if len(position) != 2 or not all(math.isfinite(v) and abs(v) <= 2_000_000 for v in position):
            raise ValueError('Invalid requested destination')
        self._position = tuple(map(float, position))

    def detect(self, channel):
        response = self.__client.measure(self.position, channel)
        kind = response['measure_result']
        if kind == 'direction':
            return b.Observation('bearing', math.radians(response['svd_deg']))
        if kind == 'near':
            return b.Observation('strong')
        if kind == 'no_signal':
            return b.Observation('none')
        raise ValueError(f'Unrecognized measure response: {kind}')

    def clear(self, channel):
        response = self.__client.clear(self.position, channel)
        success = response['clear_result'] == 'success'
        if success:
            self.last_success_virtual_s = response['virtual_time_s']
        return success


def prepare_policy(method='v4', seed=90210000):
    if method not in METHODS:
        raise ValueError(method)
    geometry = configure_geometry()
    return State(), None, geometry


def run_policy(client, state, planner, progress=None):
    device = OfficialDevice(client)
    while state.prepare():
        actions = state.ordered_actions(device)
        action = actions[0] if planner is None else planner.choose(state, device, actions)
        state.execute(device, action)
        if progress:
            progress(state, client)
    if len(state.cleared) != client.state.cleared_count:
        raise RuntimeError('Strategy clear count disagrees with accepted API replies')
    certificate = None
    if len(state.cleared) < 16:
        certificate = certify(state.scanpoints)
        if not certificate['ok']:
            raise RuntimeError('Actual scan points failed the final coverage certificate')
    report = state.report()
    report['coverage_certificate'] = certificate
    report['last_success_virtual_s'] = device.last_success_virtual_s
    if planner is not None:
        report['planning'] = planner.stats
    return report
