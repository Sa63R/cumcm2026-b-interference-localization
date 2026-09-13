"""Audit platform differences and Windows in-process/HTTP bridge equivalence."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'checks/simulator')]
import run_official as r
from test_bridge import trace_hash
from jammers_local.core import Scenario, Session, LocalClient
from jammers_local.server import Application, Server


class LocalDevice:
    def __init__(self, client):
        self.client = client
        self.position = (0., 0.)
        self.channel = 1
    def move(self, point):
        self.position = tuple(map(float, point))
    def detect(self, channel):
        response = self.client.measure(self.position, channel)
        self.channel = channel
        if response['measure_result'] == 'direction':
            return r.b.Observation('bearing', r.math.radians(response['svd_deg']))
        return r.b.Observation('strong' if response['measure_result'] == 'near' else 'none')
    def clear(self, channel):
        return self.client.clear(self.position, channel)['clear_result'] == 'success'


def normalized(history):
    return [dict(path=x['path'], position=x['request'].get('position'), channel=x['request'].get('channel'),
                 response={k: v for k, v in x['response'].items()
                           if k not in ('real_timestamp_ms', 'remaining_real_duration_s')}) for x in history]


def main():
    output = ROOT / 'windows_platform_checks'
    output.mkdir(exist_ok=False)
    sites, certificate, _ = r.prepare()
    fixtures = json.loads((ROOT / 'checks/cases.json').read_text())
    with gzip.open(ROOT / 'expected_traces.json.gz', 'rt') as stream:
        mac = json.load(stream)
    results = []
    for case in fixtures:
        scene = Scenario.from_dict(case['scenario'])
        session = Session(scene)
        client = LocalClient(session.dispatch)
        client.enter()
        r.solve_v6_lite(LocalDevice(client))
        client.exit()
        app = Application(scene)
        server = Server(app, 0)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        report = {}
        try:
            with r.SimulatorClient('local-test', base_url=f'http://127.0.0.1:{server.server_port}',
                                   log_path=output / (case['key'] + '.jsonl')) as remote:
                r.run_session(remote, report, sites, certificate)
        finally:
            server.shutdown(); server.server_close(); worker.join()
        local_trace = normalized(session.history)
        assert trace_hash(session.history) == trace_hash(app.session.history)
        assert session.engine.time_us == app.session.engine.time_us
        assert report['status'] == 'policy_completed_and_exited'
        assert len(app.session.engine.cleared) == len(app.session.engine.sources)
        expected = mac[case['key']]
        same_count = len(local_trace) == len(expected)
        same_actions = same_count and all((a['path'], a['channel']) == (b['path'], b['channel'])
                                         for a, b in zip(local_trace, expected))
        max_coordinate_difference = None
        responses_equal = False
        if same_actions:
            max_coordinate_difference = max([abs(a['position'][axis] - b['position'][axis])
                for a, b in zip(local_trace, expected) if a['position'] is not None for axis in ['x', 'y']] or [0.])
            responses_equal = all(a['response'] == b['response'] for a, b in zip(local_trace, expected))
        row = dict(case=case['key'], windows_http_trace_equal=True, all_cleared=True,
                   actions=len(local_trace), virtual_time_us=session.engine.time_us,
                   mac_virtual_time_difference_us=session.engine.time_us - case['virtual_time_us'],
                   mac_action_sequence_equal=same_actions, mac_responses_equal=responses_equal,
                   maximum_coordinate_difference_m=max_coordinate_difference)
        results.append(row)
        with gzip.open(output / (case['key'] + '-session.json.gz'), 'wt') as stream:
            json.dump(app.session.export(), stream)
        print(json.dumps(row), flush=True)
    (output / 'summary.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
