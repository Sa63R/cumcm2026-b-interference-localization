"""Use the existing HTTP client against a fresh loopback reconstruction server."""
import argparse
import json
from pathlib import Path
import sys
import threading
import benchmark as bench
from jammers_local.core import Scenario
from jammers_local.server import Application, Server
from archive_utils import read_session

sys.path.insert(0,str(bench.ROOT.parents[1]/'src'))
from simulator_client import SimulatorClient


def trace(history):
    return [(r['path'],r['request'].get('position'),r['request'].get('channel'),
             {k:v for k,v in r['response'].items() if k not in ('real_timestamp_ms','remaining_real_duration_s')})
            for r in history]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be absent or empty')
    bench.verify_sources()
    bench.initialize(1.005)
    plan=json.loads((bench.ROOT/'plan.json').read_text());case=plan['cases'][0]
    destination=args.output;destination.mkdir(parents=True,exist_ok=True)
    rows=[]
    for method in plan['methods']:
        app=Application(Scenario.from_dict(case['scenario']))
        server=Server(app,0)
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        try:
            with SimulatorClient('local-test',base_url=f'http://127.0.0.1:{server.server_port}',
                                 log_path=destination/f'{method}-client.jsonl') as client:
                client.enter();device=bench.Device(client)
                if method=='v4':report=bench.solve_v4(device)
                elif method=='v5':report=bench.solve_v5(device)
                else:report=bench.solve_v6(device,bench.V6_CONFIG)
                client.exit()
            certificate=bench.audit(app.session,report)
            expected=read_session(bench.ROOT/'main',f'{case["key"]}-{method}.json')
            assert trace(expected['history'])==trace(app.session.history), 'HTTP and in-process protocol traces differ'
            bench.dump(destination/f'{method}-session.json',dict(**app.session.export(),completion_audit=certificate))
            rows.append(dict(method=method,case_key=case['key'],trace_equal=True,
                             actions=len(app.session.history),virtual_time_us=app.session.engine.time_us))
            print(method,'HTTP complete and full trace identical',flush=True)
        finally:
            server.shutdown();server.server_close();worker.join()
    bench.dump(destination/'summary.json',rows)


if __name__=='__main__':main()
