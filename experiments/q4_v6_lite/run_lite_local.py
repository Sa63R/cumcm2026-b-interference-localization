"""Run the packaged Lite module against the local reconstructed public API."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'source_v6_lite'), str(ROOT / 'reference/simulator')]
import q4_baseline as b
from q4_v6_lite import solve_v6_lite
from local_device import Device
from jammers_local.core import Scenario, Session, LocalClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', default='q4-lite-local-demo')
    parser.add_argument('--out', type=Path, default=Path('LOCAL_ONLY_lite.json'))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f'Existing output preserved: {args.out}')
    b.DELTA = math.radians(1.005)
    b.COS_D, b.TAN_D = math.cos(b.DELTA), math.tan(b.DELTA)
    scene = Scenario.generate(4, args.seed)
    session = Session(scene)
    client = LocalClient(session.dispatch)
    client.enter()
    report = solve_v6_lite(Device(client))
    client.exit()
    snapshot = session.engine.snapshot(truth=True)
    assert snapshot['all_cleared']
    removed = [name for name in ['q4_route_critic', 'q4_rb_sharing', 'q4_v5'] if name in sys.modules]
    assert not removed
    payload = dict(local_reconstructed_only=True, all_cleared=True,
                   removed_modules_loaded=removed, snapshot=snapshot, report=report,
                   session=session.export())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    print(json.dumps(dict(output=str(args.out.resolve()), all_cleared=True,
                         task_seconds=snapshot['virtual_time_s']), ensure_ascii=False))


if __name__ == '__main__':
    main()
