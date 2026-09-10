"""Generate a reproducible LOCAL protocol-fixture report; no official connection."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from simulator_client import SimulatorClient
from tests.fake_simulator import FakeSimulator


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    destination = args.output_dir or Path("results/t01") / datetime.now(timezone.utc).strftime("local-%Y%m%dT%H%M%S%fZ")
    destination.mkdir(parents=True, exist_ok=False)
    responses = []
    with FakeSimulator() as server:
        with SimulatorClient(server.expected_robot_id, base_url=server.base_url,
                             log_path=destination / "requests.jsonl", retry_backoff_s=0) as client:
            client.enter()
            responses.append(client.measure((0, 0), 1))       # direction
            responses.append(client.measure((0, 0), 2))       # near
            server.drop_once_paths.add("/clear")
            responses.append(client.clear((0, 0), 2))         # success, lost response, replay
            responses.append(client.measure((300, 400), 1))   # near
            responses.append(client.clear((300, 400), 1))     # success
            responses.append(client.measure((300, 400), 1))   # no_signal
            responses.append(client.clear((300, 400), 1))     # no_target_in_range
            responses.append(client.measure((300, 0), 3))     # no_signal
            client.exit()
            outcomes = sorted({r.get("measure_result", r.get("clear_result")) for r in responses})
            report = {
                "run_kind": "local_protocol_fixture",
                "official_practice_completed": False,
                "description": "Two fixed test sources, zero bearing error; protocol validation only",
                "observed_results": outcomes,
                "responses": responses,
                "client_state": client.state.snapshot(),
                "fixture_state": server.state,
                "http_attempts": len(server.requests),
                "accepted_actions": client.state.accepted_actions,
                "journal": "requests.jsonl",
            }
    (destination / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(destination), "observed_results": outcomes,
                      "virtual_time_s": report["client_state"]["virtual_time_s"],
                      "official_practice_completed": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
