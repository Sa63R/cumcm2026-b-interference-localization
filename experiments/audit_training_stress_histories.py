"""Reconstruct geometry and report/physical action correspondence after stress runs."""

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

from summarize_inferred_region import audit


def run(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for policy in manifest["policies"]:
        # The independent replay uses this worktree's conservative geometry.
        # First require it to match the immutable policy source archive.
        for name in ("src/localization/omni.py", "src/localization/__init__.py", "src/geometry/__init__.py"):
            local = Path(__file__).resolve().parents[1] / name
            assert hashlib.sha256(local.read_bytes()).hexdigest() == policy["source_sha256"][name]
        files = sorted((directory / policy["name"]).glob("*.json.gz"))
        assert len(files) == manifest["case_count"]
        for path in files:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                data = json.load(stream)
            errors = []
            report = data["report"]
            if report is None:
                errors.append("No completed policy report; raw history preserved")
                reconstructed = None
            else:
                physical = [a for a in data["history"] if a["action"] in ("/measure", "/clear")]
                recorded = [a for a in report["action_history"] if a["action"] in ("measure", "clear")]
                if len(physical) != len(recorded):
                    errors.append("Policy/physical action count mismatch")
                for a, b in zip(physical, recorded):
                    expected = {"action": a["action"][1:], "position": [a["position"]["x"], a["position"]["y"]],
                                "channel": a["channel"], "virtual_time_s": a["response"]["virtual_time_s"],
                                "result": a["response"].get("measure_result", a["response"].get("clear_result"))}
                    if any(b.get(k) != v for k, v in expected.items()):
                        errors.append("Policy/physical action content mismatch")
                    if expected["result"] == "direction" and b.get("bearing_deg") != a["response"].get("svd_deg"):
                        errors.append("Policy/physical bearing mismatch")
                prepared = deepcopy(report)
                # Missing inference metadata means no inferences for these
                # baseline/geometry policies, not an invented physical record.
                prepared["strategy_parameters"].setdefault("inferred_no_signal_constraints", [])
                try:
                    reconstructed = audit(prepared)
                except (AssertionError, KeyError, ValueError) as error:
                    errors.append(f"Geometry/coverage replay: {type(error).__name__}: {error}")
                    reconstructed = None
            rows.append({"policy": policy["name"], "case_id": data["row"]["case_id"],
                         "case_sha256": data["row"]["case_sha256"],
                         "source_trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "errors": errors, "reconstructed_geometry_and_inference": reconstructed})
    summary = {"scope": "training_pressure_legal_observation_replay", "traces": len(rows),
               "passed_traces": sum(not r["errors"] for r in rows),
               "errors": sum(len(r["errors"]) for r in rows),
               "recertified_inferences": sum((r["reconstructed_geometry_and_inference"] or {}).get("inferences", 0) for r in rows),
               "rows": rows}
    (directory / "history_audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    run(parser.parse_args().directory)
