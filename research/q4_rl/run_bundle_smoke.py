"""Two prespecified TRAIN cases, three rules, real feedback and common bounds."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

from experiments.q4_rl_evaluate import (
    ROOT, DEFAULT_SPECS, source_hashes, file_digest, run_one, report_rows,
)
from q4_rl.scenarios import FAMILIES


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/q4_rl/bundle-smoke-8006600")
    args = parser.parse_args()
    output = (ROOT/args.output).resolve()
    output.relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Fresh empty output required")
    # These two independently developed, unused modules are not imported by
    # any rule here. Freeze every actual controller/evaluator dependency.
    omitted = {"src/q4_rl/bundle_network.py", "src/q4_rl/bundle_train.py"}
    frozen = {name: sha for name, sha in source_hashes().items() if name not in omitted}
    script = Path(__file__).resolve().relative_to(ROOT).as_posix()
    frozen[script] = file_digest(ROOT/script)
    save(output/"source_freeze.json", dict(files=frozen,
        omitted_unused_parallel_modules=sorted(omitted)))
    with zipfile.ZipFile(output/"source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in frozen:
            archive.write(ROOT/name, name)
    specs = dict(r8=DEFAULT_SPECS["r8"], macro_rule=DEFAULT_SPECS["heuristic"],
        bundle_rule={"entrypoint": "q4_rl.bundle_controller:run_q4_bundle",
                     "kwargs": {"max_decisions": 128, "record_transitions": True}})
    requests = []
    for seed in (8006600, 8006601):
        offset = seed-8000000
        requests.append(dict(seed=seed, split="train", family=FAMILIES[offset % len(FAMILIES)],
            source_mode="mixed" if (offset//len(FAMILIES)) % 2 == 0 else "all_directional"))
    save(output/"requests.json", requests)
    save(output/"specs.json", specs)
    rows, decision_counts = [], []
    for request in requests:
        for label, spec in specs.items():
            if any(file_digest(ROOT/name) != sha for name, sha in frozen.items()):
                raise ValueError("Frozen dependency changed")
            record = run_one(request, label, spec)
            raw = json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            (output/f"{request['seed']}-{label}.json.gz").write_bytes(gzip.compress(raw, mtime=0))
            row = record["row"]
            rows.append(row)
            learning = (record["summary"] or {}).get("learning", {})
            decision_counts.append(dict(seed=request["seed"], strategy=label,
                decisions=learning.get("decisions"), action_counts=learning.get("action_counts")))
            if label == "bundle_rule":
                assert abs(sum(t["cost_s"] for t in learning["transitions"])-row["virtual_time_s"]) < 2e-5
                for event in learning["bundle_events"]:
                    actual = record["summary"]["action_history"][event["start_actual_action_count"]:event["end_actual_action_count"]]
                    assert event["measured_channels"] == [a["channel"] for a in actual if a["action"] == "measure"]
            print(json.dumps(dict(seed=request["seed"], strategy=label,
                all_cleared=row["all_cleared"], audit=row["audit_passed"],
                time_s=row["virtual_time_s"], lower_bound_s=row["common_lower_bound_s"],
                time_over_lower_bound=row["time_over_lower_bound"],
                failed_clear_count=row["failed_clear_count"],
                decisions=learning.get("decisions")), ensure_ascii=False), flush=True)
    summary = report_rows(rows, reference="r8", samples=1000)
    summary["scope"] = "Two prespecified TRAIN functionality cases only; not independent validation or promotion evidence"
    summary["decision_counts"] = decision_counts
    save(output/"summary.json", summary)
    assert all(file_digest(ROOT/name) == sha for name, sha in frozen.items())
    save(output/"evidence.json", dict(source_freeze_verified=True,
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()}))


if __name__ == "__main__":
    main()
