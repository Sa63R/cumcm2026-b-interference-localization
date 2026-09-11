"""Evaluate one frozen RL policy on explicit, shared environment cases.

This outer evaluator owns scene construction. The policy receives only the
observation client, never case configuration, identity, seed, or lower bound.
Run in a separate interpreter: research-tree modules must not shadow the frozen
RL package. No network or simulator interface is used.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure(source_root, reference, checkpoint):
    source_root = Path(source_root).resolve(strict=True)
    for relative, expected in reference["source_sha256"].items():
        source = source_root / relative
        if not source.resolve().is_relative_to(source_root) or file_hash(source) != expected:
            raise ValueError("Frozen RL source mismatch: " + relative)
    if file_hash(checkpoint) != reference["checkpoint_sha256"]:
        raise ValueError("Frozen RL checkpoint changed")
    # Keep Python/installed dependencies, remove this script's research tree.
    current_root = Path(__file__).resolve().parents[1]
    sys.path[:] = [str(source_root / "src"), str(source_root)] + [
        p for p in sys.path if p and not Path(p).resolve().is_relative_to(current_root)]
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    return source_root


def evaluate_case(item, *, policy, reference, reference_arm=False):
    from simulation import LocalResearchSimulator
    from simulation.cases import Source
    from q4_rl.scenarios import Q4Scenario
    from experiments.run_q4_state_study import ObservationOnlyClient, digest
    from experiments.q4_rl_evaluate import audit_record
    from experiments.q4_comparison_bounds import common_bound
    config = dict(item["environment"])
    config["sources"] = tuple(Source(**source) for source in config["sources"])
    case = Q4Scenario(**config)
    if digest(case.evaluation_config()) != item["case_sha256"]:
        raise ValueError("Explicit environment differs from paired case identity")
    sim = LocalResearchSimulator(case)
    client = ObservationOnlyClient(sim.client())
    result, errors = None, []
    started = time.perf_counter()
    try:
        if reference_arm:
            from strategies.q4_clear_before_probe import run_q4_clear_before_probe
            result = run_q4_clear_before_probe(client, problem=4, max_actions=20000,
                max_active_probes=6, max_expansions=200)
        else:
            from q4_rl.controller import run_q4_rl
            policy.records.clear()
            result = run_q4_rl(client, policy=policy, problem=4, max_actions=20000,
                max_active_probes=6, max_expansions=200, max_decisions=512,
                record_transitions=False)
        errors.extend(str(e) for e in (result.error, result.exit_error) if e)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if client.state.session == "active" and client.pending_request is None:
            try:
                client.exit()
            except Exception as exc:
                errors.append(f"CleanupExit: {exc}")
        sim.finish_for_evaluation()
    elapsed = time.perf_counter()-started
    evaluation, history = sim.evaluation(), sim.observation_history()
    if digest(evaluation["ground_truth"]) != item["case_sha256"]:
        raise ValueError("Paired environment changed during execution")
    certified = bool(result and result.completion_certified_under_model)
    exited = client.state.session == "exited" and client.pending_request is None
    if result and (result.cleared_count != evaluation["cleared_total"] or
                   abs(result.virtual_time_s-evaluation["virtual_time_s"]) > 1e-6):
        errors.append("Policy/evaluator outcome mismatch")
    success = bool(certified and exited and evaluation["all_cleared"] and not errors)
    row = dict(case_id=case.case_id, seed=case.seed, stage=item["stage"], problem=4,
        strategy="compact_frozen_r8_check" if reference_arm else "compact_macro_ppo512",
        case_sha256=item["case_sha256"], successful=success,
        all_cleared=evaluation["all_cleared"], completion_certified=certified, accepted_exit=exited,
        source_total=evaluation["source_total"], cleared_total=evaluation["cleared_total"],
        virtual_time_s=evaluation["virtual_time_s"],
        penalized_time_s=evaluation["virtual_time_s"] if success else 360000.,
        program_runtime_s=elapsed, measurement_count=evaluation["measurement_count"],
        failed_clear_count=evaluation["failed_clear_count"], action_count=evaluation["action_count"],
        errors=errors, **evaluation["time_breakdown_s"])
    record = dict(row=row, summary=result.as_dict() if result else None, evaluation=evaluation,
        history=history, evaluation_phase="after_policy_termination",
        spec=({"entrypoint": "strategies.q4_clear_before_probe:run_q4_clear_before_probe",
               "kwargs": {"max_expansions": 200}} if reference_arm else reference["spec"]),
        policy_reference_sha256=item["reference_sha256"])
    record["audit"] = audit_record(record)
    if not record["audit"]["passed"]:
        row["successful"] = False
        row["penalized_time_s"] = 360000.
        errors.extend(record["audit"]["errors"])
    bound = common_bound(evaluation["ground_truth"])
    lower = bound["common_lower_bound_s"]
    if abs(lower-item["common_lower_bound_s"]) > 1e-8:
        raise ValueError("Historical lower bound differs between arms")
    record["common_lower_bound"] = bound
    row.update(common_lower_bound_s=lower,
        time_over_lower_bound=row["virtual_time_s"]/lower,
        penalized_time_over_lower_bound=row["penalized_time_s"]/lower,
        audit_passed=record["audit"]["passed"])
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-arm", action="store_true")
    args = parser.parse_args()
    reference = json.loads(args.reference.read_bytes())
    configure(args.source_root, reference, args.checkpoint)
    policy = None
    if not args.reference_arm:
        from q4_rl.network import load_policy
        policy = load_policy(args.checkpoint, deterministic=True)
    payload = json.loads(args.input.read_bytes())
    reference_hash = file_hash(args.reference)
    if payload["worker_sha256"] != file_hash(__file__):
        raise ValueError("Outer evaluator changed after freeze")
    args.output.mkdir(parents=True, exist_ok=False)
    for item in payload["cases"]:
        if item["reference_sha256"] != reference_hash:
            raise ValueError("Mismatched policy reference")
        record = evaluate_case(item, policy=policy, reference=reference,
                               reference_arm=args.reference_arm)
        path = args.output / (f"{record['row']['strategy']}-{item['environment']['seed']}-"
                              f"{item['case_sha256'][:12]}.json.gz")
        if path.exists():
            raise ValueError("Duplicate case would overwrite prior evidence")
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, allow_nan=False)
        row = record["row"]
        print(json.dumps({key: row[key] for key in ("seed", "strategy", "successful",
              "virtual_time_s", "common_lower_bound_s", "time_over_lower_bound")}), flush=True)


if __name__ == "__main__":
    main()
