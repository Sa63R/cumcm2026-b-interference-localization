"""Frozen, paired, CPU-only Q4 RL evaluation. Default: development only.

Policy factories and model files are explicit in JSON specs. No database,
official simulator, network transfer or automatic model promotion is invoked.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from experiments.run_q4_state_study import ObservationOnlyClient, digest, write_json
from experiments.q4_comparison_bounds import common_bound, FAILURE_PENALTY_S
from q4_rl.scenarios import build_case, case_requests, FAMILIES, SOURCE_MODES
from simulation.engine import LocalResearchSimulator

PROTOCOL_PATH = ROOT / "research/q4_rl/protocol.json"
COMPONENTS = ("movement_s", "switching_s", "detection_s", "optical_s", "removal_s")
DEFAULT_SPECS = {
    "state": {"entrypoint": "strategies.q4_state_search:run_q4_state_search", "kwargs": {"mode": "state_pruned", "max_expansions": 200}},
    "compact_cover": {"entrypoint": "strategies.q4_cover_search:run_q4_cover_search", "kwargs": {"profile": "compact_22", "schedule": "joint", "max_expansions": 200}},
    "combo": {"entrypoint": "strategies.q4_range_scheduling:run_q4_range_scheduling", "kwargs": {"config": "onroute", "max_expansions": 200}},
    "r8": {"entrypoint": "strategies.q4_clear_before_probe:run_q4_clear_before_probe", "kwargs": {"config": "center_once", "max_expansions": 200}},
    "heuristic": {"entrypoint": "q4_rl.controller:run_q4_rl", "kwargs": {"record_transitions": False}},
}


def entrypoint(value):
    module, name = value.split(":", 1)
    return getattr(importlib.import_module(module), name)


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process_peak_rss_bytes():
    """Process-lifetime high-water mark, explicitly not per-episode allocation."""
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except ImportError:
        try:
            import psutil
            memory = psutil.Process().memory_info()
            return int(memory.peak_wset) if hasattr(memory, "peak_wset") else None
        except (ImportError, OSError):
            return None


def source_hashes():
    files = list((ROOT / "src").rglob("*.py"))
    files += list((ROOT / "experiments").glob("audit_q4_*.py"))
    files += [Path(__file__), PROTOCOL_PATH, ROOT / "experiments/q4_comparison_bounds.py",
              ROOT / "experiments/run_q4_state_study.py", ROOT / "experiments/run_study.py"]
    return {p.relative_to(ROOT).as_posix(): file_digest(p) for p in sorted(set(files))}


def artifact_hashes(specs):
    """Explicit relative files include model/checkpoint and original JSON config."""
    output = {}
    for spec in specs.values():
        for name in spec.get("artifact_files", []):
            path = (ROOT / name).resolve()
            try:
                relative = path.relative_to(ROOT).as_posix()
            except ValueError as exc:
                raise ValueError("Artifacts must reside inside the independent task tree") from exc
            output[relative] = file_digest(path)
    return output


def validate_specs(specs):
    if not isinstance(specs, dict) or not specs:
        raise ValueError("Nonempty strategy specifications required")
    for label, spec in specs.items():
        if not isinstance(label, str) or not label or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in label):
            raise ValueError("Unsafe strategy label")
        if not isinstance(spec, dict) or not isinstance(spec.get("entrypoint"), str):
            raise ValueError("Dynamic strategy entrypoint required")
        if not isinstance(spec.get("kwargs", {}), dict):
            raise ValueError("Strategy kwargs must be an object")
        if label in ("state", "compact_cover", "combo", "r8") and spec != DEFAULT_SPECS[label]:
            raise ValueError("Frozen control configuration changed")
        if label == "learned" and (not spec.get("policy_factory") or not spec.get("artifact_files")):
            raise ValueError("Learned strategy needs a policy factory and hashed model artifacts")
        if spec.get("policy_factory"):
            factory = spec["policy_factory"]
            checkpoint = factory.get("kwargs", {}).get("checkpoint")
            if not checkpoint or checkpoint not in spec.get("artifact_files", []):
                raise ValueError("Policy factory checkpoint must be an explicitly hashed artifact")
    return specs


def completion_audit(history):
    """Observation-only certificate; never receives scenario or evaluation truth."""
    from planning.q4_directional_cover import certify_directional_cover, verify_directional_cover_certificate
    discovered, cleared = set(), set()
    negatives = {c: set() for c in range(1, 21)}
    for action in history:
        if action["response"].get("accepted") is not True:
            raise ValueError("Unaccepted action in physical history")
        c, response = action.get("channel"), action["response"]
        if action["action"] == "/measure":
            if response["measure_result"] == "no_signal" and c not in discovered and c not in cleared:
                p = action["position"]
                negatives[c].add((p["x"], p["y"]))
            elif response["measure_result"] in ("near", "direction"):
                discovered.add(c)
        elif action["action"] == "/clear" and response["clear_result"] == "success":
            cleared.add(c)
    if len(cleared) == 16:
        return {"passed": True, "reason": "sixteen_actual_clears", "cleared": 16, "coverage": {}}
    if not 10 <= len(cleared) <= 16 or not discovered <= cleared:
        return {"passed": False, "reason": "unresolved_or_too_few_clears", "cleared": len(cleared), "coverage": {}}
    cache, cover = {}, {}
    for channel in sorted(set(range(1, 21)) - cleared):
        points = tuple(sorted(negatives[channel]))
        if points not in cache:
            if len(points) < 3:
                cache[points] = {"passed": False, "status": "insufficient_actual_negative_measurements"}
            else:
                certificate = certify_directional_cover(points)
                if certificate["passed"]:
                    verify_directional_cover_certificate(points, certificate)
                cache[points] = certificate
        cover[str(channel)] = cache[points]
    return {"passed": all(c["passed"] for c in cover.values()),
            "reason": "actual_per_channel_directional_cover", "cleared": len(cleared), "coverage": cover}


def audit_record(record):
    from experiments.audit_q4_state import wire_audit
    errors, wire, certificate = [], None, None
    try:
        wire = wire_audit(record)
    except (ValueError, AssertionError, KeyError, TypeError) as exc:
        errors.append("Physical audit: " + str(exc))
    try:
        certificate = completion_audit(record["history"])
        if not certificate["passed"]:
            errors.append("No independently verified all-clear certificate")
    except (ValueError, AssertionError, KeyError, TypeError) as exc:
        errors.append("Completion audit: " + str(exc))
    return {"passed": not errors, "errors": errors, "physical": wire, "completion": certificate}


def run_one(request, label, spec, *, expected_sources=None, expected_artifacts=None):
    """One complete arm; hidden metadata is held only by this outer evaluator."""
    if expected_sources is not None and source_hashes() != expected_sources:
        raise ValueError("Source differs from evaluation freeze")
    current_artifacts = artifact_hashes({label: spec})
    if expected_artifacts is not None and any(expected_artifacts.get(k) != v for k,v in current_artifacts.items()):
        raise ValueError("Model/config differs from evaluation freeze")
    case = build_case(**request)
    sim = LocalResearchSimulator(case)
    client = ObservationOnlyClient(sim.client())
    result, errors = None, []
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    try:
        kwargs = dict(spec.get("kwargs", {}))
        factory = spec.get("policy_factory")
        if factory:
            factory_kwargs = dict(factory.get("kwargs", {}))
            factory_kwargs["checkpoint"] = ROOT / factory_kwargs["checkpoint"]
            kwargs["policy"] = entrypoint(factory["entrypoint"])(**factory_kwargs)
        result = entrypoint(spec["entrypoint"])(client, problem=4, max_actions=20000,
                    max_active_probes=6, **kwargs)
        errors.extend(str(e) for e in (result.error, result.exit_error) if e)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if client.state.session == "active" and client.pending_request is None:
            try:
                client.exit()
            except Exception as exc:
                errors.append("CleanupExit: " + str(exc))
        sim.finish_for_evaluation()
    elapsed, cpu = time.perf_counter() - started_wall, time.process_time() - started_cpu
    evaluation, history = sim.evaluation(), sim.observation_history()
    certified = bool(result and result.completion_certified_under_model)
    exited = client.state.session == "exited" and client.pending_request is None
    if result and (result.cleared_count != evaluation["cleared_total"] or
                   abs(result.virtual_time_s - evaluation["virtual_time_s"]) > 1e-6):
        errors.append("Strategy/evaluator outcome mismatch")
    success = bool(certified and exited and evaluation["all_cleared"] and not errors)
    learning = getattr(result, "learning", {}) if result is not None else {}
    row = dict(case_id=case.case_id, seed=request["seed"], problem=4, strategy=label,
        stage=request["split"], family=request["family"], source_mode=request["source_mode"],
        case_sha256=digest(evaluation["ground_truth"]), successful=success,
        all_cleared=evaluation["all_cleared"], completion_certified=certified, accepted_exit=exited,
        source_total=evaluation["source_total"], cleared_total=evaluation["cleared_total"],
        virtual_time_s=evaluation["virtual_time_s"],
        penalized_time_s=evaluation["virtual_time_s"] if success else FAILURE_PENALTY_S,
        program_runtime_s=elapsed, process_cpu_s=cpu, process_peak_rss_bytes=process_peak_rss_bytes(),
        inference_wall_s=learning.get("inference_wall_time_s",0.), feature_wall_s=learning.get("feature_wall_time_s",0.),
        fallback_virtual_s=learning.get("fallback_cost_s",0.),
        measurement_count=evaluation["measurement_count"],
        failed_clear_count=evaluation["failed_clear_count"], action_count=evaluation["action_count"],
        stop_reason=evaluation["simulator_stop_reason"], errors=errors, **evaluation["time_breakdown_s"])
    record = dict(row=row, summary=result.as_dict() if result else None, evaluation=evaluation,
                  history=history, spec=spec, request=request, evaluation_phase="after_policy_termination")
    before_audit = time.perf_counter()
    record["audit"] = audit_record(record)
    row["audit_runtime_s"] = time.perf_counter() - before_audit
    if not record["audit"]["passed"]:
        row["successful"] = False
        row["penalized_time_s"] = FAILURE_PENALTY_S
        row["errors"].extend(record["audit"]["errors"])
    before_bound = time.perf_counter()
    bound = common_bound(evaluation["ground_truth"])
    row["lower_bound_runtime_s"] = time.perf_counter()-before_bound
    if row["successful"] and row["virtual_time_s"] < bound["common_lower_bound_rounded_s"] - 1e-6:
        row["successful"] = False
        row["penalized_time_s"] = FAILURE_PENALTY_S
        row["errors"].append("Complete time below the conditional common lower bound")
    record["common_lower_bound"] = bound
    lower = bound["common_lower_bound_s"]
    row.update(common_lower_bound_s=lower, common_lower_bound_rounded_s=bound["common_lower_bound_rounded_s"],
        time_over_lower_bound=row["virtual_time_s"] / lower if row["successful"] else None,
        penalized_time_over_lower_bound=row["penalized_time_s"] / lower,
        audit_passed=record["audit"]["passed"])
    return record


def percentile(values, q):
    values = sorted(values)
    at = q * (len(values) - 1)
    low = math.floor(at)
    return values[low] * (1 - (at - low)) + values[math.ceil(at)] * (at - low)


def interval(values):
    return [percentile(values, .025), percentile(values, .975)]


def wilson_interval(successes, n):
    z, p = 1.959963984540054, successes / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return [max(0., center-half), min(1., center+half)]


def _summary(group):
    times = [r["penalized_time_s"] for r in group]
    successes = sum(r["successful"] for r in group)
    lower = sum(r["common_lower_bound_s"] for r in group)
    return dict(runs=len(group), successful=successes, all_clear_rate=successes/len(group),
        all_clear_rate_wilson_ci95=wilson_interval(successes, len(group)),
        failed_runs=len(group)-successes, failed_clear_total=sum(r["failed_clear_count"] for r in group),
        mean_failed_clears=statistics.mean(r["failed_clear_count"] for r in group),
        max_failed_clears=max(r["failed_clear_count"] for r in group),
        mean_actual_elapsed_time_s=statistics.mean(r["virtual_time_s"] for r in group),
        mean_penalized_time_s=statistics.mean(times), median_penalized_time_s=percentile(times,.5),
        p90_penalized_time_s=percentile(times,.9), p95_penalized_time_s=percentile(times,.95), max_penalized_time_s=max(times),
        mean_lower_bound_s=lower/len(group), ratio_of_sums=sum(times)/lower,
        mean_individual_penalized_ratio=statistics.mean(r["penalized_time_over_lower_bound"] for r in group),
        mean_wall_s=statistics.mean(r["program_runtime_s"] for r in group),
        p95_wall_s=percentile([r["program_runtime_s"] for r in group], .95),
        max_wall_s=max(r["program_runtime_s"] for r in group),
        mean_process_cpu_s=statistics.mean(r["process_cpu_s"] for r in group),
        total_process_cpu_s=sum(r["process_cpu_s"] for r in group),
        mean_audit_wall_s=statistics.mean(r["audit_runtime_s"] for r in group),
        mean_lower_bound_wall_s=statistics.mean(r.get("lower_bound_runtime_s",0.) for r in group),
        mean_inference_wall_s=statistics.mean(r.get("inference_wall_s",0.) for r in group),
        mean_feature_wall_s=statistics.mean(r.get("feature_wall_s",0.) for r in group),
        mean_fallback_virtual_s=statistics.mean(r.get("fallback_virtual_s",0.) for r in group),
        max_worker_lifetime_peak_rss_bytes=max((r["process_peak_rss_bytes"] for r in group if r.get("process_peak_rss_bytes") is not None),default=None),
        mean_measurements=statistics.mean(r["measurement_count"] for r in group),
        mean_components_s={k: statistics.mean(r[k] for r in group) for k in COMPONENTS},
        failure_records=[{k:r[k] for k in ("case_id","virtual_time_s","cleared_total","source_total","stop_reason","errors")} for r in group if not r["successful"]])


def _compare(group, baseline, *, samples, bootstrap_seed):
    pairs = [(baseline[r["case_id"]], r) for r in group]
    saved = [b["penalized_time_s"] - r["penalized_time_s"] for b,r in pairs]
    # Correlated stress transformations sharing a seed remain in one cluster.
    clusters = {}
    for b,r in pairs:
        clusters.setdefault(r["seed"], []).append((b,r))
    keys, rng = sorted(clusters), random.Random(bootstrap_seed)
    bootstrap_saving, bootstrap_p95, bootstrap_relative = [], [], []
    for _ in range(samples if len(keys) >= 2 else 0):
        selected = [pair for key in rng.choices(keys, k=len(keys)) for pair in clusters[key]]
        bt, ct = [b["penalized_time_s"] for b,r in selected], [r["penalized_time_s"] for b,r in selected]
        bootstrap_saving.append(statistics.mean(b-c for b,c in zip(bt,ct)))
        bootstrap_p95.append(percentile(bt,.95)-percentile(ct,.95))
        bootstrap_relative.append(1-sum(ct)/sum(bt))
    bt, ct = [b["penalized_time_s"] for b,r in pairs], [r["penalized_time_s"] for b,r in pairs]
    return dict(pairs=len(pairs), independent_seed_clusters=len(keys), all_complete=all(b["successful"] and r["successful"] for b,r in pairs),
        uncertainty_status="bootstrap_approximation" if len(keys)>=2 else "insufficient_independent_seed_clusters",
        mean_saved_s=statistics.mean(saved), mean_saving_ci95_s=interval(bootstrap_saving) if bootstrap_saving else None,
        relative_mean_saving=1-sum(ct)/sum(bt), relative_saving_ci95=interval(bootstrap_relative) if bootstrap_relative else None,
        p95_saving_s=percentile(bt,.95)-percentile(ct,.95), p95_saving_ci95_s=interval(bootstrap_p95) if bootstrap_p95 else None,
        p95_ratio=percentile(ct,.95)/percentile(bt,.95),
        wins=sum(v>1e-6 for v in saved), losses=sum(v < -1e-6 for v in saved), ties=sum(abs(v)<=1e-6 for v in saved),
        worst_regression_s=max(0.,-min(saved)),
        loss_records=[dict(case_id=r["case_id"], saved_s=v, baseline_time_s=b["virtual_time_s"], candidate_time_s=r["virtual_time_s"],
            lower_bound_s=r["common_lower_bound_s"], baseline_ratio=b["time_over_lower_bound"], candidate_ratio=r["time_over_lower_bound"]) for (b,r),v in zip(pairs,saved) if v < -1e-6])


def report_rows(rows, *, reference="r8", samples=2000, bootstrap_seed=4260911):
    if not rows or samples < 20:
        raise ValueError("Rows and at least 20 bootstrap replicates required")
    groups = {}
    for row in rows:
        groups.setdefault(row["strategy"], []).append(row)
        expected = row["virtual_time_s"] if row["successful"] else FAILURE_PENALTY_S
        if row["penalized_time_s"] != expected or (not row["successful"] and row["time_over_lower_bound"] is not None):
            raise ValueError("Failure penalty or all-clear ratio violates protocol")
        if not math.isclose(row["penalized_time_over_lower_bound"], expected/row["common_lower_bound_s"], rel_tol=1e-12):
            raise ValueError("Inconsistent ratio")
    if reference not in groups:
        raise ValueError("Reference strategy missing")
    baseline = {r["case_id"]:r for r in groups[reference]}
    for label,group in groups.items():
        if len(group) != len(baseline) or {r["case_id"] for r in group} != baseline.keys():
            raise ValueError("Missing or duplicated scenario/strategy pair")
        for r in group:
            b = baseline[r["case_id"]]
            if any(r[k] != b[k] for k in ("case_sha256","common_lower_bound_s","seed","family","source_mode","stage")):
                raise ValueError("Paired environment or denominator differs")
    strata = sorted({(r["source_mode"],r["family"]) for r in rows})
    result = {"reference":reference,"rows":sorted(rows,key=lambda r:(r["case_id"],r["strategy"])),
              "bootstrap":{"samples":samples,"seed":bootstrap_seed,"unit":"paired seed cluster"},
              "summaries":{},"paired":{},"strata":{}}
    for name,mode,family in [("all",None,None)]+[(m+"/"+f,m,f) for m,f in strata]:
        summaries, comparisons = {}, {}
        for label,group in groups.items():
            selected = group if mode is None else [r for r in group if r["source_mode"]==mode and r["family"]==family]
            summaries[label] = _summary(selected)
            if label != reference:
                comparisons[label] = _compare(selected, baseline, samples=samples, bootstrap_seed=bootstrap_seed)
        if name == "all":
            result["summaries"],result["paired"] = summaries,comparisons
        else:
            result["strata"][name] = {"summaries":summaries,"paired":comparisons}
    result["interpretation"] = "All-strata summary uses the frozen requested mixture; independent claims are assessed per predeclared stratum. Wilson intervals do not account for shared-seed stress dependence; use per-stratum intervals. Synthetic uncertainty does not cover official distribution mismatch."
    return result


def _set_cpu_environment():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = "1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--specs", type=Path)
    parser.add_argument("--strategies", nargs="+", default=["r8","heuristic"])
    parser.add_argument("--reference", default="r8")
    parser.add_argument("--stage", choices=("development","confirmation","final"), default="development")
    parser.add_argument("--start", type=int)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=["random"])
    parser.add_argument("--source-modes", nargs="+", choices=SOURCE_MODES, default=list(SOURCE_MODES))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 60 or args.bootstrap_samples < 20:
        parser.error("workers must be 1..60; bootstrap samples >=20")
    _set_cpu_environment()
    specs = validate_specs(json.loads(args.specs.read_bytes()) if args.specs else {k:DEFAULT_SPECS[k] for k in args.strategies})
    if args.reference not in specs:
        parser.error("Reference missing from selected strategies")
    requests = case_requests(args.stage,start=args.start,count=args.count,families=args.families,source_modes=args.source_modes)
    sources, artifacts = source_hashes(),artifact_hashes(specs)
    raw_spec_hash = file_digest(args.specs) if args.specs else None
    identity = dict(source_sha256=sources, artifact_sha256=artifacts, specs=specs,
                    raw_specs_sha256=raw_spec_hash, protocol_sha256=file_digest(PROTOCOL_PATH))
    if args.stage != "development":
        if args.selection is None:
            parser.error("Confirmation/final require a frozen --selection")
        selected = json.loads(args.selection.read_bytes())
        if any(selected.get(k) != v for k,v in identity.items()) or selected.get("reserved_requests",{}).get(args.stage) != requests:
            raise ValueError("Independent selection/source/model/config/reserved requests mismatch")
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    manifest = dict(**identity,stage=args.stage,requests=requests,reference=args.reference,workers=args.workers,
        bootstrap_samples=args.bootstrap_samples,bootstrap_seed=4260911,numerical_threads_per_worker=1,
        selection_sha256=file_digest(args.selection) if args.selection else None,
        runtime_scope="model load + local solver + environment; audit and lower-bound analysis separate",
        formal_simulator=False,practice_simulator=False)
    write_json(output/"manifest.json",manifest)
    write_json(output/"freeze.json",{"manifest_sha256":digest(manifest)})
    with zipfile.ZipFile(output/"source.zip","w",compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(ROOT/path,path)
    if args.specs:
        (output/"specs.original.json").write_bytes(args.specs.read_bytes())
    if args.freeze_only:
        print(json.dumps({"frozen":True,"stage":args.stage,"cases":len(requests),"arms":list(specs)}))
        return 0
    (output/"records").mkdir()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers,initializer=_set_cpu_environment) as pool:
        jobs = {pool.submit(run_one,request,label,spec,expected_sources=sources,expected_artifacts=artifacts):(request,label)
                for request in requests for label,spec in specs.items()}
        for future in as_completed(jobs):
            record = future.result()
            row = record["row"]
            name = row["case_id"]+"--"+row["strategy"]+".json.gz"
            with gzip.open(output/"records"/name,"wt",encoding="utf-8") as stream:
                json.dump(record,stream,ensure_ascii=False,allow_nan=False)
            rows.append(row)
            write_json(output/"progress.json",{"completed":len(rows),"expected":len(jobs),"latest":row})
            print(json.dumps({k:row[k] for k in ("case_id","strategy","successful","failed_clear_count","virtual_time_s","common_lower_bound_s","time_over_lower_bound","errors")}),flush=True)
    if source_hashes()!=sources or artifact_hashes(specs)!=artifacts:
        raise ValueError("Source/model/config changed during evaluation; keep raw records, no qualified summary")
    report = report_rows(rows,reference=args.reference,samples=args.bootstrap_samples)
    write_json(output/"summary.json",report)
    write_json(output/"evidence.json",{"record_sha256":{p.name:file_digest(p) for p in sorted((output/"records").glob("*.gz"))},
        "manifest_sha256":file_digest(output/"manifest.json"),"summary_sha256":file_digest(output/"summary.json")})
    print(json.dumps({"summaries":report["summaries"],"paired":report["paired"]}),flush=True)
    return 0 if all(r["successful"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
