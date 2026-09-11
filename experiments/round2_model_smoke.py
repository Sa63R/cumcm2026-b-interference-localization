"""Generic implementation smoke on explicitly already-opened Q3 development.

No database, network or formal simulator. A new process per invocation avoids
cross-worktree module caches. This is implementation evidence, not independent
performance evidence. Truth is requested only after an accepted explicit exit.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "research/round2/pilot_audit.json"
OPENED_SEEDS = tuple(range(200101,200117)) + tuple(range(203101,203117))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
        separators=(",",":"),allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def source_hashes(repository):
    return {path.relative_to(repository).as_posix():sha(path)
            for path in sorted((repository/"src").rglob("*.py"))}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repository",type=Path,required=True)
    result.add_argument("--spec",type=Path,required=True)
    result.add_argument("--seed",type=int,choices=OPENED_SEEDS,default=200114)
    result.add_argument("--output",type=Path,required=True)
    result.add_argument("--max-searches-one",action="store_true")
    result.add_argument("--disabled",action="store_true")
    return result


def validate_arguments(args):
    """Pure file/argument checks: does not import policy or construct a case."""
    repository = args.repository.resolve(strict=True)
    if repository.parent != ROOT.parent or not (repository/"src").is_dir():
        raise ValueError("repository must be a sibling project tree with src")
    if args.spec.is_absolute() or args.spec.drive:
        raise ValueError("spec must be a relative path inside the repository")
    spec_path = (repository/args.spec).resolve(strict=True)
    if not spec_path.is_relative_to(repository) or spec_path.suffix.lower() != ".json":
        raise ValueError("spec must be a JSON file inside the repository")
    output = args.output.resolve()
    if output.exists() or output.suffix.lower() != ".gz":
        raise ValueError("output must be a NEW gzip path; overwrite is forbidden")
    if args.seed not in OPENED_SEEDS:
        raise ValueError("seed is not in the explicitly already-opened allowlist")
    spec = read(spec_path)
    entry = spec.get("entrypoint","").split(":")
    if (len(entry) != 2 or not entry[0].startswith("strategies.")
            or not all(part.isidentifier() for part in entry[0].split("."))
            or not entry[1].isidentifier()):
        raise ValueError("spec must name a strategies.module:function entrypoint")
    raw_kwargs = spec.get("kwargs",{})
    if not isinstance(raw_kwargs,dict):
        raise ValueError("spec kwargs must be an object")
    effective = deepcopy(raw_kwargs)
    if effective.get("problem",3) != 3:
        raise ValueError("implementation smoke is Q3 only")
    effective.setdefault("problem",3)
    effective.setdefault("max_actions",10000)
    if args.max_searches_one:
        if not isinstance(effective.get("tree_config"),dict):
            raise ValueError("--max-searches-one requires an explicit tree_config")
        effective["tree_config"]["max_searches"] = 1
    if args.disabled:
        effective["enabled"] = False
    audit = read(AUDIT)
    if audit.get("version") != "q3-round2-posthoc-original-physical-v1":
        raise ValueError("expected the named original-LB audit format")
    supported = sorted(seed for seed in OPENED_SEEDS if str(seed) in audit.get("bounds_by_seed",{}))
    if args.seed not in supported:
        raise ValueError("seed is opened but has no LB in research/round2/pilot_audit.json; no case will be generated")
    bound = audit["bounds_by_seed"][str(args.seed)]
    if (not isinstance(bound.get("case_sha256"),str) or len(bound["case_sha256"]) != 64
            or bound.get("physical_clairvoyant_lower_s",0) <= 0):
        raise ValueError("named audit lacks a valid case identity/lower bound")
    audited = [row for row in audit["rows"] if row["seed"] == args.seed
               and row["case_sha256"] == bound["case_sha256"] and row["audit_passed"]]
    if not audited:
        raise ValueError("named bound has no passed case audit")
    helper = repository/"experiments/training_stress_reliability.py"
    if not helper.is_file():
        raise ValueError("repository is missing the observation wrapper/auditor helper")
    return {"repository":repository,"spec_path":spec_path,"output":output,"spec":spec,
            "effective_kwargs":effective,"bound":bound,"audit":audit,"helper_path":helper,
            "supported_by_available_audit":supported,"entrypoint":entry}


def run(args, checked):
    repository, spec_path = checked["repository"], checked["spec_path"]
    before = source_hashes(repository)
    if not before:
        raise ValueError("empty candidate source set")
    harness_before, spec_before, audit_before = sha(__file__),sha(spec_path),sha(AUDIT)
    helper_before = sha(checked["helper_path"])
    # Fail rather than silently reuse imported policy modules from another tree.
    prefixes = ("simulation","simulator_client","strategies","planning","geometry","localization")
    for name,module in list(sys.modules.items()):
        if name.split(".")[0] in prefixes and getattr(module,"__file__",None):
            if not Path(module.__file__).resolve().is_relative_to(repository/"src"):
                raise ValueError("run this harness in a fresh process; another tree is already imported")
    sys.path[:0] = [str(repository/"src"),str(repository)]
    helper_spec = importlib.util.spec_from_file_location("_round2_model_smoke_helper",checked["helper_path"])
    helper = importlib.util.module_from_spec(helper_spec)
    sys.modules[helper_spec.name] = helper
    helper_spec.loader.exec_module(helper)
    module = importlib.import_module(checked["entrypoint"][0])
    if not Path(module.__file__).resolve().is_relative_to(repository/"src"):
        raise ValueError("entrypoint did not load from the selected repository")
    callback = getattr(module,checked["entrypoint"][1])
    # Check kwargs before generating the already-opened development case.
    inspect.signature(callback).bind(None,**checked["effective_kwargs"])
    from simulation import LocalResearchSimulator, random_scenario
    simulator = LocalResearchSimulator(random_scenario(3,args.seed),
        max_real_duration_s=1200,max_virtual_duration_s=360000)
    client = helper.ObservationOnlyClient(simulator.client())
    report, errors, exception = None, [], None
    started, cpu_started = time.perf_counter(),time.process_time()
    try:
        report = callback(client,**checked["effective_kwargs"])
        errors.extend(str(x) for x in (report.error,report.exit_error) if x)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        exception = traceback.format_exc()
    finally:
        if client.state.session == "active" and client.pending_request is None:
            try:
                if client.exit().get("accepted") is not True:
                    errors.append("Cleanup exit was rejected")
            except Exception as exc:
                errors.append(f"CleanupExit: {type(exc).__name__}: {exc}")
    elapsed, cpu = time.perf_counter()-started,time.process_time()-cpu_started
    accepted_exit = client.state.session == "exited" and client.pending_request is None
    history = simulator.observation_history()
    accepted_exit = accepted_exit and bool(history and history[-1]["action"] == "/exit"
                                           and history[-1]["response"].get("accepted") is True)
    evaluation, audit_result, actual_case_sha, case_matches = None,None,None,False
    if accepted_exit:
        # First and only truth access: after policy termination AND actual exit.
        simulator.finish_for_evaluation()
        evaluation = simulator.evaluation()
        actual_case_sha = digest(evaluation["ground_truth"])
        case_matches = actual_case_sha == checked["bound"]["case_sha256"]
        if not case_matches:
            errors.append("Generated case differs from the already-opened audited case")
        try:
            audit_result = helper.audit_actions(history,evaluation)
            errors.extend(audit_result["errors"])
        except Exception as exc:
            errors.append(f"ActionAudit: {type(exc).__name__}: {exc}")
        if report and (report.cleared_count != evaluation["cleared_total"]
                       or abs(report.virtual_time_s-evaluation["virtual_time_s"]) > 1e-6):
            errors.append("Report differs from post-exit evaluation")
    else:
        errors.append("No accepted explicit exit; evaluation truth was not requested")
    after = source_hashes(repository)
    spec_after, harness_after, helper_after, audit_after = sha(spec_path),sha(__file__),sha(checked["helper_path"]),sha(AUDIT)
    immutable = (before == after and spec_before == spec_after and harness_before == harness_after
                 and helper_before == helper_after and audit_before == audit_after)
    if not immutable:
        errors.append("Source, spec, harness, helper or bound audit changed during smoke")
    success = bool(report and report.completion_certified_under_model and evaluation
                   and evaluation["all_cleared"] and evaluation["failed_clear_count"] == 0
                   and accepted_exit and case_matches and not errors)
    lower = checked["bound"]["physical_clairvoyant_lower_s"] if case_matches else None
    record = {"scope":f"Unfrozen implementation smoke on already-opened development {args.seed}; not independent evidence",
        "repository":str(repository),"entrypoint":checked["spec"]["entrypoint"],
        "arguments":{"repository":str(args.repository),"spec":str(args.spec),"seed":args.seed,
                     "output":str(checked["output"]),"max_searches_one":args.max_searches_one,"disabled":args.disabled},
        "original_spec":checked["spec"],"original_spec_file_sha256":spec_before,
        "original_spec_canonical_sha256":digest(checked["spec"]),"spec_after_sha256":spec_after,
        "effective_kwargs":checked["effective_kwargs"],"limits":{"real_s":1200,"virtual_s":360000},
        "source_sha256":before,"source_after_sha256":after,"source_unchanged_during_smoke":before==after,
        "harness_sha256":harness_before,"harness_after_sha256":harness_after,
        "evaluation_helper_sha256":helper_before,"evaluation_helper_after_sha256":helper_after,
        "audit_input":{"path":str(AUDIT),"sha256":audit_before,"after_sha256":audit_after,
                       "supported_by_available_audit":checked["supported_by_available_audit"],
                       "case_sha256":checked["bound"]["case_sha256"],"actual_case_sha256":actual_case_sha,
                       "case_identity_matches":case_matches},
        "summary":report.as_dict() if report else None,"evaluation":evaluation,"history":history,"audit":audit_result,
        "evaluation_phase":"after_accepted_exit" if accepted_exit else "not_requested",
        "accepted_exit":accepted_exit,"elapsed_seconds":elapsed,"program_cpu_s":cpu,
        "physical_clairvoyant_lower_s":lower,"ratio_eligible":success,
        "time_over_physical_lower":evaluation["virtual_time_s"]/lower if success else None,
        "successful":success,"errors":errors,"exception_traceback":exception,
        "all_inputs_unchanged_during_smoke":immutable,"sqlite_read":False,"network_access":False}
    checked["output"].parent.mkdir(parents=True,exist_ok=True)
    # Exclusive create protects against a second process claiming this filename.
    with checked["output"].open("xb") as raw:
        with gzip.GzipFile(fileobj=raw,mode="wb",mtime=0) as zipped:
            zipped.write(json.dumps(record,ensure_ascii=False,allow_nan=False).encode("utf-8"))
    print(json.dumps({"successful":success,"accepted_exit":accepted_exit,
        "virtual_time_s":evaluation["virtual_time_s"] if evaluation else None,
        "physical_lower_s":lower,"time_over_lower":record["time_over_physical_lower"],
        "elapsed_seconds":elapsed,"output":str(checked["output"]),"errors":errors},ensure_ascii=False))
    return 0 if success else 1


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        checked = validate_arguments(args)
    except (ValueError,OSError,KeyError,TypeError) as exc:
        cli.error(str(exc))
    raise SystemExit(run(args,checked))


if __name__ == "__main__":
    main()
