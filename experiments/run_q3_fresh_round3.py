"""Third-round offline paired runner. No official simulator entry points."""
import argparse
from dataclasses import replace
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from experiments.run_q3_fresh import StressEngine
from experiments.run_q3_fresh_round2 import all_lower_bounds, _fingerprint, _load_cases, summarize, write_snapshot, source_hashes as previous_hashes, pressure_suite
from simulation.cases import random_scenario
from strategies.q3_fresh_stepper import StepperQ3
from strategies.q3_fresh_rollout import FreshRollout
from strategies.q3_round3 import Round3Stepper, Round3Rollout, OldLoggedRollout
CONFIGS=('B','CR','G1','G12','G3','G3W','G3V')


def make_policy(name,client,log_root=None):
    directory=Path(log_root)/name if log_root is not None else None
    if name=='B':return StepperQ3(client),None
    if name=='CR':return StepperQ3(client,movable_tail=True),OldLoggedRollout(log_dir=directory)
    if name=='G1':return StepperQ3(client,movable_tail=True),Round3Rollout(log_dir=directory)
    if name=='G12':return Round3Stepper(client),Round3Rollout(evaluate_cover=True,log_dir=directory)
    from strategies.q3_round3_exploration import ExplorationRollout
    return Round3Stepper(client),ExplorationRollout(evaluate_cover=True,events=True,directions=True,
        witness_mode={'G3':'off','G3W':'record','G3V':'veto'}[name],log_dir=directory)


def source_hashes():
    result=previous_hashes()
    for folder in ('src/strategies','experiments'):
        for path in (ROOT/folder).glob('*round3*.py'):
            result[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def build_suite(suite,count=None):
    if suite=='development':
        return _load_cases(ROOT/'research/q3_fresh_round3/development_input.json')
    if suite=='nominal':
        count=128 if count is None else count
        cases=[random_scenario(3,1003000+i) for i in range(count)]
        return cases,dict(suite_groups={c.case_id:'nominal' for c in cases})
    count=96 if count is None else count
    if count%4:raise ValueError('Pressure count must contain whole four-variant base maps')
    cases=[];groups={}
    for group in range(count//4):
        seed=1006000+100*group
        index=(1,3,5,7,9,11)[group%6]
        base=pressure_suite(seed,(7 if index==11 else index)+1)[-1]
        if index==11:base=replace(base,error_mode='uniform',description='smooth_spatial_error')
        for n in (14,16):
            for rotation in (0,60):
                theta=math.radians(rotation);co,si=math.cos(theta),math.sin(theta)
                sources=tuple(replace(s,x=s.x*co-s.y*si,y=s.x*si+s.y*co) for s in base.sources[:n])
                case=replace(base,case_id=f'q3-r3-pressure-{group:03d}-n{n}-rot{rotation}',sources=sources)
                cases.append(case);groups[case.case_id]=f'q3-r3-pressure-base-{group:03d}'
    return cases,dict(suite_groups={c.case_id:'pressure' for c in cases},original_case_groups=groups)


def run_case(case, names, *, anchors=(), group="unspecified", original_case_group=None,
             uncertainty=None, on_result=None, log_root=None):
    rows, traces = [], {}
    # Evaluator-only calculations: neither the bounds nor ground truth are
    # passed to the controller or its planner.
    bounds = all_lower_bounds(case)
    robust = all_lower_bounds(case, uncertainty) if uncertainty is not None else None
    for name in names:
        engine = StressEngine(case, anchors)
        controller, planner = make_policy(name, engine.client(), log_root)
        result, error = {}, None
        wall_start, cpu_start = time.perf_counter(), time.process_time()
        try:
            result = controller.run(planner=planner)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            engine.finish_for_evaluation("policy_error")
            # Preserve the charged trajectory even if no success result exists.
            result = controller.result()
        wall, cpu = time.perf_counter() - wall_start, time.process_time() - cpu_start
        metrics = engine.evaluation()
        truth = metrics.pop("ground_truth")
        certified = bool(result.get("completion_certified", False))
        exited = metrics["simulator_stop_reason"] == "exited"
        # Engine uses its actual session enum; accepted /exit is independently
        # recognized from the legal action history as well.
        observations = engine.observation_history()
        exited = exited or bool(observations and observations[-1]["action"] == "/exit")
        success = bool(error is None and metrics["all_cleared"] and certified and exited)
        row = dict(case_id=case.case_id, strategy=name, suite_group=group,
                   original_case_group=original_case_group or case.case_id,
                   scenario_sha256=_fingerprint(truth), error=error,
                   completion_certified=certified, correct_exit=success,
                   incorrect_exit=bool(exited and not (metrics["all_cleared"] and certified)),
                   success=success, policy_wall_s=wall, policy_cpu_s=cpu,
                   planner_stats=getattr(planner, "stats", {}) if planner else {},
                   confirmation_tail_s=result.get("confirmation_tail_s") if success else None,
                   time_per_source_s=metrics["virtual_time_s"] / len(case.sources),
                   **{k: v for k, v in metrics.items() if k != "case_id"}, **bounds)
        for label in ("physical", "certified", "guarantee"):
            row[f"time_over_{label}_lower_bound"] = row["virtual_time_s"] / bounds[f"{label}_lower_bound_s"]
        if robust:
            for label in ("physical", "certified", "guarantee"):
                value = robust[f"{label}_lower_bound_s"]
                row[f"observation_robust_{label}_lower_bound_s"] = value
                row[f"time_over_observation_robust_{label}_lower_bound"] = row["virtual_time_s"] / value
        if success and row["virtual_time_s"] + 1e-5 < bounds["guarantee_lower_bound_s"]:
            row.update(error="AssertionError: successful execution is below its guarantee lower bound", success=False)
        trace = dict(search=result, observations=observations, evaluation=metrics,
                     scenario=truth, row=row, planner_stats=row["planner_stats"])
        rows.append(row)
        traces[name] = trace
        if on_result is not None:
            on_result(row, trace)
    return rows, traces


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=('development','nominal','pressure'),default='development')
    p.add_argument('--cases',type=Path)
    p.add_argument('--count',type=int)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--configs',choices=CONFIGS,nargs='+',required=True)
    a=p.parse_args()
    if len(set(a.configs))!=len(a.configs):p.error('Duplicate configuration')
    cases,metadata=_load_cases(a.cases) if a.cases else build_suite(a.suite,a.count)
    a.out.mkdir(parents=True,exist_ok=False)
    hashes=source_hashes();wall=time.perf_counter();cpu=time.process_time();rows=[]
    manifest=dict(kind='offline_round3',suite='external' if a.cases else a.suite,cases=len(cases),configs=a.configs,
        status='running',completed_cases=0,completed_runs=0,source_sha256=hashes,case_ids=[c.case_id for c in cases],
        case_definition_sha256=_fingerprint([c.evaluation_config() for c in cases]),metadata=metadata,
        validation_labels_used_for_fitting=False,total_cpu_s=0,total_wall_s=0,max_concurrent_processes=1)
    write_snapshot(a.out,rows,manifest)
    for index,case in enumerate(cases):
        def save(row,trace):
            rows.append(row)
            with gzip.open(a.out/f"trace-{index:03d}-{row['strategy']}.json.gz",'wt',encoding='utf-8',compresslevel=1) as f:
                json.dump(trace,f,ensure_ascii=False,allow_nan=False)
            manifest.update(completed_runs=len(rows),total_cpu_s=time.process_time()-cpu,total_wall_s=time.perf_counter()-wall)
            write_snapshot(a.out,rows,manifest)
            print(json.dumps({k:row[k] for k in ('case_id','strategy','success','virtual_time_s','physical_lower_bound_s','guarantee_lower_bound_s','time_over_physical_lower_bound','time_over_guarantee_lower_bound','error')}),flush=True)
        run_case(case,a.configs,anchors=metadata.get('anchors',{}).get(case.case_id,()),
            group=metadata.get('suite_groups',{}).get(case.case_id,'reconstruction'),
            original_case_group=metadata.get('original_case_groups',{}).get(case.case_id,case.case_id.split('-radius-')[0]),
            uncertainty=metadata.get('uncertainty_radii',{}).get(case.case_id),on_result=save,
            log_root=a.out/f'branches-{index:03d}')
        manifest['completed_cases']=index+1
    manifest.update(status='completed',source_unchanged=hashes==source_hashes(),total_cpu_s=time.process_time()-cpu,total_wall_s=time.perf_counter()-wall)
    write_snapshot(a.out,rows,manifest)


if __name__=='__main__':main()
