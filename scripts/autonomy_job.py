"""Bounded CPU training blocks, paired local evaluation and object-store sync."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from autonomy_runtime import constrain, inside, write


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    manifest = read(ROOT/"PACKAGE_MANIFEST.json")
    for name, expected in manifest["files"].items():
        path = inside(ROOT, name)
        if not path.is_file() or sha(path) != expected:
            raise ValueError("Package source/checkpoint differs: "+name)
    os.environ["Q3_SOURCE_COMMIT"] = manifest["source_commit"]
    return manifest


def no_network(*args, **kwargs):
    raise RuntimeError("Policy evaluation is offline; network connections prohibited")


def public_record(value):
    if isinstance(value, dict):
        return {k:public_record(v) for k,v in value.items()}
    if isinstance(value, list):
        return [public_record(v) for v in value]
    if isinstance(value, str):
        value=value.replace(str(ROOT), "<PROJECT>")
        value=re.sub(r'(?i)\b[A-Z]:[\\/][^\s\"<>]+', '<ABSOLUTE_PATH>', value)
        return re.sub(r'(?<![\w:])/(?:home|root|tmp|opt|usr|mnt|var)/[^\s\x27\"<>:,)\]]+', '<ABSOLUTE_PATH>', value)
    return value


def validate_freeze(frozen, identity):
    if (not isinstance(frozen,dict) or frozen.get("policy_identity") != identity
            or not isinstance(frozen.get("frozen_unix"),(int,float))):
        raise ValueError("Frozen policy/source/protocol identity differs")


def evaluate(args):
    # Each call is a fresh process, with a single committed policy source tree.
    package = verify()
    constrain(ROOT, 50)
    tree = ROOT/"reference_state" if args.reference_state else ROOT
    sys.path[:0] = [str(tree), str(tree/"src"), str(ROOT/"src")]
    from experiments.research_v1_eval import run_case, summarize, digest as case_digest
    from simulation import random_scenario
    from research_rl.cpu_runtime import require_cpu
    require_cpu("cpu")
    socket.socket.connect = no_network
    socket.socket.connect_ex = no_network
    socket.create_connection = no_network
    target = inside(ROOT, args.output)
    target.mkdir(parents=True, exist_ok=True)
    spec = read(ROOT/"reference_state/spec.json") if args.reference_state else {
        "name": args.label, "entrypoint": args.entrypoint,
        "kwargs": {"checkpoint": args.checkpoint, "device": "cpu", "num_threads": 1}}
    checkpoint_sha = sha(inside(ROOT,args.checkpoint)) if not args.reference_state else None
    protocol = read(ROOT/"research/autonomy/protocol.json")
    first, last = protocol[args.split+"_range"]
    policy_identity = dict(package_sha256=sha(ROOT/"PACKAGE_MANIFEST.json"),
                    protocol_sha256=sha(ROOT/"research/autonomy/protocol.json"),
                    checkpoint_sha256=checkpoint_sha, spec=spec)
    identity={**policy_identity,"first":first,"last":last}
    manifest_path = target/"manifest.json"
    if manifest_path.exists() and read(manifest_path) != identity:
        raise ValueError("Evaluation resume identity differs")
    if args.split in {"confirmation", "final"}:
        frozen = read(inside(ROOT,args.freeze_record)) if args.freeze_record else None
        validate_freeze(frozen,policy_identity)
        if frozen["frozen_unix"]> (manifest_path.stat().st_mtime if manifest_path.exists() else time.time()):
            raise ValueError("Selection must be frozen before independent evaluation starts")
    if not manifest_path.exists():write(manifest_path, identity)
    rows=[]
    for seed in range(first,last+1):
        file=target/f"case-{seed}.json.gz"
        case=random_scenario(3,seed)
        if file.exists():
            value=json.loads(gzip.decompress(file.read_bytes()))
            if (value.get("checkpoint_sha256") != checkpoint_sha or value["spec"] != spec
                    or value["row"]["case_sha256"] != case_digest(case.evaluation_config())):
                raise ValueError("Existing evaluation has a different policy")
        else:
            value=run_case(case,spec,protocol)
            value["checkpoint_sha256"]=checkpoint_sha
            value=public_record(value)
            temporary=file.with_suffix(file.suffix+".tmp")
            temporary.write_bytes(gzip.compress(json.dumps(value,ensure_ascii=False,allow_nan=False).encode(),mtime=0))
            temporary.replace(file)
        rows.append(value["row"])
    summary=summarize(rows,last-first+1)
    summary.update(checkpoint_sha256=checkpoint_sha,split=args.split)
    # Append audited original-bound statistics before exposing a result summary.
    sys.path.insert(0,str(ROOT/"research/theory_v1"))
    from audit_eval_bounds import VERSION,audit_record,digest,load_cache,physical_bounds
    cache_path=ROOT/"cpu_runs/physical_bound_cache.json"
    cache=load_cache(cache_path)
    audited=[]
    for seed in range(first,last+1):
        value=json.loads(gzip.decompress((target/f"case-{seed}.json.gz").read_bytes()))
        sources,_=audit_record(value)
        geometry=[[c,sources[c]["x"],sources[c]["y"]] for c in sorted(sources)]
        key=digest(dict(version=VERSION,geometry=geometry))
        if key not in cache:
            cache[key]=physical_bounds(sources.values())
        lower=cache[key]["physical_clairvoyant_lower_s"]
        row=value["row"]
        audited.append({**row,"physical_lower_bound_s":lower,
                        "time_over_lower":row["virtual_time_s"]/lower if row["successful"] else None})
    write(cache_path,dict(version=VERSION,entries=cache,entries_sha256=digest(cache)))
    summary.update(physical_mean_lower_s=statistics.mean(r["physical_lower_bound_s"] for r in audited),
        ratio_of_sums=(sum(r["virtual_time_s"] for r in audited)/sum(r["physical_lower_bound_s"] for r in audited)
                       if all(r["successful"] for r in audited) else None),
        audit_passed_records=len(audited),source_scope="Local synthetic worlds, exact paired case hashes; not official simulator results")
    write(target/"audited_rows.json",audited)
    write(target/"summary.json",summary)
    print(json.dumps({"evaluation":args.label,"summary":summary}),flush=True)


def _alive(pid,start):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")",1)[1].split()[19] == start
    except OSError:
        return False


def _group_alive(pid):
    try:os.killpg(pid,0)
    except ProcessLookupError:return False
    return True


ALGORITHM_ARGS = {
    "--feature-version", "--hidden", "--architecture", "--probe-candidates", "--group-alpha",
    "--bc-episodes", "--bc-epochs", "--episodes-per-update", "--lr", "--gae-lambda",
    "--entropy-coef", "--aux-bc-coef", "--epochs", "--minibatch", "--clip", "--value-coef",
    "--target-kl", "--max-grad-norm", "--max-decisions", "--memory-hidden", "--episodes-per-minibatch",
    "--groups-per-update", "--alternatives", "--gap-scale-s", "--max-pair-weight", "--kl-coef",
}


def validate_plan(plan):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", plan["name"]):
        raise ValueError("Invalid job name")
    if plan["trainer"] not in {"research_rl.train", "research_rl.train_recurrent",
                                "research_rl.train_rollout_improvement"}:
        raise ValueError("Unsupported trainer")
    expected = ("research_rl.recurrent:run_recurrent_search" if plan["trainer"].endswith("train_recurrent")
                else "research_rl:run_rl_search")
    if plan["entrypoint"] != expected:
        raise ValueError("Trainer/evaluator mismatch")
    if plan["replication"] not in (1, 2, 3) or not 1 <= plan["blocks"] <= 12:
        raise ValueError("Invalid replication/block count")
    if not 1 <= plan.get("block_seconds", 1800) <= 1800:
        raise ValueError("Use evaluation blocks of at most 30 minutes")
    values = plan["training_args"]
    if len(values) % 2 or len(set(values[::2])) != len(values[::2]):
        raise ValueError("Provide each algorithm parameter exactly once")
    for name, value in zip(values[::2], values[1::2]):
        if name not in ALGORITHM_ARGS or not re.fullmatch(r"[A-Za-z0-9_.+-]+", value):
            raise ValueError("Only public algorithm parameters are allowed")
    from sync_cpu_results import validate_remote
    validate_remote(plan["remote"])


def block_state(state, number, seconds, deadline, now=None):
    """Durable wall-clock allowance; even downtime cannot grant extra training."""
    now = time.time() if now is None else now
    blocks = state.setdefault("blocks", {})
    key = str(number)
    if key not in blocks:
        blocks[key] = dict(phase="training", started_unix=now,
                           deadline_unix=min(now+seconds, deadline-20))
    return blocks[key]


def compare_evaluations(reference, candidate, settings):
    from experiments.run_q3_comparison import bootstrap_mean_interval, percentile
    left={r["case_sha256"]:r for r in read(reference/"audited_rows.json")}
    right={r["case_sha256"]:r for r in read(candidate/"audited_rows.json")}
    if not left or left.keys()!=right.keys():
        raise ValueError("Comparison needs identical complete world hashes")
    pairs=[(left[key],right[key]) for key in sorted(left)]
    if any(x["physical_lower_bound_s"]!=y["physical_lower_bound_s"] for x,y in pairs):
        raise ValueError("Paired physical lower bounds differ")
    savings=[x["penalized_time_s"]-y["penalized_time_s"] for x,y in pairs]
    ci=bootstrap_mean_interval(savings,seed=settings["bootstrap_seed"],samples=settings["bootstrap_samples"])
    base=[x["penalized_time_s"] for x,y in pairs];new=[y["penalized_time_s"] for x,y in pairs]
    safe=all(x["successful"] and y["successful"] and x["failed_clear_count"]==y["failed_clear_count"]==0 for x,y in pairs)
    return dict(pairs=len(pairs),safe_full_clear_pairs=safe,mean_seconds_saved=statistics.mean(savings),
        paired_mean_savings_ci95_s=ci,mean_reduction_fraction=1-statistics.mean(new)/statistics.mean(base),
        p95_time_ratio=percentile(new,.95)/percentile(base,.95),max_time_ratio=max(new)/max(base),
        wins=sum(s>1e-6 for s in savings),losses=sum(s< -1e-6 for s in savings),
        maximum_paired_regression_s=max(-s for s in savings),
        reference_time_over_physical_lower=read(reference/"summary.json")["ratio_of_sums"],
        candidate_time_over_physical_lower=read(candidate/"summary.json")["ratio_of_sums"],
        strict_upgrade_on_supplied_cases=(safe and ci[0]>0 and percentile(new,.95)<=percentile(base,.95) and max(new)<=max(base)),
        scope="Development comparison; selection evidence, not independent confirmation")


def stop_group(process, grace=15):
    """Reap our session even when its leader exited before its pool workers."""
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    expires=time.monotonic()+grace
    while time.monotonic()<expires:
        process.poll()
        try:os.killpg(process.pid,0)
        except ProcessLookupError:
            process.wait()
            return
        time.sleep(.05)
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    process.wait()


def run(args):
    package=verify()
    if package.get("smoke_only"):
        raise ValueError("Do not launch server training from a smoke package")
    protocol=read(ROOT/"research/autonomy/protocol.json")
    plan=read(inside(ROOT,args.plan))
    validate_plan(plan)
    budget=constrain(ROOT,plan.get("cpu_slots",50))
    if not 1 <= plan["workers"] <= budget["cpu_slots"]-1 or not 1 <= plan["learner_threads"] <= budget["cpu_slots"]:
        raise ValueError("Worker/learner settings exceed the inherited CPU budget")
    deadline=datetime.fromisoformat(protocol["training_deadline_utc"]).timestamp()
    if time.time() >= deadline:
        raise RuntimeError("Training deadline already reached")
    import fcntl
    lock_stream=(ROOT/"runtime/job.lock").open("a+")
    try:fcntl.flock(lock_stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock_stream.close()
        raise RuntimeError("Another job owns this private package") from None
    lock=ROOT/"runtime/active_job.json"
    if lock.exists():
        prior=read(lock)
        if _alive(prior["pid"],prior["start_ticks"]):
            raise RuntimeError("A verified live job already owns this private package")
        prior_summary=inside(ROOT,"cpu_runs/"+prior["job"]+"/summary.json")
        prior_child=read(prior_summary).get("active_child") if prior_summary.exists() else None
        if prior_child and (_alive(prior_child["pid"],prior_child["start_ticks"]) or _group_alive(prior_child["pid"])):
            raise RuntimeError("Prior job child still lives; do not start another plan")
    job=inside(ROOT,"cpu_runs/"+plan["name"])
    job.mkdir(parents=True,exist_ok=True)
    identity=dict(plan=plan,package_sha256=sha(ROOT/"PACKAGE_MANIFEST.json"),protocol_sha256=sha(ROOT/"research/autonomy/protocol.json"))
    manifest_path=job/"run_manifest.json"
    if manifest_path.exists() and read(manifest_path) != identity:
        raise ValueError("Existing job identity differs; choose a fresh job name")
    write(manifest_path,identity)
    write(job/"environment.json",budget)
    state_path=job/"summary.json"
    state=read(state_path) if state_path.exists() else dict(completed_blocks=[],evaluations={},training_complete=False)
    prior_child=state.get("active_child")
    if prior_child and (_alive(prior_child["pid"],prior_child["start_ticks"]) or _group_alive(prior_child["pid"])):
        raise RuntimeError("Prior supervised child is still alive; do not duplicate training")
    pid=os.getpid()
    start_ticks=Path(f"/proc/{pid}/stat").read_text().rsplit(")",1)[1].split()[19]
    write(lock,dict(pid=pid,start_ticks=start_ticks,job=plan["name"]))
    state.update(pid=pid,start_ticks=start_ticks,status="running")
    write(state_path,state)
    stop=threading.Event()
    active=[None]
    def halt(*_):
        stop.set()
        process=active[0]
        if process is not None and process.poll() is None:
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    signal.signal(signal.SIGTERM,halt)
    signal.signal(signal.SIGINT,halt)
    # A sync thread belongs to this same affinity-limited process tree.
    from sync_cpu_results import ResultSync
    sync=ResultSync(ROOT/"cpu_runs",plan["remote"])
    def synchronize():
        last_models=-float("inf")
        while not stop.is_set():
            try:
                models=time.monotonic()-last_models>=600
                sync.cycle(include_models=models)
                if models:last_models=time.monotonic()
            except Exception as exc:
                print(json.dumps({"sync_error_type":type(exc).__name__,"will_retry":True}),flush=True)
            stop.wait(60)
    sync_thread=threading.Thread(target=synchronize,daemon=True)
    sync_thread.start()
    def command(argv,log_name,timeout):
        if stop.is_set():
            raise InterruptedError("Stop requested")
        env={**os.environ,"PYTHONPATH":str(ROOT/"src")+os.pathsep+str(ROOT)}
        log=job/"logs"/log_name;log.parent.mkdir(exist_ok=True)
        process=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,bufsize=1,start_new_session=True)
        active[0]=process
        state["active_child"] = dict(pid=process.pid,
            start_ticks=Path(f"/proc/{process.pid}/stat").read_text().rsplit(")",1)[1].split()[19])
        write(state_path,state)
        def expire():
            stop_group(process)
        timer=threading.Timer(max(1,timeout),expire);timer.start()
        try:
            with log.open("a",encoding="utf-8") as stream:
                for line in process.stdout:
                    stream.write(public_record(line));stream.flush()
            code=process.wait()
            if code:
                raise RuntimeError(f"Child failed with exit {code}; see {log_name}")
        finally:
            timer.cancel()
            stop_group(process,grace=1)
            process.wait()
            active[0]=None
            state.pop("active_child",None)
            write(state_path,state)
    trial_rel=f"cpu_runs/{plan['name']}/trial-1"
    trial=inside(ROOT,trial_rel)
    first,last=protocol["training_ranges_by_replication"][plan["replication"]-1]
    training_seed=protocol["training_seeds"][plan["replication"]-1]
    def train(block,initialize=False,block_deadline=None):
        effective_deadline=min(deadline-20,block_deadline if block_deadline else time.time()+plan.get("block_seconds",1800))
        limit=max(0,effective_deadline-time.time())
        if limit<=0:raise TimeoutError("Training deadline reached")
        wall_argument=limit
        if plan["trainer"]=="research_rl.train_recurrent" and not initialize:
            # The recurrent trainer explicitly treats --max-wall-s as lifetime
            # time; base PPO uses per-invocation time. Normalize without granting
            # another full block on resume. Its absolute deadline is authoritative.
            import torch
            from research_rl.cpu_runtime import require_cpu
            require_cpu("cpu")
            payload=torch.load(trial/"latest.pt",map_location="cpu",weights_only=False)
            wall_argument+=payload["state"]["elapsed_training_s"]
        argv=[sys.executable,"-m",plan["trainer"],"--output",trial_rel,"--device","cpu",
            "--seed",str(training_seed),"--scenario-start",str(first),"--scenario-end",str(last),
            "--max-attempted-episodes",str(last-first+1),"--updates","0" if initialize else "100000",
            "--workers",str(plan["workers"]),"--num-threads",str(plan["learner_threads"]),
            "--max-wall-s",str(wall_argument),"--checkpoint-seconds","600",
            "--deadline-utc",datetime.fromtimestamp(effective_deadline,timezone.utc).isoformat(),*plan["training_args"]]
        argv += ["--initialize-from","models/parent.pt"] if initialize else ["--resume",trial_rel+"/latest.pt"]
        command(argv,f"train-block-{block:02d}.log",limit+60)
    def evaluation(label,checkpoint=None,reference=False):
        output=f"cpu_runs/{plan['name']}/evaluation/{label}"
        argv=[sys.executable,"scripts/autonomy_job.py","evaluate","--output",output,
              "--label",label,"--split","development","--entrypoint",plan["entrypoint"]]
        if reference:argv += ["--reference-state"]
        else:argv += ["--checkpoint",checkpoint]
        command(argv,f"evaluation-{label}.log",1200)
        state["evaluations"][label]=read(inside(ROOT,output)/"summary.json")
        comparisons={}
        for baseline in ("state_search","initial"):
            reference=job/"evaluation"/baseline
            if baseline!=label and (reference/"audited_rows.json").is_file():
                comparisons[baseline]=compare_evaluations(reference,inside(ROOT,output),protocol["acceptance"])
        if comparisons:
            write(inside(ROOT,output)/"comparisons.json",comparisons)
            state["evaluations"][label]["comparisons"]=comparisons
        write(state_path,state)
    try:
        if not (trial/"latest.pt").exists():train(0,initialize=True)
        if "state_search" not in state["evaluations"]:evaluation("state_search",reference=True)
        if "initial" not in state["evaluations"]:
            initial_rel=f"cpu_runs/{plan['name']}/evaluation/initial/model.pt"
            initial=inside(ROOT,initial_rel);initial.parent.mkdir(parents=True,exist_ok=True)
            if not initial.exists():shutil.copyfile(trial/"latest.pt",initial)
            evaluation("initial",initial_rel)
        for block in range(1,plan["blocks"]+1):
            if block in state["completed_blocks"]:continue
            snapshot_rel=f"cpu_runs/{plan['name']}/evaluation/block-{block:02d}/model.pt"
            snapshot=inside(ROOT,snapshot_rel);snapshot.parent.mkdir(parents=True,exist_ok=True)
            current=block_state(state,block,plan.get("block_seconds",1800),deadline)
            write(state_path,state)
            if not snapshot.exists():
                if current["phase"]=="training" and time.time()<current["deadline_unix"]:
                    train(block,block_deadline=current["deadline_unix"])
                current["phase"]="model_ready";write(state_path,state)
                temporary=snapshot.with_suffix(".pt.tmp")
                shutil.copyfile(trial/"latest.pt",temporary);temporary.replace(snapshot)
            current.update(phase="evaluating",checkpoint_sha256=sha(snapshot));write(state_path,state)
            evaluation(f"block-{block:02d}",snapshot_rel)
            current["phase"]="complete"
            state["completed_blocks"].append(block);write(state_path,state)
        state.update(status="complete",training_complete=True)
    except BaseException as exc:
        state.update(status="interrupted" if stop.is_set() else "error",error_type=type(exc).__name__,training_complete=False)
        raise
    finally:
        write(state_path,state);stop.set();sync_thread.join(timeout=120)
        if not sync_thread.is_alive():
            try:sync.cycle(include_models=True,final=True)
            except Exception as exc:print(json.dumps({"final_sync_error_type":type(exc).__name__}),flush=True)
        if lock.exists() and read(lock).get("pid")==pid:lock.unlink()
        fcntl.flock(lock_stream.fileno(),fcntl.LOCK_UN);lock_stream.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    job=sub.add_parser("run");job.add_argument("--plan",required=True)
    evaluation=sub.add_parser("evaluate")
    evaluation.add_argument("--output",required=True);evaluation.add_argument("--checkpoint")
    evaluation.add_argument("--label",required=True);evaluation.add_argument("--reference-state",action="store_true")
    evaluation.add_argument("--entrypoint",default="research_rl:run_rl_search")
    evaluation.add_argument("--split",choices=("development","confirmation","final"),default="development")
    evaluation.add_argument("--freeze-record")
    args=p.parse_args()
    {"run":run,"evaluate":evaluate}[args.command](args)


if __name__=="__main__":
    main()
