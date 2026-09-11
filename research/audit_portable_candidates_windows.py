"""Load three actual provisional weights, local training-world smoke, no HTTP.

This is compatibility evidence only. It neither selects a candidate nor opens
extended/final or official simulator cases. Model files are never rewritten.
"""

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import platform
import statistics
import tempfile
import time
from unittest.mock import patch

import torch

from experiments import run_q3_practice
from research_rl import run_rl_search
from research_rl.network import load_policy
from research_rl.portable_checkpoint import model_tensor_digest
from research_rl.train import source_manifest
from simulation import LocalResearchSimulator, random_scenario
from tests.test_strategy import ObservationOnlyClient


NAMES = ("rl-best002", "rl-axis-initial", "rl-base-u393")
SEEDS = tuple(range(100121,100125))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts",type=Path,default=Path("../q3-v1-artifacts/rl-portable-candidates-0810"))
    parser.add_argument("--output",type=Path,default=Path("research/windows_portable_candidates.json"))
    args=parser.parse_args(argv)
    if platform.system()!="Windows":
        parser.error("This audit explicitly records the Windows CPU delivery path")
    torch.set_num_threads(1)
    directory=args.artifacts.resolve()/"rl-de7d65b-git"
    manifest_path=args.artifacts/"rl-portable-candidates-0810.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    result=dict(schema=1,scope="Provisional Windows CPU compatibility, not official practice/final evaluation or algorithm ranking",
        selection_status="provisional; no candidate selected by this audit",seeds=list(SEEDS),
        platform=platform.platform(),python=platform.python_version(),torch=torch.__version__,num_threads=1,
        input_manifest_sha256=sha(manifest_path.read_bytes()),source_manifest=source_manifest(),
        script_sha256=sha(Path(__file__).read_bytes()),candidates={})
    for name in NAMES:
        checkpoint=directory/"results/rl/v1-candidates"/(name+".pt")
        file_hash=sha(checkpoint.read_bytes())
        assert file_hash==manifest[name]["checkpoint_sha256"]
        original_spec_path=directory/"research"/("v1-candidate-"+name+".json")
        spec=json.loads(original_spec_path.read_text(encoding="utf-8"))
        assert spec["name"]==name and spec["entrypoint"]=="research_rl:run_rl_search"
        assert spec["kwargs"]["device"]=="cpu" and spec["kwargs"]["deterministic"] is True
        spec["kwargs"]["checkpoint"]=str(checkpoint)
        started=time.perf_counter()
        policy=load_policy(checkpoint,device="cpu",deterministic=True)
        load_s=time.perf_counter()-started
        assert policy.feature_version=="v3" and policy.model.hidden==96
        assert policy.architecture["name"]=="mlp" and policy.action_distribution["name"]=="flat"
        assert policy.action_schema["name"]==("axis_quantiles" if name=="rl-axis-initial" else "base")
        rows=[]
        for seed in SEEDS:
            simulator=LocalResearchSimulator(random_scenario(3,seed))
            client=ObservationOnlyClient(simulator.client())
            started=time.perf_counter()
            report=run_rl_search(client,problem=3,max_actions=10000,**spec["kwargs"])
            wall=time.perf_counter()-started
            evaluation=simulator.evaluation()
            assert evaluation["all_cleared"] and evaluation["failed_clear_count"]==0
            assert report.completion_certified_under_model and not report.error and not report.exit_error
            assert client.state.session=="exited" and client.pending_request is None
            assert abs(report.virtual_time_s-evaluation["virtual_time_s"])<1e-6
            rows.append(dict(seed=seed,all_cleared=True,failed_clear_count=0,certified=True,
                accepted_exit=True,cpu_wall_s=wall,virtual_time_s=report.virtual_time_s,
                measurements=report.measurement_count,physical_actions=report.accepted_actions,
                source_count=evaluation["source_total"]))
        attempts=[]
        def forbidden(*a,**kw):
            attempts.append("HTTP/client creation attempted")
            raise AssertionError("Compatibility dry-run must not create an HTTP client or connection")
        with tempfile.TemporaryDirectory(prefix="q3-candidate-dryrun-") as temporary:
            path=Path(temporary)/"spec.json"
            path.write_text(json.dumps(spec,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
            output=io.StringIO()
            with patch.object(run_q3_practice.SimulatorClient,"__init__",forbidden), \
                 patch("socket.create_connection",forbidden), \
                 patch("socket.socket.connect",forbidden), \
                 patch("urllib.request.OpenerDirector.open",forbidden),redirect_stdout(output):
                assert run_q3_practice.main(["--spec",str(path),"--dry-run"])==0
            dry=json.loads(output.getvalue())
        assert dry["dry_run"] and dry["simulator_requests_sent"] is False and not attempts
        assert dry["identity"]["checkpoint_sha256"]["checkpoint"]==file_hash
        assert Path(dry["spec"]["kwargs"]["checkpoint"]).samefile(checkpoint)
        assert sha(checkpoint.read_bytes())==file_hash
        result["candidates"][name]=dict(checkpoint=str(checkpoint),checkpoint_sha256=file_hash,
            model_tensor_sha256=model_tensor_digest(policy.model.state_dict()),
            original_spec_sha256=sha(original_spec_path.read_bytes()),local_spec=spec,
            feature_version=policy.feature_version,feature_dim=policy.model.feature_dim,hidden=policy.model.hidden,
            architecture=policy.architecture,action_distribution=policy.action_distribution,action_schema=policy.action_schema,
            first_model_load_s=load_s,mean_cpu_episode_wall_s=statistics.mean(r["cpu_wall_s"] for r in rows),
            dry_run=dict(cli_success=True,simulator_requests_sent=False,network_attempts=attempts,
                actual_file_sha256=file_hash,identity_sha256=sha(json.dumps(dry["identity"],sort_keys=True).encode())),
            episodes=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({name:dict(load_s=r["first_model_load_s"],mean_cpu_wall_s=r["mean_cpu_episode_wall_s"],
        successful=len(r["episodes"]),dry_run_no_http=not r["dry_run"]["simulator_requests_sent"])
        for name,r in result["candidates"].items()},indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
