"""Actual GAE u512 Windows CPU smoke and guarded practice dry-run, no HTTP."""

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


def main():
    assert platform.system()=="Windows"
    torch.set_num_threads(1)
    root=Path("../q3-v1-artifacts/milestone-0818/rl-de7d65b-git").resolve()
    checkpoint=root/"results/rl/cold-gae095-001/ppo_000512.pt"
    manifest_path=root/"results/research_v1/validation-cold-gae095-001-ppo_000512/manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    checksum=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert checksum==manifest["identity"]["checkpoint_sha256"]["checkpoint"]
    started=time.perf_counter()
    policy=load_policy(checkpoint,device="cpu",deterministic=True)
    load_s=time.perf_counter()-started
    assert policy.feature_version=="v3" and policy.action_schema["name"]=="base"
    assert policy.architecture["name"]=="mlp" and policy.action_distribution["name"]=="flat"
    spec=dict(name="rl-gae095-u512-compatibility",entrypoint="research_rl:run_rl_search",
              kwargs=dict(checkpoint=str(checkpoint),device="cpu",deterministic=True))
    rows=[]
    for seed in range(100121,100125):
        simulator=LocalResearchSimulator(random_scenario(3,seed))
        client=ObservationOnlyClient(simulator.client())
        started=time.perf_counter()
        report=run_rl_search(client,problem=3,max_actions=10000,**spec["kwargs"])
        wall=time.perf_counter()-started
        evaluation=simulator.evaluation()
        assert evaluation["all_cleared"] and evaluation["failed_clear_count"]==0
        assert report.completion_certified_under_model and not report.error and not report.exit_error
        assert client.state.session=="exited" and client.pending_request is None
        rows.append(dict(seed=seed,all_cleared=True,failed_clear_count=0,certified=True,accepted_exit=True,
                         cpu_wall_s=wall,virtual_time_s=report.virtual_time_s,actions=report.accepted_actions))
    attempts=[]
    def forbidden(*a,**kw):
        attempts.append("client or network attempted")
        raise AssertionError("No network allowed during this compatibility dry-run")
    with tempfile.TemporaryDirectory(prefix="q3-gae-dryrun-") as temporary:
        path=Path(temporary)/"spec.json"
        path.write_text(json.dumps(spec,ensure_ascii=False),encoding="utf-8")
        output=io.StringIO()
        with patch.object(run_q3_practice.SimulatorClient,"__init__",forbidden), \
             patch("socket.create_connection",forbidden),patch("socket.socket.connect",forbidden), \
             patch("urllib.request.OpenerDirector.open",forbidden),redirect_stdout(output):
            assert run_q3_practice.main(["--spec",str(path),"--dry-run"])==0
        dry=json.loads(output.getvalue())
    assert not attempts and dry["simulator_requests_sent"] is False
    assert dry["identity"]["checkpoint_sha256"]["checkpoint"]==checksum
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==checksum
    result=dict(schema=1,scope="Windows CPU compatibility only; training seeds; no official/selection/final evaluation",
        checkpoint=str(checkpoint),checkpoint_sha256=checksum,
        evaluation_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        model_tensor_sha256=model_tensor_digest(policy.model.state_dict()),source_manifest=source_manifest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        platform=platform.platform(),torch=torch.__version__,num_threads=1,
        feature_version=policy.feature_version,feature_dim=policy.model.feature_dim,hidden=policy.model.hidden,
        architecture=policy.architecture,action_schema=policy.action_schema,distribution=policy.action_distribution,
        first_model_load_s=load_s,mean_cpu_episode_wall_s=statistics.mean(r["cpu_wall_s"] for r in rows),episodes=rows,
        dry_run=dict(success=True,simulator_requests_sent=False,network_attempts=attempts,
                     checkpoint_sha256=checksum),weight_file_unchanged=True)
    Path("research/windows_gae095.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in {"source_manifest","episodes"}},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
