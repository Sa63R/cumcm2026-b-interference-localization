"""Actor cost-to-go learning, full-action anchoring and restart accounting."""
from argparse import Namespace
import copy
import gzip
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from research_rl import train_rollout_improvement as trainer
from research_rl.controller import ALGORITHM_VERSIONS, feature_schema
from research_rl.network import CandidateActorCritic, pack_observations, load_policy
from research_rl.portable_checkpoint import model_tensor_digest


@pytest.fixture(autouse=True)
def one_torch_thread():
    torch.set_num_threads(1)


def args(**changes):
    result = dict(gap_scale_s=100., max_pair_weight=2., kl_coef=1., epochs=4,
        minibatch=2, max_grad_norm=.5, pair_scope="all", alternative_sampling="uniform")
    return Namespace(**(result | changes))


def public_group(model, seed=3100011, costs=(500.,100.,300.)):
    rng = np.random.default_rng(seed)
    features = rng.normal(0,.3,(3,60)).astype(np.float32)
    context = rng.normal(0,.3,12).astype(np.float32)
    with torch.no_grad():
        logits = model(*pack_observations([dict(features=features,context=context)]))[0][0].numpy().copy()
    return dict(seed=seed,features=features,context=context,old_logits=logits,
        original_index=int(logits.argmax()),prefix_cost_s=100.,prefix_step=5,prefix_sha256="test-public-prefix",
        evaluated_candidates=[dict(index=i,total_time_s=100+cost,remaining_cost_s=cost,
            cost_to_go_s=cost,failure_penalty_s=0.,success=True,failed_clear_count=0) for i,cost in enumerate(costs)])


def make_parent(path, hidden=16):
    torch.manual_seed(782)
    model = CandidateActorCritic(hidden,60)
    torch.save(dict(algorithm=ALGORITHM_VERSIONS["v3"],hidden=hidden,model=model.state_dict(),
        feature_schema=feature_schema("v3"),architecture=model.architecture,
        action_distribution=model.action_distribution,action_schema=model.action_schema,
        state=dict(update=900,optimizer_steps=100000,episodes=99000)),path)
    return model


def cli(output, parent=None, updates=1, **flags):
    result = ["--output",str(output),"--hidden","16","--updates",str(updates),
        "--workers","0","--num-threads","1","--groups-per-update","2",
        "--epochs","1","--minibatch","1","--scenario-start","3100011",
        "--scenario-end","3100020","--max-attempted-episodes","4","--max-wall-s","60"]
    result += ["--initialize-from",str(parent)] if parent else ["--resume",str(output/"latest.pt")]
    for key,value in flags.items():
        result += ["--"+key.replace("_","-"),str(value)]
    return result


def fake_collect(task):
    model = CandidateActorCritic(task["hidden"],60)
    model.load_state_dict(task["weights"])
    group = public_group(model,task["seed"])
    from research_rl.prefix_rollouts import select_alternatives
    import random
    sampling = task.get("alternative_sampling", "uniform")
    alternatives = select_alternatives(group["old_logits"], group["original_index"], 2,
                                      random.Random(task["action_seed"]), sampling)
    by_index = {c["index"]: c for c in group["evaluated_candidates"]}
    group.update(alternative_sampling=sampling, alternative_indices=alternatives,
        evaluated_candidates=[by_index[i] for i in [group["original_index"], *alternatives]])
    return group,dict(status="complete",trajectories_started=3,
        trajectories_completed=3,trajectories_interrupted=0,simulator_action_count=90,
        replay_prefix_action_count=20,successful_trajectories=3,failed_trajectories=0,
        failed_clear_count=0,baseline_total_time_s=600.)


class FixedLogits(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = torch.nn.Parameter(torch.tensor(values,dtype=torch.float32))

    def forward(self, features, context, mask):
        return self.values[None,:].expand(len(features),-1),torch.zeros(len(features))


def test_labels_use_full_continuation_and_kl_includes_unevaluated_candidates():
    model = FixedLogits([0.,0.,0.,0.])
    # Candidate 1 can have a large immediate movement, yet its FULL remaining
    # task cost is smallest. The target must prefer it, not the nearest action.
    features = np.zeros((4,60),np.float32); features[1,8] = .9
    group = dict(features=features,context=np.zeros(12,np.float32),old_logits=np.zeros(4,np.float32),
        original_index=0,prefix_cost_s=100.,evaluated_candidates=[
            dict(index=i,total_time_s=100+c,remaining_cost_s=c,cost_to_go_s=c,
                failure_penalty_s=0,success=True,failed_clear_count=0) for i,c in enumerate((400.,100.,300.))])
    trainer.validate_group(group)
    loss, pair_before, kl = trainer.objective(model,[group],args())
    assert float(kl.detach()) == pytest.approx(0.)
    loss.backward()
    assert model.values.grad[1] < 0 < model.values.grad[0]
    with torch.no_grad():
        model.values[3] = 2.  # This action was NOT given a continuation label.
    _, pair_after, kl = trainer.objective(model,[group],args())
    assert float(pair_after.detach()) == pytest.approx(float(pair_before.detach()))
    expected = torch.distributions.kl_divergence(torch.distributions.Categorical(logits=torch.zeros(4)),
        torch.distributions.Categorical(logits=model.values))
    assert float(kl.detach()) == pytest.approx(float(expected.detach())) and float(kl.detach()) > .1


def test_original_scope_removes_actual_first_group_irrelevant_loser_ranking_gradient():
    # Public logits and audited full remaining costs from training group 3100001.
    # Relative common logit offsets have no effect on either loss. No new world,
    # future data selection, checkpoint update, or model fitting is performed.
    logits = [-16.71570587158203, -42.02256774902344, -40.65563201904297]
    costs = [889.865508, 1270.682663, 1598.310802]
    group = dict(features=np.zeros((3,60),np.float32), context=np.zeros(12,np.float32),
        old_logits=np.array(logits,np.float32), original_index=0, prefix_cost_s=2132.868024,
        evaluated_candidates=[dict(index=i,total_time_s=2132.868024+c,remaining_cost_s=c,
            cost_to_go_s=c,failure_penalty_s=0.) for i,c in enumerate(costs)])
    results = {}
    for scope in ("all", "original"):
        model = FixedLogits(logits)
        _, pair, _ = trainer.objective(model,[group],args(pair_scope=scope))
        pair.backward()  # Isolate pair signal; this is not a shared-parameter norm claim.
        results[scope] = (float(pair.detach()), model.values.grad.abs().max().item())
    assert results["all"][0] == pytest.approx(1.0626540694,rel=1e-6)
    assert results["all"][1] > .53
    assert results["original"][0] < 1e-9 and results["original"][1] < 1e-9
    pairs = trainer.ranking_pairs(group,args(pair_scope="original"))
    assert len(pairs) == 2 and all(0 in (b,w) for b,w,_ in pairs)


def test_original_scope_still_promotes_lower_full_cost_and_normalizes_actual_pairs():
    group = dict(original_index=0,features=np.zeros((3,60),np.float32),context=np.zeros(12,np.float32),
        old_logits=np.zeros(3,np.float32),evaluated_candidates=[
            dict(index=0,cost_to_go_s=400.),dict(index=1,cost_to_go_s=100.),dict(index=2,cost_to_go_s=300.)])
    model = FixedLogits([0.,0.,0.])
    _, loss, _ = trainer.objective(model,[group],args(pair_scope="original"))
    assert float(loss.detach()) == pytest.approx(1.5*np.log(2))  # weights 2 and 1, mean TWO pairs
    loss.backward()
    assert model.values.grad[0] > 0 and model.values.grad[1] < 0 and model.values.grad[2] < 0
    group["evaluated_candidates"] = group["evaluated_candidates"][:2]
    _, one_pair, _ = trainer.objective(model,[group],args(pair_scope="original"))
    assert float(one_pair.detach()) == pytest.approx(2*np.log(2))


@pytest.mark.parametrize("scope",["all","original"])
def test_scope_does_not_change_full_candidate_kl_including_unevaluated_action(scope):
    model = FixedLogits([0.,0.,0.,2.])
    group = dict(features=np.zeros((4,60),np.float32),context=np.zeros(12,np.float32),
        old_logits=np.zeros(4,np.float32),original_index=0,
        evaluated_candidates=[dict(index=i,cost_to_go_s=c) for i,c in enumerate((400.,100.,300.))])
    _, _, kl = trainer.objective(model,[group],args(pair_scope=scope))
    expected = torch.distributions.kl_divergence(torch.distributions.Categorical(logits=torch.zeros(4)),
        torch.distributions.Categorical(logits=model.values))
    assert float(kl.detach()) == pytest.approx(float(expected.detach()))


def test_actual_optimizer_changes_actor_and_shared_trunk_but_never_critic():
    torch.manual_seed(52); model = CandidateActorCritic(16,60)
    groups = [public_group(model,3100011+i) for i in range(4)]
    before = {k:v.clone() for k,v in model.state_dict().items()}
    optimizer = torch.optim.Adam(trainer.freeze_critic(model),lr=.003)
    result = trainer.update_actor(model,optimizer,groups,args())
    assert result["optimizer_steps"] == 8
    assert result["groups_used_by_optimizer"] == 4
    assert result["after"]["pairwise_loss"] < result["before"]["pairwise_loss"]
    assert any(not torch.equal(v,before[k]) for k,v in model.state_dict().items() if k.startswith("actor."))
    assert any(not torch.equal(v,before[k]) for k,v in model.state_dict().items() if k.startswith("encoder."))
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items() if k.startswith("critic."))
    measured = trainer.policy_statistics(model,groups,args())
    assert result["after"]["full_candidate_kl"] == pytest.approx(measured["full_candidate_kl"])


def test_equal_costs_do_not_drift_adam_or_invent_a_preference():
    model = CandidateActorCritic(16,60)
    optimizer = torch.optim.Adam(trainer.freeze_critic(model),lr=.001)
    group = public_group(model,costs=(100.,100.,100.))
    before = model_tensor_digest(model.state_dict())
    result = trainer.update_actor(model,optimizer,[group],args())
    assert result["optimizer_steps"] == 0 and result["skipped_equal_costs"]
    assert before == model_tensor_digest(model.state_dict()) and not optimizer.state


def test_partial_epoch_reports_only_groups_actually_used(monkeypatch):
    model = CandidateActorCritic(16,60)
    groups = [public_group(model,3100011+i) for i in range(3)]
    optimizer = torch.optim.Adam(trainer.freeze_critic(model),lr=.001)
    clock = [0.]; used = []
    monkeypatch.setattr(trainer.time,"monotonic",lambda:clock[0])
    def step(new_groups):
        used.append(new_groups); clock[0] = 1.
    result = trainer.update_actor(model,optimizer,groups,args(minibatch=1),stop_at=.5,on_step=step)
    assert result["optimizer_steps"] == 1 and result["groups_used_by_optimizer"] == 1
    assert used == [1] and result["deadline_reached"]


def test_init_only_preserves_original_inference_schema_and_resets_counters(tmp_path,monkeypatch):
    parent = tmp_path/"parent.pt"; original = make_parent(parent); output = tmp_path/"init"
    monkeypatch.setattr(trainer,"collect_group",lambda _:pytest.fail("init-only must not sample"))
    assert trainer.main(cli(output,parent,updates=0)) == 0
    payload = torch.load(output/"latest.pt",weights_only=False)
    assert payload["training_algorithm"] == trainer.TRAINING_ALGORITHM
    assert payload["state"] == payload["training_state"]
    assert payload["state"]["update"] == payload["state"]["optimizer_steps"] == payload["state"]["episodes"] == 0
    assert not payload["optimizer"]["state"]
    policy = load_policy(output/"latest.pt")
    group = public_group(original)
    assert all(torch.equal(payload["model"][k],v) for k,v in original.state_dict().items())
    assert policy(group["features"],group["context"],0)[0] == group["original_index"]


@pytest.mark.parametrize("scope,sampling",[("all","uniform"),("original","runner_up_uniform")])
def test_resume_matches_uninterrupted_training_and_counts_worlds_separately(tmp_path,monkeypatch,scope,sampling):
    parent = tmp_path/"parent.pt"; make_parent(parent)
    monkeypatch.setattr(trainer,"collect_group",fake_collect)
    uninterrupted, split = tmp_path/"all",tmp_path/"split"
    flags = dict(pair_scope=scope,alternative_sampling=sampling)
    trainer.main(cli(uninterrupted,parent,updates=2,**flags))
    trainer.main(cli(split,parent,updates=1,**flags))
    trainer.main(cli(split,updates=2,**flags))
    a,b = [torch.load(p/"latest.pt",weights_only=False) for p in (uninterrupted,split)]
    assert all(torch.equal(a["model"][k],v) for k,v in b["model"].items())
    state = b["state"]
    assert state["update"] == 2 and state["optimizer_steps"] == 4
    assert state["episodes"] == state["completed_groups"] == state["attempted_groups"] == state["learned_groups"] == 4
    assert state["actual_rollout_trajectories"] == state["completed_rollout_trajectories"] == 12
    assert state["simulator_actions"] == 360
    assert state["next_seed"] == 3100015 and "world groups" in state["episode_unit"]
    assert a["optimizer"]["param_groups"] == b["optimizer"]["param_groups"]
    for key, buffers in a["optimizer"]["state"].items():
        assert all(torch.equal(value,b["optimizer"]["state"][key][name]) for name,value in buffers.items())
    assert b["objective"]["pair_scope"] == scope and b["objective"]["alternative_sampling"] == sampling
    logs = [json.loads(line) for line in (split/"training.jsonl").read_text().splitlines()]
    assert all(row["pair_scope"] == scope and row["alternative_sampling"] == sampling for row in logs)
    assert all(row["losses"]["after"]["pair_scope"] == scope for row in logs)


def test_reservation_survives_interrupt_and_never_reuses_seed(tmp_path,monkeypatch):
    parent = tmp_path/"parent.pt"; make_parent(parent); output = tmp_path/"interrupted"
    def interrupt(task):
        saved = torch.load(output/"latest.pt",weights_only=False)
        assert saved["state"]["attempted_groups"] == 2
        assert saved["state"]["next_seed"] == 3100013
        raise KeyboardInterrupt("simulated post-reservation interruption")
    monkeypatch.setattr(trainer,"collect_group",interrupt)
    with pytest.raises(KeyboardInterrupt):
        trainer.main(cli(output,parent))
    received = []
    def collect(task):
        received.append(task["seed"]); return fake_collect(task)
    monkeypatch.setattr(trainer,"collect_group",collect)
    trainer.main(cli(output,updates=1))
    state = torch.load(output/"latest.pt",weights_only=False)["state"]
    assert received == [3100013,3100014]
    assert state["attempted_groups"] == 4 and state["episodes"] == 2
    assert state["abandoned_reserved_groups"] == 2 and state["unreported_work_batches"] == 1
    assert state["counters_are_lower_bounds_after_unreported_interruption"]


@pytest.mark.parametrize("status",["administrative_timeout","invalid_execution"])
def test_incomplete_group_prevents_learning_the_fast_part_of_batch(tmp_path,monkeypatch,status):
    parent = tmp_path/"parent.pt"; original = make_parent(parent); output = tmp_path/status
    def collect(task):
        if task["seed"] == 3100012:
            return None,dict(status=status,trajectories_started=1,trajectories_completed=0,
                trajectories_interrupted=1,simulator_action_count=7)
        return fake_collect(task)
    monkeypatch.setattr(trainer,"collect_group",collect)
    if status == "invalid_execution":
        with pytest.raises(RuntimeError,match="no group"):
            trainer.main(cli(output,parent))
    else:
        trainer.main(cli(output,parent))
    payload = torch.load(output/"latest.pt",weights_only=False)
    assert payload["state"]["optimizer_steps"] == payload["state"]["learned_groups"] == 0
    assert all(torch.equal(payload["model"][k],v) for k,v in original.state_dict().items())
    assert payload["state"]["episodes"] == 1 and payload["state"]["actual_rollout_trajectories"] == 4


def test_resume_cannot_change_objective_or_enlarge_lifetime_budget(tmp_path,monkeypatch):
    parent = tmp_path/"parent.pt"; make_parent(parent); output = tmp_path/"frozen"
    monkeypatch.setattr(trainer,"collect_group",fake_collect)
    trainer.main(cli(output,parent,updates=0))
    with pytest.raises(ValueError,match="objective"):
        trainer.main(cli(output,kl_coef=2.))
    with pytest.raises(ValueError,match="objective"):
        trainer.main(cli(output,pair_scope="original"))
    with pytest.raises(ValueError,match="objective"):
        trainer.main(cli(output,alternative_sampling="runner_up_uniform"))
    with pytest.raises(ValueError,match="enlarge"):
        trainer.main(cli(output,max_attempted_episodes=5))


def test_new_modes_bind_worker_task_and_evidence_and_reject_mislabelled_collection(tmp_path,monkeypatch):
    parent = tmp_path/"parent.pt"; make_parent(parent)
    seen = []
    def collect(task):
        seen.append(task["alternative_sampling"])
        group,metrics = fake_collect(task)
        metrics.update(seed=task["seed"],evidence=[dict(public_only=True)])
        return group,metrics
    monkeypatch.setattr(trainer,"collect_group",collect)
    output = tmp_path/"bound"
    trainer.main(cli(output,parent,updates=1,pair_scope="original",alternative_sampling="runner_up_uniform"))
    assert seen == ["runner_up_uniform"]*2
    payload = torch.load(output/"latest.pt",weights_only=False)
    assert payload["args"]["pair_scope"] == "original"
    assert payload["objective"]["alternative_sampling"] == "runner_up_uniform"
    evidence = json.loads(gzip.decompress((output/"evidence/group-3100011.json.gz").read_bytes()))
    assert evidence["group"]["alternative_sampling"] == "runner_up_uniform"
    trainer.validate_group(evidence["group"],"runner_up_uniform")
    wrong = copy.deepcopy(evidence["group"])
    wrong["alternative_sampling"] = "uniform"
    with pytest.raises(ValueError,match="sampling"):
        trainer.validate_group(wrong,"runner_up_uniform")
    wrong = copy.deepcopy(evidence["group"])
    wrong["evaluated_candidates"][1:] = reversed(wrong["evaluated_candidates"][1:])
    wrong["alternative_indices"] = [c["index"] for c in wrong["evaluated_candidates"][1:]]
    with pytest.raises(ValueError,match="runner-up"):
        trainer.validate_group(wrong,"runner_up_uniform")


def test_two_legal_candidates_need_only_one_label_pair_for_both_modes():
    torch.manual_seed(64); model = CandidateActorCritic(16,60)
    group = public_group(model)
    group["features"] = group["features"][:2].copy()
    with torch.no_grad():
        group["old_logits"] = model(*pack_observations([group]))[0][0].numpy().copy()
    group["original_index"] = int(group["old_logits"].argmax())
    group["evaluated_candidates"] = group["evaluated_candidates"][:2]
    group["alternative_sampling"] = "runner_up_uniform"
    for scope in ("all","original"):
        trainer.validate_group(group,"runner_up_uniform")
        assert len(trainer.ranking_pairs(group,args(pair_scope=scope))) == 1
    optimizer = torch.optim.Adam(trainer.freeze_critic(model),lr=.001)
    result = trainer.update_actor(model,optimizer,[group],
        args(pair_scope="original",alternative_sampling="runner_up_uniform",epochs=1,minibatch=1))
    assert result["optimizer_steps"] == 1


def test_small_evidence_is_separate_and_never_overwrites(tmp_path):
    metrics = dict(seed=3100011,status="complete",evidence=[dict(public=[1,2,3])])
    model = CandidateActorCritic(16,60); group = public_group(model)
    state = dict(evidence_files={})
    trainer.save_small_evidence(tmp_path,[(group,metrics)],"fixed-actor",state)
    assert "evidence" not in metrics and metrics["evidence_file"] in state["evidence_files"]
    payload = json.loads(gzip.decompress((tmp_path/metrics["evidence_file"]).read_bytes()))
    assert payload["actor_tensor_sha256"] == "fixed-actor"
    assert payload["group"]["original_index"] == group["original_index"]
    with pytest.raises(ValueError,match="overwrite"):
        trainer.save_small_evidence(tmp_path,[(group,dict(seed=3100011,evidence=[{}]))],"fixed-actor",state)


def test_real_complete_continuation_group_drives_actor_update_and_has_physical_lb():
    from research_rl.prefix_rollouts import collect_group
    from simulation import random_scenario
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research/theory_v1"))
    from audit_eval_bounds import audit_record, physical_bounds
    torch.manual_seed(441); model = CandidateActorCritic(16,60)
    # Fixed new training case, never a validation or official simulator world.
    group,metrics = collect_group(dict(seed=3100701,weights=model.state_dict(),hidden=16,
        action_seed=337,capture_evidence=True))
    assert metrics["status"] == "complete" and group is not None
    assert metrics["trajectories_completed"] == 3
    evidence = metrics["evidence"]
    # Ground truth below is accessed only after every completed continuation.
    sources = None
    for run in evidence:
        sources,_ = audit_record(dict(row=run["row"],evaluation=run["evaluation"],
            summary=run["summary"],history=run["history"],evaluation_phase="after_policy_termination"))
    bound = physical_bounds(sources.values())["physical_clairvoyant_lower_s"]
    times = [row["total_time_s"] for row in group["evaluated_candidates"]]
    assert all(t >= bound for t in times)
    print(json.dumps(dict(smoke_training_seed=3100701,physical_lower_bound_s=bound,
        complete_trajectory_times_s=times,time_over_lower_bound=[t/bound for t in times])))
    optimizer = torch.optim.Adam(trainer.freeze_critic(model),lr=.001)
    result = trainer.update_actor(model,optimizer,[group],args(epochs=1,minibatch=1))
    assert result["optimizer_steps"] == 1 and np.isfinite(result["after"]["full_candidate_kl"])
