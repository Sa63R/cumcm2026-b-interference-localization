"""Action-schema compatibility, legal geometry, actual optimization and replay."""

from argparse import Namespace
import json
import math

import numpy as np
import pytest

from research_rl import run_rl_search
from research_rl.action_sets import action_schema, action_schema_from_args, checkpoint_action_schema, controller_for
from research_rl.axis_probes import AxisProbeRLSearch, axis_templates
from research_rl.controller import ALGORITHM_VERSIONS, CONTEXT_DIM, feature_schema
from research_rl.joint_scan import JointScanRLSearch
from localization.omni import OmniCandidateRegion
from simulation import LocalResearchSimulator, random_scenario
from tests.test_strategy import ObservationOnlyClient


def test_templates_are_in_certified_reception_and_are_new_positions():
    region = OmniCandidateRegion().observe((0, 0), 0)
    points = axis_templates(region)
    from research.audit_axis_probe_candidates import axis_proposals
    assert points == tuple(axis_proposals(region)[0])
    assert 1 <= len(points) <= 14
    center = region.enclosing_disk().center
    assert any(abs(p.x-center[0]) > 100 for p in points)
    for p in points:
        assert max(math.hypot(x-p.x, y-p.y) for x,y in region.vertices) <= 1000-1e-7
        # The guarantee extends to interior convex combinations, not only vertices.
        for i,a in enumerate(region.vertices):
            b = region.vertices[(i+1)%len(region.vertices)]
            x,y = .3*a[0]+.7*b[0], .3*a[1]+.7*b[1]
            assert math.hypot(x-p.x,y-p.y) <= 1000


@pytest.mark.parametrize("seed", range(110101,110109))
def test_extended_teacher_keeps_old_actions_and_old_feature_rows_exact(seed):
    class CheckedAxis(AxisProbeRLSearch):
        def _features(self, candidates, remaining):
            features, context = super()._features(candidates, remaining)
            indices = [i for i,c in enumerate(candidates) if c.option != 6]
            before, old_context = JointScanRLSearch._features(self, [candidates[i] for i in indices], remaining)
            assert context == old_context
            assert [features[i] for i in indices] == before
            assert len(candidates) <= 140+16*20
            assert all(len(row)==60 and all(math.isfinite(v) for v in row) for row in features)
            extra = [c for c in candidates if c.option==6]
            assert all(c.kind=="probe" for c in extra)
            assert len({(c.channel,round(c.point.x,6),round(c.point.y,6)) for c in extra}) == len(extra)
            return features, context
    scenario = random_scenario(3,seed)
    first, second = LocalResearchSimulator(scenario), LocalResearchSimulator(scenario)
    extended = CheckedAxis(ObservationOnlyClient(first.client()), lambda f,c,t:t).run()
    old = run_rl_search(ObservationOnlyClient(second.client()),policy=lambda f,c,t:t,feature_version="v3")
    assert extended.action_history == old.action_history
    assert extended.virtual_time_s == old.virtual_time_s
    assert extended.learning["axis_probe_measurements"] == 0
    assert extended.learning["axis_candidates_offered_sum"] > 0
    assert first.evaluation()["all_cleared"]


@pytest.mark.parametrize("seed", range(110111,110115))
def test_axis_actions_are_executed_labelled_and_bounded_with_observation_only_client(seed):
    def prefer_axis(features, context, teacher):
        return next((i for i,row in enumerate(features) if row[1]==1 and row[18]==1), teacher)
    simulator = LocalResearchSimulator(random_scenario(3,seed))
    report = run_rl_search(ObservationOnlyClient(simulator.client()), policy=prefer_axis,
                           feature_version="v3", probe_candidates="axis_quantiles")
    evaluation = simulator.evaluation()
    labelled = [a for a in report.action_history if a.get("rl_probe_family")=="axis_quantiles"]
    assert labelled and all(a["action"]=="measure" for a in labelled)
    assert report.learning["axis_probe_measurements"] == len(labelled)
    assert report.learning["decisions"] <= 252
    assert report.learning["fallback_counts"].get("decision_limit",0)==0
    assert evaluation["all_cleared"] and evaluation["failed_clear_count"]==0
    assert report.completion_certified_under_model
    assert report.learning["base_probe_measurements"]+len(labelled)==report.learning["action_counts"]["probe"]
    for channel in report.cleared_channels:
        assert sum(a["channel"]==channel for a in labelled) <= 6


def test_action_schema_metadata_is_separate_from_feature_width():
    assert action_schema_from_args(Namespace()) == action_schema()
    assert checkpoint_action_schema({}) == action_schema()
    assert controller_for("v3",action_schema("axis_quantiles")) is AxisProbeRLSearch
    assert controller_for("v3",action_schema()) is JointScanRLSearch
    with pytest.raises(ValueError,match="v3 feature"):
        controller_for("v2",action_schema("axis_quantiles"))
    with pytest.raises(ValueError,match="missing action schema"):
        checkpoint_action_schema({"args":{"probe_candidates":"axis_quantiles"}})
    with pytest.raises(ValueError,match="contradicts"):
        checkpoint_action_schema({"action_schema":action_schema(),"args":{"probe_candidates":"axis_quantiles"}})


def test_torch_schema_migration_loading_and_real_update(tmp_path):
    torch = pytest.importorskip("torch")
    from research_rl.network import CandidateActorCritic, load_policy, pack_observations
    from research_rl.train import initialize_from, episode, compute_returns, update, validate_resume, source_manifest
    torch.set_num_threads(1)
    torch.manual_seed(9901)
    base = CandidateActorCritic(16,60)
    extended = CandidateActorCritic(16,60,action_schema=action_schema("axis_quantiles"))
    payload = dict(algorithm=ALGORITHM_VERSIONS["v3"],feature_schema=feature_schema("v3"),
                   hidden=16,model=base.state_dict(),source_manifest=source_manifest())
    with pytest.raises(ValueError,match="action schema change"):
        initialize_from(extended,payload)
    initialize_from(extended,payload,allow_action_schema_change=True)
    assert all(torch.equal(v,extended.state_dict()[k]) for k,v in base.state_dict().items())
    saved = dict(payload,model=extended.state_dict(),action_schema=extended.action_schema)
    with pytest.raises(ValueError,match="action schema change"):
        initialize_from(base,saved)  # Old paired callers must not lose the extension.
    with pytest.raises(ValueError,match="action schema differs"):
        validate_resume(saved,Namespace(feature_version="v3",hidden=16))
    validate_resume(saved,Namespace(feature_version="v3",hidden=16,probe_candidates="axis_quantiles"))
    path=tmp_path/"axis.pt";torch.save(saved,path)
    policy=load_policy(path)
    assert policy.feature_version=="v3" and policy.action_schema==action_schema("axis_quantiles")
    simulator=LocalResearchSimulator(random_scenario(3,110201))
    with pytest.raises(ValueError,match="action schema differ"):
        run_rl_search(simulator.client(),policy=policy,probe_candidates="base")
    records,metrics=episode((110201,extended.state_dict(),16,9902,False,64,None,"v3",
                             extended.architecture,extended.action_distribution,extended.action_schema))
    assert metrics["success"] and metrics["learning"]["action_schema"]==extended.action_schema
    assert metrics["reward_cost_s"]==pytest.approx(metrics["virtual_time_s"])
    logits,_=extended(*pack_observations(records))
    expected=torch.distributions.Categorical(logits=logits).log_prob(torch.tensor([r["action"] for r in records]))
    assert torch.allclose(expected,torch.tensor([r["log_prob"] for r in records]),atol=1e-6)
    compute_returns(records)
    before={k:v.clone() for k,v in extended.state_dict().items()}
    args=Namespace(bc_epochs=1,epochs=1,minibatch=32,clip=.2,value_coef=.5,entropy_coef=.01,
                   aux_bc_coef=0,max_grad_norm=.5,target_kl=.03)
    loss=update(extended,torch.optim.Adam(extended.parameters(),lr=3e-4),records,args)
    assert loss["optimizer_steps"]>0 and any(not torch.equal(v,extended.state_dict()[k]) for k,v in before.items())


def test_explicit_new_trial_initialization_records_changed_action_probabilities(tmp_path):
    torch = pytest.importorskip("torch")
    from research_rl.network import CandidateActorCritic
    from research_rl.train import main
    model=CandidateActorCritic(16,60)
    path=tmp_path/"base.pt"
    torch.save(dict(algorithm=ALGORITHM_VERSIONS["v3"],feature_schema=feature_schema("v3"),
                    hidden=16,model=model.state_dict()),path)
    output=tmp_path/"new"
    args=["--output",str(output),"--device","cpu","--hidden","16","--feature-version","v3",
          "--probe-candidates","axis_quantiles","--initialize-from",str(path),
          "--scenario-start","110221","--updates","1","--episodes-per-update","2",
          "--workers","0","--epochs","1","--max-wall-s","90"]
    assert main(args)==0
    initialized=torch.load(output/"initialized.pt",weights_only=False)
    migration=initialized["state"]["initialization"]
    assert not migration["preserves_initial_probabilities"]
    assert migration["source_action_schema"]==action_schema()
    assert migration["target_action_schema"]==action_schema("axis_quantiles")
    saved=torch.load(output/"latest.pt",weights_only=False)
    assert saved["action_schema"]==action_schema("axis_quantiles") and saved["state"]["update"]==1
    resume=[x for x in args]
    index=resume.index("--initialize-from");resume[index:index+2]=["--resume",str(output/"latest.pt")]
    assert main(resume+["--updates","2"])==0
    assert torch.load(output/"latest.pt",weights_only=False)["state"]["update"]==2
