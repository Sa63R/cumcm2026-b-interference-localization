"""Real public-history execution, unchanged base behavior and PPO provenance."""
import math

import pytest

from localization.omni import OmniCandidateRegion
from research_rl import run_rl_search
from research_rl.action_sets import action_schema, checkpoint_action_schema, controller_for
from research_rl.controller import ALGORITHM_VERSIONS, feature_schema
from research_rl.joint_scan import JointScanRLSearch
from research_rl.range_probes import RangeProbeRLSearch, range_templates, OPTION_ID
from simulation import LocalResearchSimulator, random_scenario
from simulator_client.state import Position
from tests.test_strategy import ObservationOnlyClient


def test_templates_depend_on_robot_approach_and_certify_the_whole_region():
    region=OmniCandidateRegion().observe((0,0),0)
    initial=range_templates(region,(0,0),0)
    moved=range_templates(region,(600,0),0)
    assert moved and moved != initial
    center=region.enclosing_disk().center
    assert all(p.x==pytest.approx((600+center[0])/2) for p in moved)
    assert all(abs(p.y-center[1]/2) in (50.,150.) for p in moved)
    for point in moved:
        assert max(math.hypot(x-point.x,y-point.y) for x,y in region.vertices)<=1000-1e-7
        for i,a in enumerate(region.vertices):
            b=region.vertices[(i+1)%len(region.vertices)]
            assert math.dist((.25*a[0]+.75*b[0],.25*a[1]+.75*b[1]),(point.x,point.y))<=1000
    assert range_templates(None,(0,0),0)==()
    assert range_templates(region,(math.nan,0),0)==()
    assert range_templates(region,(0,0),math.inf)==()


@pytest.mark.parametrize("seed",[110301,110302])
def test_teacher_keeps_all_base_actions_features_and_full_trajectory(seed):
    class CheckedRange(RangeProbeRLSearch):
        def _features(self,candidates,remaining):
            features,context=super()._features(candidates,remaining)
            indices=[i for i,c in enumerate(candidates) if c.option!=OPTION_ID]
            old,old_context=JointScanRLSearch._features(self,[candidates[i] for i in indices],remaining)
            assert context==old_context and [features[i] for i in indices]==old
            assert all(len(f)==60 and all(math.isfinite(v) for v in f) for f in features)
            extra=[c for c in candidates if c.option==OPTION_ID]
            assert len(extra)<=4*16
            assert len({(c.channel,round(c.point.x,6),round(c.point.y,6)) for c in extra})==len(extra)
            for c in extra:
                assert c.kind=="probe"
                assert max(math.dist(v,(c.point.x,c.point.y)) for v in self.regions[c.channel].vertices)<=1000-1e-7
                assert (round(c.point.x,6),round(c.point.y,6)) not in self.observed_positions.get(c.channel,set())
            return features,context
    case=random_scenario(3,seed)
    first,second=LocalResearchSimulator(case),LocalResearchSimulator(case)
    report=CheckedRange(ObservationOnlyClient(first.client()),lambda f,c,t:t).run()
    old=run_rl_search(ObservationOnlyClient(second.client()),policy=lambda f,c,t:t,feature_version="v3")
    assert report.action_history==old.action_history
    assert report.virtual_time_s==old.virtual_time_s
    assert report.learning["range_candidates_offered_sum"]>0
    assert report.learning["range_probe_measurements"]==0
    assert first.evaluation()["all_cleared"]


@pytest.mark.parametrize("seed",[110311,110312])
def test_network_can_execute_new_family_with_public_client_and_safe_fallback(seed):
    def prefer_range(features,context,teacher):
        return next((i for i,row in enumerate(features) if row[1]==1 and row[18]==OPTION_ID/6),teacher)
    sim=LocalResearchSimulator(random_scenario(3,seed))
    report=run_rl_search(ObservationOnlyClient(sim.client()),policy=prefer_range,
                         feature_version="v3",probe_candidates="range_probes")
    labelled=[a for a in report.action_history if a.get("rl_probe_family")=="range_probes"]
    assert labelled and all(a["action"]=="measure" and a["result"]!="no_signal" for a in labelled)
    assert report.learning["range_probe_measurements"]==len(labelled)
    assert report.learning["range_probe_measurements"]+report.learning["base_probe_measurements"]==report.learning["action_counts"]["probe"]
    assert sim.evaluation()["all_cleared"] and sim.evaluation()["failed_clear_count"]==0
    assert report.completion_certified_under_model
    assert report.learning["decisions"]<=256


def test_schema_does_not_silently_change_legacy_or_axis_controller():
    assert controller_for("v3",action_schema("range_probes")) is RangeProbeRLSearch
    assert controller_for("v3",action_schema()) is JointScanRLSearch
    with pytest.raises(ValueError,match="v3 feature"):
        controller_for("v2",action_schema("range_probes"))
    with pytest.raises(ValueError,match="unchanged base"):
        controller_for("v4",action_schema("range_probes"))
    with pytest.raises(ValueError,match="missing action schema"):
        checkpoint_action_schema({"args":{"probe_candidates":"range_probes"}})
    with pytest.raises(ValueError,match="contradicts"):
        checkpoint_action_schema({"action_schema":action_schema(),"args":{"probe_candidates":"range_probes"}})


def test_real_cpu_warm_start_train_resume_and_incompatible_resume_rejection(tmp_path):
    torch=pytest.importorskip("torch")
    from research_rl.network import CandidateActorCritic,load_policy
    from research_rl.train import main
    torch.set_num_threads(1)
    torch.manual_seed(97021)
    parent=CandidateActorCritic(16,60)
    path=tmp_path/"parent.pt"
    torch.save(dict(algorithm=ALGORITHM_VERSIONS["v3"],feature_schema=feature_schema("v3"),
                    hidden=16,model=parent.state_dict()),path)
    output=tmp_path/"run"
    args=["--output",str(output),"--device","cpu","--hidden","16","--feature-version","v3",
          "--probe-candidates","range_probes","--initialize-from",str(path),
          "--scenario-start","110321","--scenario-end","110330","--updates","1",
          "--episodes-per-update","2","--workers","0","--epochs","1","--bc-episodes","0",
          "--max-wall-s","90","--gae-lambda","0.95"]
    assert main(args)==0
    initial=torch.load(output/"initialized.pt",map_location="cpu",weights_only=False)
    assert all(torch.equal(v,initial["model"][k]) for k,v in parent.state_dict().items())
    transfer=initial["state"]["initialization"]
    assert transfer["source_action_schema"]==action_schema()
    assert transfer["target_action_schema"]==action_schema("range_probes")
    assert not transfer["preserves_initial_probabilities"]
    learned=torch.load(output/"latest.pt",map_location="cpu",weights_only=False)
    assert learned["state"]["update"]==1 and learned["state"]["optimizer_steps"]>0
    assert any(not torch.equal(v,learned["model"][k]) for k,v in initial["model"].items())
    assert load_policy(output/"latest.pt").action_schema==action_schema("range_probes")
    resume=args[:];i=resume.index("--initialize-from");resume[i:i+2]=["--resume",str(output/"latest.pt")]
    assert main(resume+["--updates","2"])==0
    resumed=torch.load(output/"latest.pt",map_location="cpu",weights_only=False)
    assert resumed["state"]["update"]==2 and resumed["state"]["episodes"]==4
    assert resumed["state"]["initialization"]["sha256"]==transfer["sha256"]
    with pytest.raises(SystemExit):
        main(resume+["--probe-candidates","base"])
