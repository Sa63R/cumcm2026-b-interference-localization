"""Analytic probabilities and actual PPO consistency for the grouped ablation."""

from argparse import Namespace
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research_rl.controller import ALGORITHM_VERSIONS, CONTEXT_DIM, Candidate, feature_schema
from research_rl.distributions import (adjusted_logits, checkpoint_distribution,
    distribution_from_args, distribution_spec, group_sizes, merge_probe_diagnostics,
    sample_probe_diagnostics, validate_distribution)
from research_rl.network import CandidateActorCritic, TorchPolicy, load_policy, pack_observations
from research_rl.train import compute_returns, episode, initialize_from, main, source_manifest, update, validate_resume, legal_training_seed
from simulator_client.state import Position


def features():
    rows = np.zeros((5, 60), dtype=np.float32)
    rows[:3, 0] = 1
    rows[:3, 45] = np.arange(1, 4) / 20
    rows[3:, 1] = 1
    rows[3:, 4] = [0.3, 0.4]
    rows[3:, 45] = 4 / 20
    return rows


def tensors():
    return pack_observations([dict(features=features(), context=[0.0] * CONTEXT_DIM)])


def payload(model, *, legacy=False):
    result = dict(algorithm=ALGORITHM_VERSIONS["v3"], hidden=model.hidden,
                  feature_schema=feature_schema("v3"), model=model.state_dict(),
                  architecture=model.architecture, source_manifest=source_manifest())
    if not legacy:
        result["action_distribution"] = model.action_distribution
    return result


def test_group_sizes_and_analytic_equal_score_probabilities():
    f, _, mask = tensors()
    logits = torch.zeros((1, 5))
    assert group_sizes(f, mask).tolist() == [[3, 3, 3, 2, 2]]
    grouped = adjusted_logits(logits, f, mask, distribution_spec(1))
    expected = torch.tensor([[1/6, 1/6, 1/6, 1/4, 1/4]])
    assert torch.allclose(grouped.softmax(-1), expected)
    assert adjusted_logits(logits, f, mask, distribution_spec(0)) is logits


def test_flat_legacy_frozen_network_logits_and_sampled_trajectory_are_exact():
    # Exact git-show fixture from d33074b, rather than a rewritten reference
    # implementation that could accidentally repeat a current bug.
    path = Path(__file__).parent / "fixtures" / "rl_network_d33074b.py"
    spec = importlib.util.spec_from_file_location("research_rl._frozen_d33074b", path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    from research_rl import run_rl_search
    from simulation import LocalResearchSimulator, random_scenario
    torch.set_num_threads(1)
    torch.manual_seed(923)
    previous = legacy.CandidateActorCritic(16, 60).eval()
    current = CandidateActorCritic(16, 60).eval()
    current.load_state_dict(previous.state_dict())
    for old, new in zip(previous(*tensors()), current(*tensors())):
        assert torch.equal(old, new)
    results = []
    for policy in (legacy.TorchPolicy(previous, deterministic=False),
                   TorchPolicy(current, deterministic=False, capture_diagnostics=True)):
        simulator = LocalResearchSimulator(random_scenario(3, 100731))
        torch.manual_seed(924)
        report = run_rl_search(simulator.client(), policy=policy, feature_version="v3", max_decisions=80)
        assert simulator.evaluation()["all_cleared"]
        results.append(report)
    assert results[0].virtual_time_s == results[1].virtual_time_s
    assert results[0].action_history == results[1].action_history


def test_group_sampling_is_equivalent_to_logmeanexp_group_then_within_group():
    f, _, mask = tensors()
    raw = torch.tensor([[2.0, -1.0, 0.0, 0.4, -0.8]])
    actual = adjusted_logits(raw, f, mask, distribution_spec(1)).softmax(-1)
    group_scores = torch.stack((torch.logsumexp(raw[0, :3], 0)-math.log(3),
                                torch.logsumexp(raw[0, 3:], 0)-math.log(2)))
    group = group_scores.softmax(0)
    expected = torch.cat((group[0]*raw[0, :3].softmax(0), group[1]*raw[0, 3:].softmax(0)))
    assert torch.allclose(actual[0], expected)
    # Within-source conditional probabilities do not change just from alpha.
    assert torch.allclose(actual[0, 3:] / actual[0, 3:].sum(), raw[0, 3:].softmax(0))


def test_grouped_likelihood_gradient_matches_analytic_distribution():
    f, _, mask = tensors()
    raw = torch.zeros((1, 5), requires_grad=True)
    logits = adjusted_logits(raw, f, mask, distribution_spec(1))
    (-torch.log_softmax(logits, -1)[0, 4]).backward()
    assert torch.allclose(raw.grad, torch.tensor([[1/6, 1/6, 1/6, 1/4, -3/4]]))


def test_masked_padding_and_permutation_preserve_group_probabilities():
    f, context, mask = tensors()
    f = torch.cat((f, f[:, :1]), 1)
    mask = torch.cat((mask, torch.zeros((1, 1), dtype=torch.bool)), 1)
    torch.manual_seed(921)
    model = CandidateActorCritic(16, 60, action_distribution=distribution_spec(1)).eval()
    order = [4, 2, 5, 0, 3, 1]
    with torch.no_grad():
        original, value = model(f, context, mask)
        permuted, value2 = model(f[:, order], context, mask[:, order])
        alone, value3 = model(f[:, :5], context, mask[:, :5])
    assert torch.allclose(permuted, original[:, order], atol=1e-6)
    assert torch.allclose(value, value2, atol=1e-6)
    assert torch.allclose(alone, original[:, :5], atol=1e-6)
    assert torch.allclose(value, value3, atol=1e-6)
    assert original.softmax(-1)[0, 5].item() == 0


def test_diagnostics_measure_source_conditional_entropy_not_full_action_entropy():
    rows = features()
    logits = np.array([7.0, 7.0, 7.0, math.log(3), 0.0])
    stats = sample_probe_diagnostics(logits, rows, [False, False, False, True, False], 4)
    expected_entropy = -0.75*math.log(0.75)-0.25*math.log(0.25)
    assert stats["entropy_sum"] == pytest.approx(expected_entropy)
    assert stats["center_probability_sum"] == pytest.approx(0.75)
    assert stats["selected_probes"] == 1 and stats["selected_center_probes"] == 0
    merged = merge_probe_diagnostics([stats, stats])
    assert merged["source_groups"] == 2
    assert merged["conditional_entropy"] == pytest.approx(expected_entropy)
    assert merged["normalized_conditional_entropy"] == pytest.approx(expected_entropy/math.log(2))
    assert merged["center_conditional_probability"] == pytest.approx(0.75)
    assert merge_probe_diagnostics([])["conditional_entropy"] is None


def test_center_metadata_is_legal_geometry_not_teacher_option_number():
    model = CandidateActorCritic(16, 60)
    policy = TorchPolicy(model, capture_diagnostics=True)
    candidates = [Candidate("probe", 1, Position(5, 0), option=0),
                  Candidate("probe", 1, Position(0, 0), option=2)]
    regions = {1: SimpleNamespace(enclosing_disk=lambda: SimpleNamespace(center=(0, 0)))}
    policy.prepare_candidates(candidates, regions)
    assert policy.center_mask == [False, True]


def test_explicit_distribution_migration_changes_probability_not_weights_or_values():
    old = CandidateActorCritic(16, 60).eval()
    new = CandidateActorCritic(16, 60, action_distribution=distribution_spec(1)).eval()
    with pytest.raises(ValueError, match="distribution change"):
        initialize_from(new, payload(old, legacy=True))
    initialize_from(new, payload(old, legacy=True), allow_distribution_change=True)
    assert all(torch.equal(value, new.state_dict()[key]) for key, value in old.state_dict().items())
    f, context, mask = tensors()
    before, value = old(f, context, mask)
    after, value2 = new(f, context, mask)
    assert torch.equal(value, value2)
    assert torch.allclose(after, before-torch.log(torch.tensor([[3,3,3,2,2]])))
    with pytest.raises(ValueError, match="distribution change"):
        initialize_from(old, payload(new))  # Old paired callers must reject it.


def test_distribution_checkpoint_loading_and_strict_resume(tmp_path):
    model = CandidateActorCritic(16, 60, action_distribution=distribution_spec(1))
    path = tmp_path / "grouped.pt"
    torch.save(payload(model), path)
    policy = load_policy(path)
    assert policy.action_distribution == distribution_spec(1)
    rows = features()
    context = [0.0] * CONTEXT_DIM
    action, logprob, _ = policy(rows, context, 0)
    logits, _ = model(*tensors())
    assert action == logits.argmax(-1).item()
    assert logprob == pytest.approx(torch.log_softmax(logits, -1)[0, action].item())
    args = Namespace(feature_version="v3", hidden=16, group_alpha=1)
    validate_resume(payload(model), args)
    with pytest.raises(ValueError, match="distribution differs"):
        validate_resume(payload(model), Namespace(feature_version="v3", hidden=16))
    assert distribution_from_args(Namespace()) == distribution_spec(0)
    assert checkpoint_distribution({}) == distribution_spec(0)
    with pytest.raises(ValueError, match="missing action distribution"):
        checkpoint_distribution(dict(args={"group_alpha": 1}))
    with pytest.raises(ValueError, match="contradicts"):
        checkpoint_distribution(dict(action_distribution=distribution_spec(0), args={"group_alpha": 1}))


@pytest.mark.parametrize("value", [-1, 0.5, 2, True, None])
def test_invalid_group_alpha_rejected(value):
    with pytest.raises(ValueError):
        distribution_spec(value)


def test_grouped_mode_rejects_wrong_feature_schema_and_unknown_metadata():
    with pytest.raises(ValueError, match="v3 feature"):
        CandidateActorCritic(16, 44, action_distribution=distribution_spec(1))
    with pytest.raises(ValueError):
        validate_distribution(dict(version=2, name="flat"))


def test_grouped_episode_old_logprob_matches_actual_ppo_distribution():
    torch.set_num_threads(1)
    torch.manual_seed(922)
    model = CandidateActorCritic(16, 60, action_distribution=distribution_spec(1))
    records, metrics = episode((100701, model.state_dict(), 16, 912, False, 64,
                                None, "v3", model.architecture, model.action_distribution))
    assert metrics["success"] and metrics["reward_cost_s"] == pytest.approx(metrics["virtual_time_s"])
    logits, values = model(*pack_observations(records))
    actions = torch.tensor([r["action"] for r in records])
    expected = torch.distributions.Categorical(logits=logits).log_prob(actions)
    assert torch.allclose(expected, torch.tensor([r["log_prob"] for r in records]), atol=1e-6)
    assert metrics["sampling_probe_diagnostics"]["source_groups"] > 0
    compute_returns(records)
    before = {k:v.clone() for k,v in model.state_dict().items()}
    args = Namespace(bc_epochs=1, epochs=1, minibatch=32, clip=0.2, value_coef=0.5,
                     entropy_coef=0.01, aux_bc_coef=0, max_grad_norm=0.5, target_kl=0.03)
    result = update(model, torch.optim.Adam(model.parameters(), lr=3e-4), records, args)
    assert result["optimizer_steps"] > 0
    assert any(not torch.equal(before[k],v) for k,v in model.state_dict().items())


def test_grouped_training_saves_distribution_and_resumes(tmp_path):
    args = ["--output", str(tmp_path), "--device", "cpu", "--hidden", "16",
            "--feature-version", "v3", "--group-alpha", "1", "--scenario-start", "100711",
            "--bc-episodes", "2", "--bc-epochs", "1", "--updates", "1",
            "--episodes-per-update", "2", "--workers", "0", "--epochs", "1", "--max-wall-s", "90"]
    assert main(args) == 0
    saved = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert saved["action_distribution"] == distribution_spec(1)
    assert saved["state"]["update"] == 1
    assert main(args+["--resume",str(tmp_path/"latest.pt"),"--updates","2"]) == 0
    saved = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert saved["state"]["update"] == 2
    logs = [json.loads(line) for line in (tmp_path/"training.jsonl").read_text().splitlines()]
    assert all("sampling_probe_diagnostics" in entry for entry in logs)


@pytest.mark.parametrize("seed, allowed", [(99999, False), (100000, False), (100001, True),
    (199999, True), (200000, False), (1999, False), (2000, True), (5099, True),
    (5100, False), (6000, False), (6047, False)])
def test_training_seed_boundaries_follow_frozen_protocol(seed, allowed):
    assert legal_training_seed(seed) is allowed
