"""Tiny fixtures only: no simulator, new scene, remote access or real training."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.distributions import Categorical

from q4_rl import micro_network, micro_train, scst_network as net, scst_train as train


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters, **kwargs):
        super().__init__(parameters, **kwargs)
        self.calls = 0

    def step(self, *args, **kwargs):
        self.calls += 1
        return super().step(*args, **kwargs)


def records_for(model, count=2, action=0):
    rows = []
    for i in range(count):
        row = {"global_features": [float(i)/10.] * 13,
               "candidate_features": [[0.] * 50, [1.] * 50],
               "action_index": action, "cost_s": 1., "fallback_cost_s": 0.}
        with torch.no_grad():
            logits, _ = model(*net.pack_observations([row]))
            row["log_prob"] = float(Categorical(logits=logits).log_prob(torch.tensor([action]))[0])
        rows.append(row)
    return rows


def leg_for(model, *, seed=8006000, leg="sample", count=2, baseline_time=4., policy_hash="a"*64):
    records = records_for(model, count) if leg == "sample" else []
    actual = float(count) if leg == "sample" else baseline_time
    metrics = {"seed": seed, "success": True, "actual_time_s": actual, "penalized_time_s": actual,
               "common_lower_bound_s": 1., "failed_clear_count": 0}
    metrics.update({key: .01 for key in ("wall_time_s", "worker_cpu_s", "policy_wall_s", "policy_cpu_s",
                                        "posthoc_bound_wall_s", "posthoc_bound_cpu_s")})
    return {"fixture_only": True, "seed": seed, "leg": leg, "policy_checkpoint_sha256": policy_hash,
            "records": records, "metrics": metrics}


def pair_for(model, count=2, seed=8006000):
    return {"sample": leg_for(model, count=count, seed=seed),
            "greedy": leg_for(model, leg="greedy", seed=seed)}


def frozen_warmstart(path):
    model = micro_network.MicroCandidateActorCritic(hidden=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    micro_train.save_checkpoint(path, model, optimizer,
        {"warmstart_completed": 2, "ppo_batches": 0, "episodes": 2, "pending_batch": None},
        {"warmstart_episodes": 2, "max_decisions": 512, "learning_rate": 3e-4})
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fake_leg(task):
    model = net.model_from_metadata(task["network"])
    model.load_state_dict(task["model"])
    return leg_for(model, seed=task["seed"], leg=task["leg"], policy_hash=task["policy_checkpoint_sha256"])


class SCSTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        net.configure_cpu()

    def setUp(self):
        torch.manual_seed(12)
        random.seed(12)
        self.model = net.SCSTCandidateActorCritic(hidden=4)

    def test_gradient_is_trajectory_mean_of_sums_one_optimizer_step(self):
        pairs = [pair_for(self.model, count=2), pair_for(self.model, count=1, seed=8006001)]
        reference = copy.deepcopy(self.model)
        expected = 0.
        for pair in pairs:
            rows = pair["sample"]["records"]
            logits, _ = reference(*net.pack_observations(rows))
            selected = Categorical(logits=logits).log_prob(torch.zeros(len(rows), dtype=torch.long))
            expected = expected-train.pair_advantage(**pair)*selected.sum()/len(pairs)
        expected.backward()
        optimizer = CountingSGD(self.model.parameters(), lr=.01)
        result = train.scst_update(self.model, optimizer, pairs, minibatch_size=1, max_grad_norm=1e6)
        self.assertEqual(optimizer.calls, 1)
        self.assertEqual(result["sampled_steps"], 3)
        self.assertAlmostEqual(result["loss"], float(expected.detach()), places=7)
        for (name, parameter), (_, expected_parameter) in zip(self.model.named_parameters(), reference.named_parameters()):
            with self.subTest(parameter=name):
                if expected_parameter.grad is None:
                    self.assertIsNone(parameter.grad)
                else:
                    torch.testing.assert_close(parameter.grad, expected_parameter.grad, atol=1e-7, rtol=1e-5)
        self.assertTrue(all(p.grad is None for p in self.model.critic.parameters()))

    def test_greedy_features_truth_and_lower_bound_cannot_enter_loss(self):
        pair = pair_for(self.model)
        pair["greedy"]["records"] = [{"candidate_features": [[float("nan")]]}]
        pair["greedy"]["evaluation"] = {"ground_truth": "not a feature"}
        pair["greedy"]["common_bound"] = {"common_lower_bound_s": 1e30}
        optimizer = CountingSGD(self.model.parameters(), lr=.01)
        train.scst_update(self.model, optimizer, [pair], minibatch_size=1)
        self.assertEqual(optimizer.calls, 1)

    def test_failure_penalty_and_full_fallback_remain_in_advantage(self):
        pair = pair_for(self.model)
        pair["sample"]["records"][-1]["cost_s"] = 101.
        pair["sample"]["records"][-1]["fallback_cost_s"] = 100.
        pair["sample"]["metrics"].update(actual_time_s=102., success=False, penalized_time_s=360000.)
        self.assertAlmostEqual(train.pair_advantage(**pair), (4.-360000.)/1000.)
        accounting = train.attach_returns(pair["sample"]["records"], actual_time_s=102., success=False)
        self.assertEqual(accounting["reward_cost_s"], 360000.)
        self.assertAlmostEqual(pair["sample"]["records"][0]["return"], -360.)
        pair["sample"]["metrics"]["penalized_time_s"] = 102.
        with self.assertRaises(ValueError):
            train.pair_advantage(**pair)

    def test_nonzero_entry_terminal_tail_is_billed_exactly_once(self):
        accounting = {"decision_cost_s": 40., "fallback_cost_s": 60.,
                      "uncovered_cost_s": 3.5, "total_billed_cost_s": 103.5}
        train.verify_billing(103.5, accounting)
        with self.assertRaises(ValueError):
            train.verify_billing(100., accounting)
        accounting["uncovered_cost_s"] = 7.
        with self.assertRaises(ValueError):
            train.verify_billing(103.5, accounting)

    def test_mismatched_scene_or_policy_rejected(self):
        for field, value in (("seed", 8006001), ("policy_checkpoint_sha256", "b"*64)):
            pair = pair_for(self.model)
            pair["greedy"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                train.pair_advantage(**pair)

    def test_no_off_policy_multiple_update_and_no_step_after_stop(self):
        pair = pair_for(self.model)
        pair["sample"]["records"][0]["log_prob"] += 1.
        optimizer = CountingSGD(self.model.parameters(), lr=.01)
        with self.assertRaises(ValueError):
            train.scst_update(self.model, optimizer, [pair])
        self.assertEqual(optimizer.calls, 0)
        with self.assertRaises(train.TrainingStop):
            train.scst_update(self.model, optimizer, [pair_for(self.model)], stop_check=lambda: True)
        self.assertEqual(optimizer.calls, 0)

    def test_frozen_micro_initialization_and_scst_checkpoint_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            warm = root/"warmstart.pt"
            digest = frozen_warmstart(warm)
            model, initialization = net.initialize_micro_warmstart(warm, digest)
            source = torch.load(warm, weights_only=True)
            self.assertEqual(initialization, {"type": "frozen-micro-bc-weights-v1", "sha256": digest})
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, source["model"][name], atol=0., rtol=0.)
            with self.assertRaises(ValueError):
                net.initialize_micro_warmstart(warm, "0"*64)
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
            train.scst_update(model, optimizer, [pair_for(model)])
            checkpoint = root/"scst.pt"
            config = {"learning_rate": 3e-4, "max_decisions": 512}
            state = {"pairs": 1, "pending_batch": {"seeds": [8006001]}}
            train.save_checkpoint(checkpoint, model, optimizer, state, config, initialization)
            expected_rng = (random.random(), torch.rand(3))
            restored, restored_optimizer, restored_state, _, restored_init = train.restore_checkpoint(checkpoint)
            self.assertEqual(random.random(), expected_rng[0])
            torch.testing.assert_close(torch.rand(3), expected_rng[1], atol=0., rtol=0.)
            self.assertEqual(restored_state, state)
            self.assertEqual(restored_init, initialization)
            self.assertEqual(len(restored_optimizer.state), len(optimizer.state))
            for actual, expected in zip(restored.parameters(), model.parameters()):
                torch.testing.assert_close(actual, expected, atol=0., rtol=0.)
            with self.assertRaises(ValueError):
                net.load_policy(warm)
            with self.assertRaises(ValueError):
                micro_network.load_policy(checkpoint)

    def test_reject_post_ppo_or_wrong_horizon_initialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"warm.pt"
            frozen_warmstart(path)
            original = torch.load(path, weights_only=True)
            for mode in ("ppo", "horizon"):
                saved = copy.deepcopy(original)
                if mode == "ppo": saved["state"]["ppo_batches"] = 1
                else: saved["config"]["max_decisions"] = 128
                torch.save(saved, path)
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    net.initialize_micro_warmstart(path, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_individual_raw_leg_written_once_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, index = root/"leg.json.gz", root/"index.jsonl"
            leg = leg_for(self.model)
            train.write_leg(path, leg, index)
            original = path.read_bytes()
            self.assertEqual(json.loads(gzip.decompress(original)), leg)
            with self.assertRaises(FileExistsError): train.write_leg(path, leg, index)
            self.assertEqual(len(index.read_text().splitlines()), 1)
            self.assertEqual(path.read_bytes(), original)

    def test_administrative_resume_replays_pair_without_skipping_or_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            warm, output = root/"warm.pt", root/"run"
            digest = frozen_warmstart(warm)
            common = ["--output", str(output), "--scenario-start", "8006000", "--scenario-end", "8006000",
                      "--batch-pairs", "1", "--minibatch-size", "1", "--max-batches", "1", "--max-wall-seconds", "60",
                      "--deadline", "2099-01-01T00:00:00+00:00"]
            calls = []
            def interrupted(task):
                calls.append((task["seed"], task["leg"]))
                if task["leg"] == "greedy":
                    return {"seed": task["seed"], "leg": task["leg"], "records": [], "administrative_skip": "fixture_stop"}
                return fake_leg(task)
            with patch.object(train, "rollout_leg", interrupted):
                state = train.main(common+["--initialize-micro-warmstart", str(warm), "--initialize-sha256", digest])
            self.assertEqual(state["batches"], 0)
            self.assertEqual(state["pending_batch"]["seeds"], [8006000])
            old_raw = {path.name: path.read_bytes() for path in (output/"raw").glob("*.json.gz")}
            manifest = json.loads(next((output/"raw").glob("*-manifest.json")).read_text())
            self.assertEqual(hashlib.sha256((output/manifest["policy_checkpoint"]).read_bytes()).hexdigest(),
                             manifest["policy_checkpoint_sha256"])
            with patch.object(train, "rollout_leg", fake_leg):
                state = train.main(common+["--resume", str(output/"latest.pt")])
            self.assertEqual(state["pairs"], 1)
            self.assertEqual(state["rollouts"], 2)
            self.assertEqual(state["attempted_rollouts"], 4)
            self.assertIsNone(state["pending_batch"])
            self.assertEqual(len(list((output/"raw").glob("*.json.gz"))), 4)
            for name, content in old_raw.items(): self.assertEqual((output/"raw"/name).read_bytes(), content)

    def test_interrupted_update_restores_weights_optimizer_rng_and_full_reservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            warm, output = root/"warm.pt", root/"run"
            digest = frozen_warmstart(warm)
            arguments = ["--output", str(output), "--initialize-micro-warmstart", str(warm),
                "--initialize-sha256", digest, "--scenario-start", "8006000", "--scenario-end", "8006000",
                "--cpu-budget", "1", "--workers", "1", "--batch-pairs", "1", "--max-batches", "1",
                "--max-wall-seconds", "60", "--deadline", "2099-01-01T00:00:00+00:00"]
            def broken_update(model, optimizer, pairs, **kwargs):
                optimizer.zero_grad()
                sum(parameter.sum() for parameter in model.parameters()).backward()
                optimizer.step()
                torch.rand(5)
                random.random()
                raise train.TrainingStop("fixture interrupt after a partial update")
            with patch.object(train, "rollout_leg", fake_leg), patch.object(train, "scst_update", broken_update):
                state = train.main(arguments)
            before = torch.load(next(output.glob("batch-*-policy.pt")), weights_only=True)
            after = torch.load(output/"latest.pt", weights_only=True)
            self.assertEqual(state["batches"], 0)
            self.assertEqual(state["pending_batch"], before["state"]["pending_batch"])
            self.assertEqual(state["stop_reason"], "administrative_update_rolled_back")
            self.assertEqual(after["optimizer"], before["optimizer"])
            self.assertEqual(after["rng"]["python"], before["rng"]["python"])
            torch.testing.assert_close(after["rng"]["torch"], before["rng"]["torch"], atol=0., rtol=0.)
            for name, value in before["model"].items():
                torch.testing.assert_close(after["model"][name], value, atol=0., rtol=0.)


if __name__ == "__main__":
    unittest.main()
