"""Tiny public fixtures only. Run with the G3 source tree on PYTHONPATH."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from experiments import q4_bc_fit_diagnostic as cli
from q4_rl import memory_network, micro_network
from q4_rl.training_journal import EpisodeJournal


def write(path, value):
    Path(path).write_bytes(gzip.compress(json.dumps(value).encode()))


def checkpoint(path, module):
    model = (memory_network.MemoryCandidateActorCritic(hidden=4) if module is memory_network
             else micro_network.MicroCandidateActorCritic(hidden=4))
    torch.save({"version":module.CHECKPOINT_VERSION, "controller_entrypoint":module.CONTROLLER_ENTRYPOINT,
        "feature_schema":module.feature_schema(), "network":model.metadata(), "model":model.state_dict(),
        "device":"cpu", "objective":module.OBJECTIVE, "config":{"architecture":"mlp"}}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def episode(module, *, seed=8001000):
    row = {"global_features":[0.]*13, "candidate_features":[[0.]*module.CANDIDATE_DIM,[1.]*module.CANDIDATE_DIM],
           "action_index":0, "action_kind":"measure"}
    return {"fixture_only":True, "seed":seed, "records":[row, copy.deepcopy(row)],
        "controller_entrypoint":module.CONTROLLER_ENTRYPOINT,
        "controller_learning":{"feature_schema":module.feature_schema(),
            "micro_steps":[{"kind":"measure","role":"cover"}]*2},
        "metrics":{"seed":seed,"split":"train","success":True,"actual_time_s":2.,
                   "penalized_time_s":2.,"common_lower_bound_s":1.,"failed_clear_count":0}}


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        micro_network.configure_cpu()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def run_fixture(self, module, payload=None, *, suffix="run", journal=False):
        ckpt = self.root/f'{suffix}.pt'
        digest = checkpoint(ckpt, module)
        batch = self.root/f'{suffix}.json.gz'
        row = episode(module) if payload is None else payload
        if journal:
            EpisodeJournal(batch, write).append(row)
        else:
            write(batch, [row])
        result = cli.diagnose(batches=[batch], checkpoint=ckpt, checkpoint_sha256=digest,
            policy_loader=f'{module.__name__}:load_policy', output=self.root/suffix)
        return result, ckpt, digest, batch

    def test_g3_strict_real_loader_and_13_58_pack(self):
        result, ckpt, digest, _ = self.run_fixture(memory_network, journal=True)
        self.assertEqual(result['network']['candidate_dim'], 58)
        self.assertEqual(result['groups']['all']['records'], 2)
        self.assertEqual(len(result['inputs']), 2)
        with self.assertRaises(ValueError):
            cli.load_frozen(ckpt, digest, 'q4_rl.micro_network:load_policy')

    def test_g1_old_list_and_journal_same_classification(self):
        torch.manual_seed(11)
        old, _, _, _ = self.run_fixture(micro_network, suffix='old')
        torch.manual_seed(11)
        new, _, _, _ = self.run_fixture(micro_network, suffix='new', journal=True)
        self.assertEqual(old['groups'], new['groups'])

    def test_g3_rejects_truncated_features(self):
        row = episode(memory_network)
        row['records'][0]['candidate_features'] = [[0.]*50,[1.]*50]
        with self.assertRaises(ValueError):
            self.run_fixture(memory_network, row)

    def test_nontraining_or_sampled_labels_rejected(self):
        for i, changed in enumerate(('seed','sample')):
            row = episode(micro_network, seed=8100000 if changed=='seed' else 8001000)
            if changed=='sample': row['records'][0]['log_prob'] = -1.
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.run_fixture(micro_network, row, suffix=f'bad{i}')

    def test_checkpoint_hash_and_loader_allowlist(self):
        path = self.root/'model.pt'
        digest = checkpoint(path, micro_network)
        with self.assertRaises(ValueError):
            cli.load_frozen(path, '0'*64, 'q4_rl.micro_network:load_policy')
        with self.assertRaises(ValueError):
            cli.load_frozen(path, digest, 'unverified:load_policy')

    def test_failed_teacher_episode_retained_and_actual_ratio_not_misleading(self):
        row = episode(micro_network)
        row['metrics'].update(success=False, penalized_time_s=360000.)
        result, _, _, _ = self.run_fixture(micro_network, row)
        self.assertEqual(result['groups']['all']['records'], 2)
        context = result['teacher_rollout_context']
        self.assertEqual(context['all_cleared'], 0)
        self.assertIsNone(context['all_clear_actual_time_over_sum_bound'])
        self.assertEqual(context['sum_penalized_time_over_sum_bound'], 360000.)

    def test_index_hash_and_directory_escape_rejected(self):
        path = self.root/'batch.json.gz'
        EpisodeJournal(path, write).append(episode(micro_network))
        index = json.loads(gzip.decompress(path.read_bytes()))
        index['episodes'][0]['sha256'] = '0'*64
        write(path, index)
        with self.assertRaises(ValueError): list(cli.iter_episodes(path, [], input_number=1))
        index['episodes'][0]['path'] = '../escape.json.gz'
        write(path, index)
        with self.assertRaises(ValueError): list(cli.iter_episodes(path, [], input_number=1))

    def test_database_and_existing_output_rejected(self):
        with self.assertRaises(ValueError): cli.gzip_path(self.root/'practice_training.sqlite3')
        batch = self.root/'batch.json.gz'
        output = self.root/'existing'
        output.mkdir()
        (output/'sentinel').write_text('unchanged')
        with self.assertRaises(FileExistsError):
            cli.diagnose(batches=[batch], checkpoint=self.root/'unused.pt', checkpoint_sha256='0'*64,
                policy_loader='q4_rl.micro_network:load_policy', output=output)
        self.assertEqual((output/'sentinel').read_text(), 'unchanged')


if __name__ == '__main__':
    unittest.main()
