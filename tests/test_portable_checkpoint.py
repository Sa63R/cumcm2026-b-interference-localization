from argparse import Namespace
from pathlib import Path, PurePosixPath, PureWindowsPath
import pickle

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from research_rl.controller import CONTEXT_DIM, FEATURE_DIMS
from research_rl.network import CandidateActorCritic, load_policy
from research_rl.portable_checkpoint import export_checkpoint, model_tensor_digest, path_entries, portable_paths
from research_rl.train import save_checkpoint, validate_resume


def test_normalizes_both_path_flavors_without_touching_arrays_or_tensor_storage():
    tensor, array = torch.tensor([1., 2.]), np.arange(5)
    original = {'path': PurePosixPath('/training/model.pt'), 'nested':
                [PureWindowsPath('D:/training/model.pt'), (tensor, array)]}
    converted = portable_paths(original)
    assert converted['path'] == '/training/model.pt'
    assert converted['nested'][0] == 'D:\\training\\model.pt'
    assert converted['nested'][1][0] is tensor
    assert converted['nested'][1][1] is array
    assert isinstance(original['path'], PurePosixPath)
    assert not path_entries(converted)


def test_dictionary_path_collision_is_rejected():
    with pytest.raises(ValueError, match='merge dictionary keys'):
        portable_paths({PurePosixPath('/x'): 1, '/x': 2})


def test_training_checkpoint_paths_and_export_preserve_model_optimizer_and_rng(tmp_path):
    torch.manual_seed(55)
    model = CandidateActorCritic(16, FEATURE_DIMS['v3'])
    optimizer = torch.optim.Adam(model.parameters())
    sum(parameter.square().sum() for parameter in model.parameters()).backward()
    optimizer.step()
    args = Namespace(feature_version='v3', architecture='mlp', hidden=16, group_alpha=0,
                     output=tmp_path, initialize_from=PurePosixPath('/old/checkpoint.pt'))
    state = {'update': 1, 'nested': (Path('relative'), PureWindowsPath('D:/old/trial'))}
    source = tmp_path / 'training.pt'
    save_checkpoint(source, model, optimizer, args, state)
    original = torch.load(source, map_location='cpu', weights_only=False)
    assert not path_entries(original)
    validate_resume(original, args)
    # Also exercise export for a legacy payload with native concrete paths.
    original['args']['output'] = tmp_path
    torch.save(original, source)
    before = source.read_bytes()
    destination = tmp_path / 'portable.pt'
    record = export_checkpoint(source, destination)
    result = torch.load(destination, map_location='cpu', weights_only=False)
    assert source.read_bytes() == before
    assert record['converted_paths'][0]['location'] == '$.args.output'
    assert result['source_manifest'] == original['source_manifest']
    assert model_tensor_digest(result['model']) == model_tensor_digest(original['model'])
    assert torch.equal(result['torch_rng'], original['torch_rng'])
    assert pickle.dumps(result['numpy_rng']) == pickle.dumps(original['numpy_rng'])
    assert result['python_rng'] == original['python_rng']
    for key in original['optimizer']['state']:
        for name, value in original['optimizer']['state'][key].items():
            other = result['optimizer']['state'][key][name]
            assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other
    policy = load_policy(destination)
    features = torch.rand(1, 3, FEATURE_DIMS['v3'])
    context, mask = torch.rand(1, CONTEXT_DIM), torch.ones(1, 3, dtype=torch.bool)
    model.eval()
    with torch.no_grad():
        expected = model(features, context, mask)
        actual = policy.model(features, context, mask)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    with pytest.raises(ValueError, match='new output path'):
        export_checkpoint(source, destination)
    with pytest.raises(ValueError, match='new output path'):
        export_checkpoint(source, source)


def test_model_digest_detects_values_names_and_shape():
    digest = model_tensor_digest({'x': torch.arange(4, dtype=torch.float32)})
    assert digest != model_tensor_digest({'y': torch.arange(4, dtype=torch.float32)})
    assert digest != model_tensor_digest({'x': torch.arange(4, dtype=torch.float32).reshape(2, 2)})
    assert digest != model_tensor_digest({'x': torch.arange(4, dtype=torch.float32) + 1})
