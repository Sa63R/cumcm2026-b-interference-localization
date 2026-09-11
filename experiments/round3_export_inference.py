"""Strip training/private metadata while preserving the frozen inference model.

Only run on the user's trusted checkpoint. No training or case evaluation occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repository', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.source.resolve() == args.output.resolve():
        raise ValueError('A new output is required; never overwrite the source')
    sys.path.insert(0, str(args.repository.resolve() / 'src'))
    import torch
    from research_rl.network import load_policy
    from research_rl.portable_checkpoint import model_tensor_digest
    from research_rl.controller import CONTEXT_DIM

    torch.set_num_threads(1)
    source = torch.load(args.source, map_location='cpu', weights_only=False)
    required = ('algorithm', 'hidden', 'architecture', 'action_distribution',
                'action_schema', 'feature_version', 'feature_schema', 'model')
    clean = {key: source[key] for key in required if key in source}
    # Metadata must be JSON primitives, never path objects or arbitrary pickle.
    json.dumps({key: value for key, value in clean.items() if key != 'model'})
    original_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    model_hash = model_tensor_digest(source['model'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(clean, args.output)
    reloaded = torch.load(args.output, map_location='cpu', weights_only=True)
    assert set(reloaded) == set(clean)
    assert model_tensor_digest(reloaded['model']) == model_hash
    old = load_policy(args.source, device='cpu', deterministic=True)
    new = load_policy(args.output, device='cpu', deterministic=True)
    generator = torch.Generator(device='cpu').manual_seed(54019)
    checks = []
    # Synthetic tensors check exact inference equivalence, not performance.
    with torch.no_grad():
        for count in (1, 2, 7, 23, 50):
            features = torch.randn(2, count, old.model.feature_dim, generator=generator)
            context = torch.randn(2, CONTEXT_DIM, generator=generator)
            mask = torch.ones(2, count, dtype=torch.bool)
            if count > 1:
                mask[1, -1] = False
            old_logits, old_value = old.model(features, context, mask)
            new_logits, new_value = new.model(features, context, mask)
            assert torch.equal(old_logits, new_logits)
            assert torch.equal(old_value, new_value)
            checks.append({'candidates': count, 'logits_bitwise_equal': True,
                           'value_bitwise_equal': True})
    report = {'original_file_sha256': original_hash,
              'inference_file_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
              'model_tensor_sha256': model_hash,
              'retained_keys': sorted(clean), 'named_tensors': len(source['model']),
              'checks': checks,
              'scope': 'Same network and inference schema; training metadata removed. No retraining.'}
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
