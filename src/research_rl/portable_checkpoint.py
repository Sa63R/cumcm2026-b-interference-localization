"""Portable path metadata and explicit export of trusted training checkpoints.

Run legacy export on the checkpoint's source OS. Model tensors, optimizer,
RNG and source provenance are retained; only pathlib objects become strings.
This does not monkeypatch pathlib or alter action/feature semantics.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePath

import torch


def portable_paths(value):
    """Copy metadata containers without changing tensor/array objects."""
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            converted = portable_paths(key)
            if converted in result:
                raise ValueError('Path normalization would merge dictionary keys')
            result[converted] = portable_paths(item)
        return result
    if isinstance(value, list):
        return [portable_paths(item) for item in value]
    if isinstance(value, tuple):
        return tuple(portable_paths(item) for item in value)
    return value


def path_entries(value, prefix='$'):
    if isinstance(value, PurePath):
        return [{'location': prefix, 'class': type(value).__name__, 'text': str(value)}]
    result = []
    if isinstance(value, dict):
        for key, item in value.items():
            result.extend(path_entries(key, prefix + '.<key>'))
            result.extend(path_entries(item, prefix + '.' + str(key)))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            result.extend(path_entries(item, prefix + '[' + str(index) + ']'))
    return result


def model_tensor_digest(state):
    """Hash named dense tensor values, shape and dtype, independent of .pt ZIP."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise ValueError('Expected dense named model tensors')
        tensor = tensor.detach().cpu().contiguous()
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(',', ':')).encode()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        for block in (header, raw):
            digest.update(len(block).to_bytes(8, 'big'))
            digest.update(block)
    return digest.hexdigest()


def export_checkpoint(input_path, output_path):
    source, target = Path(input_path), Path(output_path)
    if source.resolve() == target.resolve() or target.exists():
        raise ValueError('Export requires a new output path; original checkpoints remain unchanged')
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    payload = torch.load(source, map_location='cpu', weights_only=False)
    entries = path_entries(payload)
    model_hash = model_tensor_digest(payload['model'])
    converted = portable_paths(payload)
    converted['portable_export'] = {
        'version': 1, 'original_file_sha256': source_hash,
        'model_tensor_sha256': model_hash, 'converted_paths': entries,
        'exported_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Metadata paths only; source manifests, algorithm and model values retained'}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.tmp')
    torch.save(converted, temporary)
    reloaded = torch.load(temporary, map_location='cpu', weights_only=False)
    if path_entries(reloaded) or model_tensor_digest(reloaded['model']) != model_hash:
        raise ValueError('Portable export verification failed')
    temporary.replace(target)
    record = {**converted['portable_export'], 'output_file_sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
    target.with_suffix(target.suffix + '.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(export_checkpoint(args.input, args.output), indent=2))


if __name__ == '__main__':
    main()
