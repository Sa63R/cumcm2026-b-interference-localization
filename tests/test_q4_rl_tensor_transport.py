"""Numerical and byte equivalence of the measured training I/O speedups."""
import gzip
import io
import json
import math

import pytest
import torch

from q4_rl.network import pack_observations, configure_cpu
from q4_rl.train import _write_batch


def test_tensor_pack_is_bit_identical_to_old_scalar_validated_pack():
    configure_cpu()
    rows = []
    for count in (1, 7, 3):
        rows.append({"global_features": [(-1)**i * (i+.3)/17 for i in range(10)],
                     "candidate_features": [[math.sin(i+j)/3 for j in range(16)] for i in range(count)]})
    expected_g = torch.zeros(3, 10)
    expected_c = torch.zeros(3, 7, 16)
    expected_mask = torch.zeros(3, 7, dtype=torch.bool)
    for i, row in enumerate(rows):
        assert all(math.isfinite(float(v)) for v in row["global_features"])
        assert all(math.isfinite(float(v)) for candidate in row["candidate_features"] for v in candidate)
        expected_g[i] = torch.tensor(row["global_features"], dtype=torch.float32)
        expected_c[i, :len(row["candidate_features"])] = torch.tensor(row["candidate_features"], dtype=torch.float32)
        expected_mask[i, :len(row["candidate_features"])] = True
    for actual, expected in zip(pack_observations(rows), (expected_g, expected_c, expected_mask)):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"), 1e100])
@pytest.mark.parametrize("field", ["global_features", "candidate_features"])
def test_nonfinite_or_float32_overflow_is_rejected(value, field):
    row = {"global_features": [0.]*10, "candidate_features": [[0.]*16]}
    if field == "global_features":
        row[field][0] = value
    else:
        row[field][0][0] = value
    with pytest.raises(ValueError, match="non-finite"):
        pack_observations([row])


def test_compressed_batch_has_identical_decoded_bytes(tmp_path):
    value = [{"中文": "真实观测", "cost_s": 1.234567, "features": [0., -0., 1e-30, True, None]},
             {"failed": True, "return": -360.0}]
    old = io.StringIO()
    json.dump(value, old, ensure_ascii=False, allow_nan=False)
    path = tmp_path / "batch.json.gz"
    _write_batch(path, value)
    assert gzip.decompress(path.read_bytes()) == old.getvalue().encode("utf-8")
    assert not path.with_suffix(".gz.tmp").exists()
