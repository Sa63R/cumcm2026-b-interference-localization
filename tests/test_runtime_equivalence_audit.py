"""Round-trip evidence hashes preserve declared integer channel-map keys."""
from copy import deepcopy
import json

import pytest

from experiments.audit_runtime_equivalence import (
    SUMMARY_RUNTIME, digest, recompute_metrics, remove_times,
)


def test_source_estimates_int_key_json_roundtrip_restores_only_declared_field():
    memory = {
        "row": {"program_runtime_s": 1.0},
        "summary": {
            "program_runtime_s": 1.0,
            "action_history": [{"channel": 10, "virtual_time_s": 20.0}],
            "source_estimates": {2: {"center": [2., 0.]}, 10: {"center": [10., 0.]}},
            # Arbitrary numeric-looking string keys must retain their type.
            "strategy_parameters": {"named_values": {"2": "second", "10": "tenth"}},
        },
        "evaluation": {"wall_time_s": 2.0},
        "history": [],
    }
    expected = digest(remove_times(memory["summary"], SUMMARY_RUNTIME))
    raw = json.loads(json.dumps(memory))
    original = deepcopy(raw)
    assert digest(remove_times(raw["summary"], SUMMARY_RUNTIME)) != expected
    assert recompute_metrics(raw)["summary_without_runtime_sha256"] == expected
    assert raw == original  # Auditing must leave raw evidence untouched.


def test_source_estimates_schema_rejects_noncanonical_channel_keys():
    raw = {"row": {}, "summary": {"source_estimates": {"02": {}}, "action_history": []},
           "evaluation": {}, "history": []}
    with pytest.raises(AssertionError):
        recompute_metrics(raw)
