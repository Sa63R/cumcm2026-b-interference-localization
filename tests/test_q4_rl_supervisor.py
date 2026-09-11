import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("q4_supervisor", Path(__file__).resolve().parents[1] / "scripts/q4_cpu_supervisor.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_cpu_allowance_respects_shared_quota_and_existing_work():
    assert module.cpu_allowance(50, 160, 96) == 50
    assert module.cpu_allowance(60, 160, 96, other_cores=40) == 48
    assert module.cpu_allowance(50, 160, 96, other_cores=80) == 8
    assert module.cpu_allowance(50, 6, 96) == 6
    assert module.cpu_allowance(50, 160, 4) == 1
    for request in (0, 61, True):
        with pytest.raises(ValueError):
            module.cpu_allowance(request, 160, 96)


def test_detects_q4_v1_actual_quota(monkeypatch):
    def read(path, default=None):
        if str(path).endswith("cpu.cfs_quota_us"):
            return 9600000
        if str(path).endswith("cpu.cfs_period_us"):
            return 100000
        return default
    monkeypatch.setattr(module, "read_number", read)
    assert module.quota_cores() == 96
