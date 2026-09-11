"""CPU policy, large fresh partitions and handoff resource guards."""
import importlib.util
from pathlib import Path
import pytest

torch = pytest.importorskip("torch")
from research_rl.cpu_runtime import require_cpu
from research_rl.network import load_policy
from research_rl.train import legal_training_seed, main

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cpu_handoff", root/"scripts/cpu_handoff.py")
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "mps", "xpu"])
def test_non_cpu_device_rejected_before_model_load(device):
    with pytest.raises(ValueError, match="CPU only"):
        load_policy("does-not-exist.pt", device=device)


def test_cpu_guard_does_not_query_accelerator(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("GPU backend queried")
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", forbidden)
    require_cpu()


def test_training_cli_rejects_gpu_before_output_creation(tmp_path):
    with pytest.raises(SystemExit):
        main(["--output", str(tmp_path/"gpu"), "--device", "cuda"])
    assert not (tmp_path/"gpu").exists()


def test_cpu_build_requirement(monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    with pytest.raises(RuntimeError, match="CPU-only"):
        require_cpu()


def test_new_training_ranges_exclude_all_evaluation_partitions():
    assert all(legal_training_seed(s) for s in (1000001, 1199936, 1200001, 1400001, 1900384))
    assert all(not legal_training_seed(s) for s in (6000, 6047, 800000, 810000, 900000, 2100001, 2200001))


def test_workers_respect_cpu_and_memory_limits():
    assert handoff.choose_worker_counts({"available_cpu_slots": 96, "available_memory_gib": 128}) == [16,32,64,96]
    assert handoff.choose_worker_counts({"available_cpu_slots": 24, "available_memory_gib": 128}) == [16,24]
    counts = handoff.choose_worker_counts({"available_cpu_slots": 96, "available_memory_gib": 15})
    assert max(counts) <= 16


def test_training_command_has_no_cuda_and_has_fresh_initialization(tmp_path):
    args = handoff.training_command(tmp_path, seed=1, scenario=1000001, workers=32,
        threads=4, episodes=128, updates=100, seconds=10, checkpoint=tmp_path/"parent.pt")
    assert args[args.index("--device")+1] == "cpu"
    assert "--initialize-from" in args and "--resume" not in args
    assert args[args.index("--gae-lambda")+1] == ".95"
