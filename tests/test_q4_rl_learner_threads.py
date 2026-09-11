"""CPU update threads preserve default numerics and full batch transactions."""
from contextlib import nullcontext
import random

import pytest

torch = pytest.importorskip("torch")

from q4_rl import micro_train, memory_train
from q4_rl.learner_threads import learner_update_threads
from tests import test_q4_rl_micro_training as micro_fixture
from tests import test_q4_rl_memory_training as memory_fixture


@pytest.fixture(autouse=True)
def single_thread_runtime():
    micro_train.configure_cpu()
    yield
    micro_train.configure_cpu()


@pytest.fixture(params=[(micro_train, micro_fixture), (memory_train, memory_fixture)],
                ids=["g1", "g3"])
def api(request):
    return request.param


def assert_exact(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_exact(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_exact(x, y)
    else:
        assert a == b


def checkpoint(folder):
    return torch.load(folder/"latest.pt", weights_only=True)


def args_for(fixture, output, threads=1, **options):
    args = fixture.arguments(output)
    args.extend(["--cpu-budget", str(threads), "--learner-threads", str(threads)])
    for flag, value in options.items():
        args.extend(["--"+flag.replace("_", "-"), str(value)])
    return args


def test_default_matches_unwrapped_legacy_driver_exactly(api, tmp_path, monkeypatch):
    train, fixture = api
    def rollout(task):
        assert torch.get_num_threads() == torch.get_num_interop_threads() == 1
        return fixture.fake_rollout(task)
    monkeypatch.setattr(train, "rollout", rollout)
    current = tmp_path/"current"
    train.main(fixture.arguments(current))
    monkeypatch.setattr(train, "learner_update_threads", lambda threads: nullcontext())
    legacy = tmp_path/"legacy"
    train.main(args_for(fixture, legacy))
    a, b = checkpoint(current), checkpoint(legacy)
    for key in ("model", "optimizer", "rng", "config"):
        assert_exact(a[key], b[key])
    assert "learner_threads" not in a["config"]
    for key in ("episodes", "batches", "ppo_batches", "next_seed", "pending_batch", "next_attempt"):
        assert_exact(a["state"][key], b["state"][key])
    # Old checkpoints without the new field resume explicitly at one thread.
    train.main(args_for(fixture, current)+["--resume", str(current/"latest.pt")])
    assert "learner_threads" not in checkpoint(current)["config"]


@pytest.mark.parametrize("threads", [2, 4])
@pytest.mark.parametrize("phase", ["imitation", "ppo"])
def test_actual_update_expands_only_learner_and_restores(api, tmp_path, monkeypatch, threads, phase):
    train, fixture = api
    seen = []
    def rollout(task):
        assert torch.get_num_threads() == torch.get_num_interop_threads() == 1
        return fixture.fake_rollout(task)
    monkeypatch.setattr(train, "rollout", rollout)
    real_update = getattr(train, phase+"_update")
    def update(model, optimizer, records, **kwargs):
        before = {key: value.clone() for key, value in model.state_dict().items()}
        def forward(module, inputs):
            seen.append(torch.get_num_threads())
            assert torch.get_num_threads() == threads
            assert torch.get_num_interop_threads() == 1
        hook = model.register_forward_pre_hook(forward)
        try:
            result = real_update(model, optimizer, records, **kwargs)
        finally:
            hook.remove()
        assert result["updates"] > 0
        assert any(not torch.equal(before[key], value) for key, value in model.state_dict().items())
        return result
    monkeypatch.setattr(train, phase+"_update", update)
    train.main(args_for(fixture, tmp_path, threads, warmstart_episodes=2 if phase == "imitation" else 0))
    assert seen and torch.get_num_threads() == 1
    saved = checkpoint(tmp_path)
    assert saved["config"]["learner_threads"] == threads
    assert saved["state"]["last_progress"]["update"]["learner_threads"] == threads


@pytest.mark.parametrize("threads", [2, 4])
def test_interrupted_real_update_restores_threads_and_replays_transaction(api, tmp_path, monkeypatch, threads):
    train, fixture = api
    actions = []
    def rollout(task):
        assert torch.get_num_threads() == 1
        actions.append((task["seed"], task["action_seed"]))
        return fixture.fake_rollout(task)
    monkeypatch.setattr(train, "rollout", rollout)
    real_update = train.ppo_update
    def interrupted(*args, **kwargs):
        real_update(*args, **kwargs)
        raise train.TrainingStop("interrupt after real multi-thread parameter update")
    monkeypatch.setattr(train, "ppo_update", interrupted)
    output = tmp_path/"resume"
    state = train.main(args_for(fixture, output, threads))
    assert state["episodes"] == 0 and state["pending_batch"] is not None
    assert torch.get_num_threads() == 1
    reserved = checkpoint(output)
    initial = torch.load(output/"random.pt", weights_only=True)
    for key in ("model", "optimizer"):
        assert_exact(reserved[key], initial[key])
    raw = output/"batch-000000-attempt-000000.json.gz"
    original_bytes = raw.read_bytes()
    for changed in (1, 4 if threads == 2 else 2):
        with pytest.raises(SystemExit):
            train.main(args_for(fixture, output, changed)+["--resume", str(output/"latest.pt")])
    assert raw.read_bytes() == original_bytes
    monkeypatch.setattr(train, "ppo_update", real_update)
    train.main(args_for(fixture, output, threads)+["--resume", str(output/"latest.pt")])
    n = len(actions)//2
    assert actions[:n] == actions[n:]
    assert raw.read_bytes() == original_bytes
    control = tmp_path/"control"
    train.main(args_for(fixture, control, threads))
    a, b = checkpoint(output), checkpoint(control)
    for key in ("model", "optimizer", "rng", "config"):
        assert_exact(a[key], b[key])
    assert torch.get_num_threads() == 1


@pytest.mark.parametrize("workers,threads,budget,valid", [
    (1, 1, 1, True), (1, 2, 2, True), (1, 4, 4, True),
    (8, 2, 10, True), (8, 4, 12, True), (8, 4, 11, False),
    (1, 4, 3, False), (1, 3, 4, False), (0, 1, 2, False), (1, 1, 61, False)])
def test_parser_budget_is_aware_of_serial_or_parallel_rollout(api, tmp_path, workers, threads, budget, valid):
    train, fixture = api
    args = args_for(fixture, tmp_path, threads, workers=workers, cpu_budget=budget)
    if valid:
        train._arguments(args)
    else:
        with pytest.raises(SystemExit):
            train._arguments(args)


def test_real_worker_entry_reasserts_single_thread_before_scene(api):
    train, _ = api
    torch.set_num_threads(4)
    result = train.rollout(dict(seed=8005000, action_seed=1, mode="ppo", deadline_epoch=0.))
    assert result["administrative_skip"] == "deadline_before_start"
    assert torch.get_num_threads() == torch.get_num_interop_threads() == 1


@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
def test_context_restores_after_all_exception_types_without_rng_change(exception):
    before_torch, before_python = torch.get_rng_state().clone(), random.getstate()
    with pytest.raises(exception):
        with learner_update_threads(4):
            assert torch.get_num_threads() == 4
            raise exception("fixture")
    assert torch.get_num_threads() == 1
    assert torch.equal(before_torch, torch.get_rng_state())
    assert before_python == random.getstate()
