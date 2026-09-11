"""Isolation, durable budgets, public provenance and Linux process teardown.

No training, simulator, database, server, or object-store connection is used.
Linux-only integration checks report a skip on Windows rather than a pass.
"""

import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location("autonomy_test_" + name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime = load_script("autonomy_runtime")
job = load_script("autonomy_job")
builder = load_script("build_autonomy_package")


def plan():
    from sync_cpu_results import REMOTE_BASE
    return dict(name="isolated-test", trainer="research_rl.train", entrypoint="research_rl:run_rl_search",
                replication=1, blocks=2, block_seconds=1800, remote=REMOTE_BASE + "test-only",
                cpu_slots=50, workers=48, learner_threads=16,
                training_args=["--feature-version", "v3", "--hidden", "96", "--lr", "1e-4"])


@pytest.mark.parametrize("relative", ["../outside", "nested/../../outside"])
def test_private_path_rejects_parent_traversal(tmp_path, relative):
    with pytest.raises(ValueError):
        runtime.inside(tmp_path / "job", relative)


def test_private_path_rejects_absolute_and_sibling_prefix(tmp_path):
    root = tmp_path / "job"
    assert runtime.inside(root, "trial/checkpoint.pt") == root / "trial/checkpoint.pt"
    for path in (root / "checkpoint.pt", tmp_path / "job-other/checkpoint.pt"):
        with pytest.raises(ValueError):
            runtime.inside(root, path)


def test_private_path_resolves_symlink_escape(tmp_path):
    root, other = tmp_path / "job", tmp_path / "outside"
    root.mkdir()
    other.mkdir()
    try:
        (root / "link").symlink_to(other, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("This host does not permit the temporary symlink fixture")
    with pytest.raises(ValueError, match="escaped"):
        runtime.inside(root, "link/checkpoint.pt")


@pytest.mark.parametrize("values,expected", [
    ({}, None),
    ({"/sys/fs/cgroup/cpu.max": "max 100000"}, None),
    ({"/sys/fs/cgroup/cpu.max": "250000 100000"}, 2.5),
    ({"/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "300000",
      "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000"}, 3.0),
    ({"/sys/fs/cgroup/cpu.max": "250000 100000",
      "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us": "150000",
      "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us": "100000"}, 1.5),
    ({"/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "-1",
      "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000"}, None),
])
def test_visible_cgroup_v1_v2_quota_uses_strictest_limit(monkeypatch, values, expected):
    monkeypatch.setattr(runtime, "_text", values.get)
    assert runtime.cpu_quota() == expected


def test_whole_process_affinity_is_clamped_and_accelerators_hidden(monkeypatch, tmp_path):
    affinity = {0, 1, 2, 3, 4, 5}
    changes, nice_changes = [], []

    def set_affinity(pid, cpus):
        assert pid == 0  # Parent itself; future children inherit this mask.
        affinity.clear()
        affinity.update(cpus)
        changes.append(set(cpus))

    topology = "\n\n".join(f"processor: {p}\nphysical id: {s}\ncore id: {c}"
                           for p, s, c in ((0, 0, 0), (1, 0, 1), (2, 1, 0),
                                          (3, 1, 1), (4, 0, 0), (5, 1, 0)))
    monkeypatch.setattr(runtime, "_text", lambda path: topology if path == "/proc/cpuinfo" else None)
    monkeypatch.setattr(runtime, "cpu_quota", lambda: 2.75)
    monkeypatch.setattr(runtime.os, "sched_getaffinity", lambda pid: set(affinity), raising=False)
    monkeypatch.setattr(runtime.os, "sched_setaffinity", set_affinity, raising=False)
    monkeypatch.setattr(runtime.os, "PRIO_PROCESS", 0, raising=False)
    monkeypatch.setattr(runtime.os, "getpriority", lambda kind, pid: 0, raising=False)
    monkeypatch.setattr(runtime.os, "nice", lambda increment: nice_changes.append(increment), raising=False)
    monkeypatch.setattr(runtime.os, "environ", {"CUDA_VISIBLE_DEVICES": "0"})
    result = runtime.constrain(tmp_path, slots=50)
    assert result["cpu_slots"] == 2 and changes == [{0, 2}]
    assert nice_changes == [10]
    assert result["affinity_cpus"] == [0, 2]
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        assert runtime.os.environ[key] == ""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "GOMAXPROCS"):
        assert runtime.os.environ[key] == "1"
    for key in ("TMPDIR", "XDG_CACHE_HOME", "TORCH_HOME", "PYTHONPYCACHEPREFIX"):
        assert Path(runtime.os.environ[key]).is_relative_to(tmp_path / "runtime")


@pytest.mark.parametrize("slots", [0, 61])
def test_cpu_cap_rejected_before_affinity_mutation(slots, tmp_path, monkeypatch):
    def forbidden(*args):
        pytest.fail("Invalid cap attempted to mutate CPU affinity")
    monkeypatch.setattr(runtime.os, "sched_setaffinity", forbidden, raising=False)
    with pytest.raises(ValueError, match="1..60"):
        runtime.constrain(tmp_path, slots=slots)


def test_block_resume_keeps_original_deadline_even_after_downtime():
    state = {}
    first = job.block_state(state, 1, seconds=1800, deadline=10000, now=100)
    assert first["deadline_unix"] == 1900
    persisted = json.loads(json.dumps(state))
    resumed = job.block_state(persisted, 1, seconds=1800, deadline=20000, now=5000)
    assert resumed == first
    assert resumed["deadline_unix"] < 5000  # No replacement 30-minute budget.
    second = job.block_state(persisted, 2, seconds=1800, deadline=5500, now=5000)
    assert second["deadline_unix"] == 5480


@pytest.mark.parametrize("flag,value", [
    ("--output", "elsewhere"), ("--device", "cuda"), ("--workers", "96"),
    ("--num-threads", "96"), ("--scenario-start", "5500001"),
    ("--scenario-end", "5500256"), ("--seed", "1"), ("--max-attempted-episodes", "999999"),
    ("--max-wall-s", "999999"), ("--updates", "999999"),
    ("--deadline-utc", "2030-01-01"), ("--initialize-from", "other.pt"), ("--resume", "other.pt"),
])
def test_plan_cannot_override_controlled_paths_resources_or_partitions(flag, value):
    selected = plan()
    job.validate_plan(selected)
    selected["training_args"].extend([flag, value])
    with pytest.raises(ValueError):
        job.validate_plan(selected)


def test_plan_rejects_duplicate_options_and_private_argument_values():
    for addition in (["--lr", "1"], ["--probe-candidates", "/home/private-user/data"], ["--epochs"]):
        selected = plan()
        selected["training_args"].extend(addition)
        with pytest.raises(ValueError):
            job.validate_plan(selected)


def test_package_verify_binds_actual_source_bytes_and_rejects_escape(monkeypatch, tmp_path):
    monkeypatch.setattr(job, "ROOT", tmp_path)
    monkeypatch.setenv("Q3_SOURCE_COMMIT", "prior")
    (tmp_path / "policy.py").write_bytes(b"original policy")
    manifest = dict(source_commit="a" * 40, files={"policy.py": hashlib.sha256(b"original policy").hexdigest()})
    runtime.write(tmp_path / "PACKAGE_MANIFEST.json", manifest)
    assert job.verify() == manifest
    (tmp_path / "policy.py").write_bytes(b"changed policy")
    with pytest.raises(ValueError, match="differs"):
        job.verify()
    manifest["files"] = {"../unrelated.py": "b" * 64}
    runtime.write(tmp_path / "PACKAGE_MANIFEST.json", manifest)
    with pytest.raises(ValueError):
        job.verify()


def test_public_parent_preserves_exact_tensors_but_omits_private_metadata(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    payload = dict(algorithm="q3-joint-scan-ppo-v3", hidden=16,
                   model={"weight": torch.tensor([[1.25, -3.0], [0, 9.5]], dtype=torch.float32)},
                   args={"output": "/home/private-user/job", "token": "private-test-token"},
                   optimizer={"private": "private-test-token"},
                   state={"update": 7, "episodes": 20,
                          "initialization": {"path": "/home/private-user/parent.pt"}},
                   python_rng="private-test-token")
    source = tmp_path / "parent.pt"
    torch.save(payload, source)
    expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(builder, "PARENT_SHA", expected_hash)
    public = builder.public_parent(source)
    result = torch.load(io.BytesIO(public), map_location="cpu", weights_only=False)
    assert torch.equal(result["model"]["weight"], payload["model"]["weight"])
    assert result["model"]["weight"].device.type == "cpu"
    assert result["initialization_source_sha256"] == expected_hash
    assert result["state"]["update"] == 7 and result["state"]["episodes"] == 20
    assert not ({"args", "optimizer", "python_rng"} & set(result))
    assert "initialization" not in result["state"]
    assert b"private-test-token" not in public and b"/home/private-user" not in public


def test_wrong_parent_digest_is_rejected_before_pickle_load(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    source = tmp_path / "wrong.pt"
    source.write_bytes(b"not the frozen parent")
    monkeypatch.setattr(builder, "PARENT_SHA", "0" * 64)
    loads = []
    monkeypatch.setattr(torch, "load", lambda *a, **k: loads.append(True))
    with pytest.raises((ValueError, AssertionError)):
        builder.public_parent(source)
    assert loads == []


def test_freeze_binds_policy_source_protocol_checkpoint_and_spec():
    identity = dict(package_sha256="a" * 64, protocol_sha256="b" * 64,
                    checkpoint_sha256="c" * 64,
                    spec={"entrypoint": "research_rl:run_rl_search", "kwargs": {"checkpoint": "selected/model.pt"}})
    frozen = dict(policy_identity=copy.deepcopy(identity), frozen_unix=100)
    job.validate_freeze(frozen, identity)
    for key in identity:
        changed = copy.deepcopy(identity)
        changed[key] = {"entrypoint": "changed:run"} if key == "spec" else "f" * 64
        with pytest.raises(ValueError):
            job.validate_freeze(frozen, changed)
    for missing in (None, {}, {"checkpoint_sha256": identity["checkpoint_sha256"]}):
        with pytest.raises(ValueError):
            job.validate_freeze(missing, identity)


def test_public_evaluation_record_redacts_paths_without_changing_physical_data(monkeypatch):
    monkeypatch.setattr(job, "ROOT", Path("/home/private-user/project"))
    record = dict(diagnostic='File "/home/private-user/project/src/policy.py", line 4\n'
                             'File "C:\\Users\\private-user\\python\\lib.py", line 8',
                  row={"errors": ["could not load /home/private-user/model.pt"], "virtual_time_s": 123.456},
                  history=[{"action": "/measure", "position": {"x": -123.5, "y": 900}, "channel": 3}],
                  spec={"kwargs": {"checkpoint": "cpu_runs/run/trial/model.pt"}})
    public = job.public_record(record)
    assert "private-user" not in json.dumps(public)
    assert public["history"] == record["history"]
    assert public["row"]["virtual_time_s"] == 123.456
    assert public["spec"] == record["spec"]


def test_stop_group_signals_owned_group_even_when_leader_already_exited(monkeypatch):
    calls, reaped = [], []
    monkeypatch.setattr(job.os, "killpg", lambda pid, sig: calls.append((pid, sig)), raising=False)
    monkeypatch.setattr(job.signal, "SIGKILL", 9, raising=False)
    process = SimpleNamespace(pid=424242, poll=lambda: 0, wait=lambda: reaped.append(True))
    job.stop_group(process, grace=0)
    assert calls == [(424242, signal.SIGTERM), (424242, 9)]
    assert reaped


@pytest.mark.skipif(sys.platform != "linux", reason="Actual inherited affinity requires Linux; not run on Windows")
def test_linux_descendant_inherits_parent_cpu_mask_and_priority(tmp_path):
    script = (
        "import json,os,subprocess,sys; "
        f"sys.path.insert(0,{str(ROOT / 'scripts')!r}); "
        "from autonomy_runtime import constrain; "
        f"constrain({str(tmp_path)!r},slots=2); "
        "child=subprocess.check_output([sys.executable,'-c',"
        "'import json,os; print(json.dumps([sorted(os.sched_getaffinity(0)),os.getpriority(os.PRIO_PROCESS,0)]))'],text=True); "
        "print(json.dumps({'parent':[sorted(os.sched_getaffinity(0)),os.getpriority(os.PRIO_PROCESS,0)],'child':json.loads(child)}))"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=15, check=True)
    value = json.loads(result.stdout)
    assert value["parent"] == value["child"]
    assert 1 <= len(value["parent"][0]) <= 2 and value["parent"][1] >= 10


@pytest.mark.skipif(sys.platform != "linux", reason="Actual process-group/pipe teardown requires Linux; not run on Windows")
def test_linux_stop_group_closes_orphan_pipe_without_killing_unrelated_process():
    descendant = "import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print(os.getpid(),flush=True); time.sleep(60)"
    leader = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{descendant!r}]); time.sleep(.3)"
    process = subprocess.Popen([sys.executable, "-c", leader], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        assert int(process.stdout.readline().strip()) > 0
        process.wait(timeout=5)  # Leader is gone; its descendant holds stdout.
        assert process.poll() == 0
        started = time.monotonic()
        job.stop_group(process, grace=.1)
        process.communicate(timeout=3)  # EOF proves the descendant released the pipe.
        assert time.monotonic() - started < 5
        assert unrelated.poll() is None
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        unrelated.kill()
        unrelated.wait(timeout=5)
        process.communicate(timeout=5)
