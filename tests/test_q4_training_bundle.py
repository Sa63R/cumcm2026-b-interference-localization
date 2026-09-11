import pytest

from scripts.q4_training_bundle import validate_config


def config(*names, budget=9):
    return {"run": "wave", "jobs": [dict(name=name, module="q4_rl.micro_train", argv=[
        "--output", f"runs/wave/{name}/training", "--workers", str(budget-1),
        "--cpu-budget", str(budget), "--architecture", "induced" if name == "attention" else "mlp"])
        for name in names]}


def test_multiple_architectures_share_one_run(tmp_path):
    run, jobs = validate_config(config("mlp", "attention", "warm_ppo"), tmp_path)
    assert run == tmp_path/"runs/wave" and len(jobs) == 3
    assert all(job["output"].startswith("runs/wave/") for job in jobs)


def test_combined_compute_budget_is_enforced(tmp_path):
    with pytest.raises(ValueError, match="exceed 50"):
        validate_config(config("one", "two", budget=26), tmp_path)


def test_external_output_and_arbitrary_modules_are_rejected(tmp_path):
    value = config("one")
    value["jobs"][0]["argv"][1] = "../outside"
    with pytest.raises(ValueError):
        validate_config(value, tmp_path)
    value = config("one")
    value["jobs"][0]["module"] = "untrusted"
    with pytest.raises(ValueError):
        validate_config(value, tmp_path)


def test_initial_checkpoint_path_needs_hash_and_cannot_escape(tmp_path):
    value = config("one")
    checkpoint = tmp_path/"warmstart.pt"
    checkpoint.write_bytes(b"test fixture")
    value["jobs"][0]["argv"].extend(["--initialize-micro-warmstart", "warmstart.pt"])
    with pytest.raises(ValueError, match="together"):
        validate_config(value, tmp_path)
    value["jobs"][0]["argv"].extend(["--initialize-sha256", "a"*64])
    assert len(validate_config(value, tmp_path)[1]) == 1


def test_nonempty_output_cannot_be_reused(tmp_path):
    output = tmp_path/"runs/wave/one/training"
    output.mkdir(parents=True)
    (output/"latest.pt").write_bytes(b"existing")
    with pytest.raises(ValueError, match="fresh output"):
        validate_config(config("one"), tmp_path)
