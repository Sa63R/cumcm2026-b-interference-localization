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


def threaded_config(module, threads, *, workers=8, budget=12):
    return {"run": "wave", "jobs": [dict(name="one", module=module, argv=[
        "--output", "runs/wave/one/training", "--workers", str(workers),
        "--cpu-budget", str(budget), "--learner-threads", str(threads)])]}


@pytest.mark.parametrize("module", ["q4_rl.micro_train", "q4_rl.memory_train"])
def test_threaded_learners_must_fit_concurrent_budget(tmp_path, module):
    assert validate_config(threaded_config(module, 4), tmp_path)[1]
    with pytest.raises(ValueError, match="allowance"):
        validate_config(threaded_config(module, 4, budget=11), tmp_path)
    assert validate_config(threaded_config(module, 1, workers=1, budget=1), tmp_path)[1]
    assert validate_config(threaded_config(module, 4, workers=1, budget=4), tmp_path)[1]


@pytest.mark.parametrize("module", ["q4_rl.train", "q4_rl.scst_train"])
def test_other_trainers_cannot_accept_thread_option(tmp_path, module):
    with pytest.raises(ValueError, match="forbidden"):
        validate_config(threaded_config(module, 1), tmp_path)


@pytest.mark.parametrize("threads", ["2.0", "true", "3", "0", "-1"])
def test_thread_count_requires_declared_integer_choice(tmp_path, threads):
    with pytest.raises(ValueError):
        validate_config(threaded_config("q4_rl.micro_train", threads), tmp_path)


@pytest.mark.parametrize("module", ["q4_rl.micro_train", "q4_rl.memory_train"])
def test_gae_is_only_a_declared_finite_actor_estimator(tmp_path, module):
    value = threaded_config(module, 4)
    value["jobs"][0]["argv"].extend(["--gae-lambda", ".97"])
    assert validate_config(value, tmp_path)[1]
    for bad in ("nan", "inf", "-.1", "1.1"):
        value["jobs"][0]["argv"][-1] = bad
        with pytest.raises(ValueError, match="lambda"):
            validate_config(value, tmp_path)


@pytest.mark.parametrize("module", ["q4_rl.train", "q4_rl.scst_train"])
def test_gae_option_is_not_admitted_for_other_modules(tmp_path, module):
    value = {"run": "wave", "jobs": [dict(name="one", module=module, argv=[
        "--output", "runs/wave/one/training", "--workers", "1", "--cpu-budget", "2", "--gae-lambda", ".97"])]}
    with pytest.raises(ValueError, match="forbidden"):
        validate_config(value, tmp_path)


def test_memory_initialization_has_distinct_whitelist_and_task_path(tmp_path):
    (tmp_path/"g3.pt").write_bytes(b"test fixture")
    value = threaded_config("q4_rl.memory_train", 4)
    value["jobs"][0]["argv"].extend(["--initialize-memory-warmstart", "g3.pt", "--initialize-sha256", "a"*64])
    assert validate_config(value, tmp_path)[1]
    value["jobs"][0]["module"] = "q4_rl.micro_train"
    with pytest.raises(ValueError, match="forbidden"):
        validate_config(value, tmp_path)
    value["jobs"][0]["module"] = "q4_rl.memory_train"
    value["jobs"][0]["argv"][-3] = "../outside.pt"
    with pytest.raises(ValueError):
        validate_config(value, tmp_path)
