"""Synthetic JSON only: no real scenario generation, policy or sealed data."""
import gzip
import json

import pytest

from experiments import research_v1_selection as selection


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(selection.subprocess, "check_output", lambda *a, **k: "a" * 40)
    protocol = dict(name="synthetic-selection-only", primary_reference="rollout_frozen", rollout_config={"particles": 4},
        partitions={name: dict(seed_start=start, seed_stop_exclusive=start + 2)
                    for name, start in (("validation_extended", 1), ("final_random", 11), ("final_stress", 21))},
        limits=dict(virtual_seconds_per_case=360000))
    protocol_path, environment = tmp_path / "protocol.json", tmp_path / "environment.json"
    write(protocol_path, protocol)
    write(environment, dict(system="Linux", python_version="3.12.14", packages={"numpy": "2.2.6", "torch": "2.9.1+cpu"}))
    rule = tmp_path / "rule.md"
    rule.write_text("Synthetic rule fixture, no real selection data.", encoding="utf-8")

    def candidate(name):
        source = tmp_path / name
        for filename in ("src/simulation/engine.py", "src/simulator_client/client.py",
                         "experiments/research_v1_eval.py", "experiments/run_q3_comparison.py"):
            path = source / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# identical synthetic source bytes\n", encoding="utf-8")
        write(source / "research/v1_protocol.json", protocol)
        (source / "weights.bin").write_bytes(b"synthetic checkpoint, not a model")
        spec = dict(name=name, entrypoint="synthetic:forbidden", kwargs={"checkpoint": "weights.bin"})
        if name == "rollout_frozen":
            spec = dict(name=name, entrypoint="strategies:run_search",
                        kwargs=dict(variant="rollout", rollout_config=protocol["rollout_config"]))
        spec_path, freeze_path = source / "spec.json", source / "freeze.json"
        write(spec_path, spec)
        write(freeze_path, dict(git_commit="a" * 40, frozen_at="2020-01-01T00:00:00+00:00", strategy=name,
                               identity=selection.live_identity(source, spec, protocol)))
        return dict(id=name, source_dir=str(source), spec=str(spec_path), freeze=str(freeze_path), environment=str(environment))

    baseline = candidate("rollout_frozen")
    directions = {name: dict(candidates=[candidate(name + "_a"), candidate(name + "_b")], fallback=name + "_a")
                  for name in selection.DIRECTIONS}
    plan = dict(protocol=str(protocol_path), rule=str(rule), baseline=baseline, directions=directions)
    plan_path, registry_path = tmp_path / "plan.json", tmp_path / "registry.json"
    write(plan_path, plan)
    return dict(root=tmp_path, plan=plan, plan_path=plan_path, registry_path=registry_path, protocol=protocol)


def registered(setup):
    return selection.register(setup["plan_path"], setup["registry_path"])


def evaluations(setup, registry, split="validation_extended", ids=None, costs=None):
    paths = {}
    for name in ids or registry["entries"]:
        entry = registry["entries"][name]
        directory = setup["root"] / "evaluations" / split / name
        part = setup["protocol"]["partitions"][split]
        seeds = list(range(part["seed_start"], part["seed_stop_exclusive"]))
        rows = []
        cost = (costs or {}).get(name, 1000. if name == registry["baseline"] else 900.)
        for seed in seeds:
            truth = dict(problem=3, case_id=f"synthetic-{split}-{seed}", seed=seed, sources=[])
            row = dict(case_id=truth["case_id"], seed=seed, case_sha256=selection.digest(truth), strategy=name,
                       successful=True, all_cleared=True, completion_certified=True, accepted_exit=True,
                       source_total=10, cleared_total=10, failed_clear_count=0, measurement_count=1,
                       action_count=3, errors=[], virtual_time_s=cost, penalized_time_s=cost,
                       program_runtime_s=.01, movement_s=cost - 50, detection_s=0., switching_s=0., optical_s=30., removal_s=20.)
            evaluation = {key: row[key] for key in ("all_cleared", "source_total", "cleared_total", "failed_clear_count",
                                                   "measurement_count", "action_count", "virtual_time_s")}
            evaluation.update(kind="local_research_only", ground_truth=truth,
                              time_breakdown_s={key: row[key] for key in selection.COMPONENTS})
            record = dict(row=row, spec=entry["spec_value"], evaluation=evaluation, history=[{}, {}, {}],
                          evaluation_phase="after_policy_termination", summary=None)
            directory.mkdir(parents=True, exist_ok=True)
            with gzip.open(directory / f"case-{seed}.json.gz", "wt", encoding="utf-8") as handle:
                json.dump(record, handle)
            rows.append(row)
        write(directory / "manifest.json", dict(identity=entry["identity"], strategy=entry["strategy"], split=split, seeds=seeds))
        write(directory / "rows.json", rows)
        paths[name] = str(directory)
    return paths


def change_record(directory, seed, action, sync_row=True):
    from pathlib import Path
    directory = Path(directory)
    archive = directory / f"case-{seed}.json.gz"
    with gzip.open(archive, "rt", encoding="utf-8") as handle:
        record = json.load(handle)
    action(record)
    with gzip.open(archive, "wt", encoding="utf-8") as handle:
        json.dump(record, handle)
    if sync_row:
        rows = selection.read_json(directory / "rows.json")
        rows = [record["row"] if row["seed"] == seed else row for row in rows]
        write(directory / "rows.json", rows)


def finalized(setup, registry, costs=None):
    index = setup["root"] / "extended-index.json"
    paths = evaluations(setup, registry, costs=costs)
    write(index, paths)
    output = setup["root"] / "selection.json"
    return selection.finalize(setup["registry_path"], index, output), output, paths


def test_registry_reads_no_scenarios_and_relative_checkpoint_matches_source_cwd(setup):
    result = registered(setup)
    assert result["baseline"] == "rollout_frozen"
    assert result["directions"]["state"]["tie_order"] == ["state_a", "state_b"]
    assert result["entries"]["rl_a"]["identity"]["checkpoint_sha256"]["checkpoint"]
    assert not (setup["root"] / "evaluations").exists()
    assert selection.verify_registry(setup["registry_path"]) == result
    with pytest.raises(FileExistsError):
        registered(setup)


@pytest.mark.parametrize("problem", ["four", "fallback", "duplicate", "git", "future_freeze", "environment", "source", "checkpoint", "protocol"])
def test_register_rejects_unfrozen_or_ambiguous_input(setup, monkeypatch, problem):
    from pathlib import Path
    plan = setup["plan"]
    candidate = plan["directions"]["state"]["candidates"][0]
    if problem == "four":
        plan["directions"]["state"]["candidates"] *= 2
    elif problem == "fallback":
        plan["directions"]["state"]["fallback"] = "not-registered"
    elif problem == "duplicate":
        candidate["id"] = "rollout_frozen"
    elif problem == "git":
        monkeypatch.setattr(selection.subprocess, "check_output", lambda *a, **k: "b" * 40)
    elif problem == "future_freeze":
        value = selection.read_json(candidate["freeze"])
        value["frozen_at"] = "2100-01-01T00:00:00+00:00"
        write(Path(candidate["freeze"]), value)
    elif problem == "environment":
        environment = setup["root"] / "other-env.json"
        value = selection.read_json(candidate["environment"])
        value["packages"]["numpy"] = "other"
        write(environment, value)
        candidate["environment"] = str(environment)
    elif problem == "source":
        (Path(candidate["source_dir"]) / "src/simulation/engine.py").write_text("changed")
    elif problem == "checkpoint":
        (Path(candidate["source_dir"]) / "weights.bin").write_bytes(b"changed")
    else:
        write(Path(candidate["source_dir"]) / "research/v1_protocol.json", {})
    write(setup["plan_path"], plan)
    with pytest.raises(ValueError):
        registered(setup)
    assert not setup["registry_path"].exists()


def test_changed_physics_is_rejected_even_if_individually_refrozen(setup):
    from pathlib import Path
    candidate = setup["plan"]["directions"]["state"]["candidates"][0]
    source = Path(candidate["source_dir"])
    (source / "src/simulation/engine.py").write_text("different physics")
    freeze = selection.read_json(candidate["freeze"])
    freeze["identity"] = selection.live_identity(source, selection.read_json(candidate["spec"]), setup["protocol"])
    write(Path(candidate["freeze"]), freeze)
    with pytest.raises(ValueError, match="physical client"):
        registered(setup)


def test_refreezing_a_different_baseline_does_not_change_the_predeclared_reference(setup):
    from pathlib import Path
    candidate = setup["plan"]["baseline"]
    spec = selection.read_json(candidate["spec"])
    spec["kwargs"]["rollout_config"]["particles"] = 5
    write(Path(candidate["spec"]), spec)
    freeze = selection.read_json(candidate["freeze"])
    freeze["identity"] = selection.live_identity(Path(candidate["source_dir"]), spec, setup["protocol"])
    write(Path(candidate["freeze"]), freeze)
    with pytest.raises(ValueError, match="predeclared rollout"):
        registered(setup)


def test_selection_uses_unrounded_mean_and_tie_order_without_ci_gate(setup):
    registry = registered(setup)
    selected, output, _ = finalized(setup, registry, costs={"state_a": 900., "state_b": 899.99999999})
    assert selected["decisions"]["state"]["selected"] == "state_b"
    assert selected["decisions"]["rl"]["selected"] == "rl_a"
    assert all(d["passed_extended_screening"] for d in selected["decisions"].values())
    with pytest.raises(FileExistsError):
        selection.finalize(setup["registry_path"], setup["root"] / "extended-index.json", output)


def test_baseline_failed_clear_forces_fixed_fallback_but_not_a_pass(setup):
    registry = registered(setup)
    paths = evaluations(setup, registry, costs={"state_b": 800.})
    def failed_clear(record):
        record["row"]["failed_clear_count"] = 1
        record["evaluation"]["failed_clear_count"] = 1
    change_record(paths[registry["baseline"]], 1, failed_clear)
    index, output = setup["root"] / "index.json", setup["root"] / "selected.json"
    write(index, paths)
    selected = selection.finalize(setup["registry_path"], index, output)
    for direction, decision in selected["decisions"].items():
        assert decision["selected"] == registry["directions"][direction]["fallback"]
        assert decision["fallback_used"] and not decision["passed_extended_screening"]
        assert not decision["baseline_usable"]


def test_p95_is_ratio_of_percentiles_and_screen_excludes_failed_candidate(setup):
    registry = registered(setup)
    paths = evaluations(setup, registry, costs={"state_a": 1050.00001, "state_b": 1050.})
    def early_failure(record):
        record["row"].update(successful=False, all_cleared=False, cleared_total=0,
                             completion_certified=False, penalized_time_s=360000, errors=["synthetic failure"])
        record["evaluation"].update(all_cleared=False, cleared_total=0)
    change_record(paths["rl_b"], 1, early_failure)
    index = setup["root"] / "index.json"
    write(index, paths)
    selected = selection.finalize(setup["registry_path"], index, setup["root"] / "selected.json")
    assert selected["decisions"]["state"]["selected"] == "state_b"
    assert not selected["decisions"]["state"]["screens"]["state_a"]["passed"]
    assert selected["decisions"]["rl"]["screens"]["rl_b"]["mean_penalized_s"] == 180450.
    assert not selected["decisions"]["rl"]["screens"]["rl_b"]["passed"]


@pytest.mark.parametrize("corruption", ["rows", "truth", "spec", "phase", "history", "fees", "negative", "missing", "pair"])
def test_archive_audit_rejects_corrupted_or_unpaired_evidence(setup, corruption):
    from pathlib import Path
    registry = registered(setup)
    paths = evaluations(setup, registry)
    directory = paths["state_a"]
    if corruption == "missing":
        (Path(directory) / "case-1.json.gz").unlink()
    else:
        def change(record):
            if corruption == "rows":
                record["row"]["program_runtime_s"] = 1.
            elif corruption == "truth":
                record["evaluation"]["ground_truth"]["seed"] = 99
            elif corruption == "spec":
                record["spec"]["entrypoint"] = "another:strategy"
            elif corruption == "phase":
                record["evaluation_phase"] = "during_policy"
            elif corruption == "history":
                record["history"].pop()
            elif corruption == "fees":
                record["row"]["movement_s"] += 1
            elif corruption == "negative":
                record["row"]["program_runtime_s"] = -1.
            else:
                record["evaluation"]["ground_truth"]["sources"] = ["different synthetic case"]
                record["row"]["case_sha256"] = selection.digest(record["evaluation"]["ground_truth"])
        change_record(directory, 1, change, sync_row=corruption != "rows")
    index = setup["root"] / "index.json"
    write(index, paths)
    with pytest.raises(ValueError):
        selection.finalize(setup["registry_path"], index, setup["root"] / "selected.json")


def final_paths(setup, registry, selected):
    ids = [registry["baseline"]] + [d["selected"] for d in selected["decisions"].values()]
    index = {split: evaluations(setup, registry, split=split, ids=ids) for split in ("final_random", "final_stress")}
    path = setup["root"] / "final-index.json"
    write(path, index)
    return path, index


def test_final_chain_audit_preserves_selected_identities_and_raw_archives(setup):
    registry = registered(setup)
    selected, selection_path, _ = finalized(setup, registry)
    index_path, _ = final_paths(setup, registry, selected)
    output = setup["root"] / "audit.json"
    result = selection.audit_final(setup["registry_path"], selection_path, index_path, output)
    assert result["identity_and_archive_checks_passed"]
    assert result["all_final_methods_reliable"]
    assert len(result["partitions"]["final_stress"]["evaluations"]) == 4
    assert len(result["partitions"]["final_random"]["evaluations"]["state_a"]["evidence"]["archives_sha256"]) == 2
    with pytest.raises(FileExistsError):
        selection.audit_final(setup["registry_path"], selection_path, index_path, output)


@pytest.mark.parametrize("corruption", ["identity", "replacement", "selection_evidence", "decision", "rule", "source"])
def test_final_rejects_post_selection_changes(setup, corruption):
    from pathlib import Path
    registry = registered(setup)
    selected, selection_path, extended = finalized(setup, registry)
    index_path, index = final_paths(setup, registry, selected)
    if corruption == "identity":
        manifest_path = Path(index["final_stress"]["state_a"]) / "manifest.json"
        manifest = selection.read_json(manifest_path)
        manifest["identity"]["checkpoint_sha256"]["checkpoint"] = "changed"
        write(manifest_path, manifest)
    elif corruption == "replacement":
        index["final_stress"]["state_b"] = index["final_stress"].pop("state_a")
        write(index_path, index)
    elif corruption == "selection_evidence":
        change_record(extended["state_a"], 1, lambda r: r["row"].update(program_runtime_s=9.))
    elif corruption == "decision":
        selected["decisions"]["state"]["selected"] = "state_b"
        # Even self-consistent rehashing cannot bypass the deterministic decision.
        selected["payload_sha256"] = selection.digest({k: v for k, v in selected.items() if k != "payload_sha256"})
        write(selection_path, selected)
    elif corruption == "rule":
        Path(registry["rule_path"]).write_text("changed")
    else:
        (Path(registry["entries"]["state_a"]["source_dir"]) / "weights.bin").write_bytes(b"new checkpoint")
    with pytest.raises(ValueError):
        selection.audit_final(setup["registry_path"], selection_path, index_path, setup["root"] / "audit.json")


def test_final_keeps_reliable_identity_audit_separate_from_failed_baseline_acceptance(setup):
    registry = registered(setup)
    selected, selection_path, _ = finalized(setup, registry)
    index_path, index = final_paths(setup, registry, selected)
    def failed_clear(record):
        record["row"]["failed_clear_count"] = record["evaluation"]["failed_clear_count"] = 1
    change_record(index["final_stress"]["rollout_frozen"], 21, failed_clear)
    result = selection.audit_final(setup["registry_path"], selection_path, index_path, setup["root"] / "audit.json")
    assert result["identity_and_archive_checks_passed"]
    assert not result["all_final_methods_reliable"]
    assert not result["partitions"]["final_stress"]["baseline_usable"]
