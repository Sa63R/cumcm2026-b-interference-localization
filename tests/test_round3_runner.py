"""File-only runner contract tests. No scenario generation or model execution."""
from copy import deepcopy
import gzip
import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "experiments/round3_runner.py"
spec = importlib.util.spec_from_file_location("round3_runner_tested", MODULE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def frozen_policy(tmp_path, monkeypatch):
    repo, output = tmp_path / "repo", tmp_path / "batch"
    (repo / "src").mkdir(parents=True)
    (repo / "src/policy.py").write_text("# fixture\n")
    output.mkdir()
    for helper in runner.HELPERS:
        path = repo / helper
        path.parent.mkdir(exist_ok=True)
        path.write_text("# unchanged fixture helper\n")
    checkpoint = tmp_path / "input.pt"
    checkpoint.write_bytes(b"inference fixture, not a serialized model")
    recipe = {"repository": str(repo), "commit": "fixture-commit", "spec": "spec.json",
              "weights": {"checkpoint": {"path": str(checkpoint), "sha256": runner.sha(checkpoint)}}}
    runner.write(repo / "spec.json", {"entrypoint": "policy:run", "kwargs": {"checkpoint": "input.pt"}})
    monkeypatch.setattr(runner.subprocess, "check_output",
                        lambda args, **kwargs: "fixture-commit\n" if args[1] == "rev-parse" else "")
    policy = runner.inspect_recipe("rl", recipe)
    archived = output / "weights/rl.pt"
    archived.parent.mkdir()
    archived.write_bytes(checkpoint.read_bytes())
    policy["weights"]["checkpoint"]["archive_path"] = "weights/rl.pt"
    policy["spec"] = deepcopy(policy["original_spec"])
    policy["spec"]["kwargs"]["checkpoint"] = str(archived.resolve())
    policy["spec_sha256"] = runner.digest(policy["spec"])
    (output / "rl-source.zip").write_bytes(b"archive fixture")
    policy["source_archive_sha256"] = runner.sha(output / "rl-source.zip")
    return repo, output, policy, recipe


def test_intact_policy_and_cpu_contract(frozen_policy):
    _, output, policy, _ = frozen_policy
    runner.verify_policy(output, "rl", policy)
    assert runner.THREAD_ENV["CUDA_VISIBLE_DEVICES"] == "-1"
    assert runner.THREAD_ENV["OMP_NUM_THREADS"] == "1"
    assert runner.sha(runner.LEGACY_PATH) == runner.LEGACY_SHA


@pytest.mark.parametrize("target", ["input_model", "archived_model", "source", "helper", "spec", "archive"])
def test_frozen_content_mutations_rejected(frozen_policy, target):
    repo, output, policy, _ = frozen_policy
    targets = {"input_model": Path(policy["weights"]["checkpoint"]["input_path"]),
               "archived_model": output / "weights/rl.pt", "source": repo / "src/policy.py",
               "helper": repo / runner.HELPERS[0], "spec": repo / "spec.json",
               "archive": output / "rl-source.zip"}
    targets[target].write_bytes(b"modified content")
    with pytest.raises(ValueError):
        runner.verify_policy(output, "rl", policy)


def test_declared_checkpoint_hash_required(frozen_policy):
    _, _, _, recipe = frozen_policy
    recipe["weights"]["checkpoint"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Checkpoint hash"):
        runner.inspect_recipe("rl", recipe)


def test_effective_path_does_not_change_stage_identity(frozen_policy):
    _, _, policy, _ = frozen_policy
    other = deepcopy(policy)
    other["spec"]["kwargs"]["checkpoint"] = "other-stage/archive.pt"
    assert runner.policy_identity(policy) == runner.policy_identity(other)
    other["weights"]["checkpoint"]["sha256"] = "different tensors"
    assert runner.policy_identity(policy) != runner.policy_identity(other)


def test_registered_stage_and_old_smoke_only():
    recipes = {"candidate": {"spec": "fixture"}}
    protocol = {"created_experiments": {"x": {"recipes_sha256": runner.digest(recipes),
                                              "pilot_seeds": [926091200101, 926091200132]}}}
    assert runner.stage_definition({}, "anything", "smoke", {}) == ([200114], None)
    assert len(runner.stage_definition(protocol, "x", "pilot", recipes)[0]) == 32
    with pytest.raises(ValueError, match="not registered"):
        runner.stage_definition(protocol, "x", "pilot", {"different": 1})
    with pytest.raises(KeyError):
        runner.stage_definition(protocol, "undeclared", "pilot", recipes)
    protocol["created_experiments"]["x"]["pilot_seeds"] = [200113, 200115]
    with pytest.raises(ValueError, match="seed interval"):
        runner.stage_definition(protocol, "x", "pilot", recipes)


def sample_row(label="baseline", seed=200114, success=True, cost=100):
    return {"strategy": label, "seed": seed, "case_sha256": "one-physical-world", "successful": success,
            "failed_clear_count": 0, "virtual_time_s": cost,
            "penalized_time_s": cost if success else 360000,
            "program_runtime_s": .1, "program_cpu_s": .05, "measurement_count": 2, "movement_s": cost-12}


def test_pair_identity_and_failures_are_not_dropped():
    baseline = [sample_row(seed=1), sample_row(seed=2)]
    candidate = [sample_row("rl", 1, False), sample_row("rl", 2, True, 90)]
    result = runner.compare_rows(baseline, candidate)
    assert result["pairs"] == 2 and result["safe"] is False
    assert result["mean_saved_s"] == (100-360000+10)/2
    assert (result["wins"], result["losses"]) == (1, 1)
    candidate[0]["case_sha256"] = "different-world"
    with pytest.raises(ValueError, match="case hash"):
        runner.compare_rows(baseline, candidate)


@pytest.mark.parametrize("seeds", [[], [1], [1, 1], [1, 3]])
def test_missing_extra_or_duplicate_pairs_rejected(seeds):
    with pytest.raises(ValueError):
        runner.compare_rows([sample_row(seed=1), sample_row(seed=2)], [sample_row("rl", s) for s in seeds])


def make_record(manifest, label, seed, success=True):
    return {"row": sample_row(label, seed, success), "spec": manifest["policies"][label]["spec"],
            "frozen_manifest_sha256": runner.digest(manifest),
            "round3_integrity": {"before_and_after_verified": True}}


def test_integrity_and_failure_penalty_required():
    manifest = {"policies": {"rl": {"spec": {}}}, "limits": runner.LIMITS}
    record = make_record(manifest, "rl", 200114, False)
    runner.verify_record(record, manifest, "rl", 200114)
    record["row"]["penalized_time_s"] = 100
    with pytest.raises(ValueError, match="Failure penalty"):
        runner.verify_record(record, manifest, "rl", 200114)
    record = make_record(manifest, "rl", 200114)
    record.pop("round3_integrity")
    with pytest.raises(ValueError, match="integrity"):
        runner.verify_record(record, manifest, "rl", 200114)


def test_complete_summary_retains_failure_and_disables_smoke_gate(tmp_path, monkeypatch):
    manifest = {"policies": {"baseline": {"spec": {}}, "rl": {"spec": {}}},
                "limits": runner.LIMITS, "seeds": [200114], "stage": "smoke"}
    runner.write(tmp_path / "manifest.json", manifest)
    (tmp_path / "records").mkdir()
    for label in manifest["policies"]:
        with gzip.open(tmp_path / "records" / f"{label}-200114.json.gz", "wt", encoding="utf-8") as stream:
            json.dump(make_record(manifest, label, 200114, label == "baseline"), stream)
    monkeypatch.setattr(runner, "verify_frozen", lambda output: manifest)
    result = runner.summarize(tmp_path)
    assert result["averages"]["rl"]["runs"] == 1
    assert result["averages"]["rl"]["mean_s"] == 360000
    assert result["averages"]["rl"]["successful"] == 0
    assert result["comparisons_vs_rl"]["baseline"]["pairs"] == 1
    assert not result["comparisons"]["rl"]["pilot_expansion_gate"]
    assert result["source_and_weights_verified_after_batch"]
    (tmp_path / "records/rl-200114.json.gz").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        runner.summarize(tmp_path)


@pytest.fixture
def prior_audit(tmp_path):
    manifest = {"policies": {"baseline": {}, "rl": {}}, "seeds": [200114]}
    (tmp_path / "records").mkdir()
    records = {}
    rows = []
    for label in manifest["policies"]:
        relative = f"records/{label}-200114.json.gz"
        path = tmp_path / relative
        path.write_bytes(b"completed archive fixture")
        records[relative] = runner.sha(path)
        rows.append({"input_path": str(path), "input_sha256": runner.sha(path), "audit_passed": True})
    summary = {"records_sha256": records}
    runner.write(tmp_path / "manifest.json", manifest)
    runner.write(tmp_path / "summary.json", summary)
    audit = {"all_audits_passed": True, "all_runner_archives_available_and_verified": True,
             "batches": [{"path": str(tmp_path), "manifest_sha256": runner.sha(tmp_path / "manifest.json"),
                          "summary_sha256": runner.sha(tmp_path / "summary.json")}], "rows": rows, "records": len(rows)}
    runner.write(tmp_path / "audit.json", audit)
    return tmp_path, manifest, summary, audit


def test_complete_independent_audit_accepted(prior_audit):
    path, manifest, summary, _ = prior_audit
    assert runner.verify_prior_audit(path, manifest, summary)["sha256"] == runner.sha(path / "audit.json")


@pytest.mark.parametrize("mutation", ["not_passed", "stale_manifest", "stale_summary", "record_change", "missing_row", "duplicate_row", "failed_row", "missing_file"])
def test_stale_incomplete_or_failed_audit_rejected(prior_audit, mutation):
    path, manifest, summary, audit = prior_audit
    if mutation == "not_passed":
        audit["all_audits_passed"] = False
    elif mutation in ("stale_manifest", "stale_summary"):
        audit["batches"][0][mutation.removeprefix("stale_")+"_sha256"] = "wrong"
    elif mutation == "record_change":
        Path(audit["rows"][0]["input_path"]).write_bytes(b"mutated")
    elif mutation == "missing_row":
        audit["rows"].pop()
    elif mutation == "duplicate_row":
        audit["rows"][1] = audit["rows"][0]
    elif mutation == "failed_row":
        audit["rows"][0]["audit_passed"] = False
    else:
        Path(audit["rows"][0]["input_path"]).unlink()
    runner.write(path / "audit.json", audit)
    with pytest.raises(ValueError):
        runner.verify_prior_audit(path, manifest, summary)
