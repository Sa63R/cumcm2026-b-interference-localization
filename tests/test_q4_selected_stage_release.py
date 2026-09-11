"""Exercise actual runner CLI preflight, stopping before a case can run."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

from experiments import run_q4_round2 as runner
from experiments.evaluate_q4_feedback_symmetry import CANDIDATES, SPECS, selected_specs


class ExecutionBoundaryReached(Exception):
    pass


@pytest.fixture
def cli(tmp_path, monkeypatch):
    # One real file is hashed and archived; only the scope of this synthetic
    # source map is substituted. The source/spec/seed comparisons run in main.
    source = tmp_path / "src" / "sentinel.py"
    source.parent.mkdir()
    source.write_text("# metadata-only frozen source\n", encoding="utf-8")
    actual_hashes = {"src/sentinel.py": hashlib.sha256(source.read_bytes()).hexdigest()}
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "hashes", lambda: {
        path: hashlib.sha256((tmp_path/path).read_bytes()).hexdigest() for path in actual_hashes})
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: "synthetic-test-commit\n")
    entered = []
    class StopBeforeCases:
        def __init__(self, **kwargs):
            entered.append(kwargs)
        def __enter__(self):
            raise ExecutionBoundaryReached()
        def __exit__(self, *args):
            return False
    monkeypatch.setattr(runner, "ProcessPoolExecutor", StopBeforeCases)
    monkeypatch.setattr(runner, "one", lambda *a, **k: pytest.fail("No case may be generated or executed"))
    def invoke(selection, specs, *, stage="confirmation", start=625101, count=128):
        selection_path, specs_path = tmp_path/"selection.json", tmp_path/"specs.json"
        selection_path.write_text(json.dumps(selection), encoding="utf-8")
        specs_path.write_text(json.dumps(specs), encoding="utf-8")
        output = tmp_path/"run"
        monkeypatch.setattr(sys, "argv", ["run_q4_round2.py", "--selection", str(selection_path),
            "--specs", str(specs_path), "--stage", stage, "--start", str(start),
            "--count", str(count), "--workers", "1", "--output", str(output)])
        runner.main()
    return invoke, actual_hashes, entered, tmp_path


def selection_for(source, chosen=CANDIDATES[0]):
    return dict(passed=True, selected=chosen, specs=copy.deepcopy(SPECS),
        selected_specs=copy.deepcopy(selected_specs(chosen)), source_sha256=source.copy(),
        reserved_seeds={"confirmation": list(range(625101, 625229)),
                        "stress": list(range(625301, 625385))})


@pytest.mark.parametrize("chosen", CANDIDATES)
def test_actual_cli_releases_four_arms_with_original_selection_hash(cli, chosen):
    invoke, source, entered, root = cli
    selection = selection_for(source, chosen)
    with pytest.raises(ExecutionBoundaryReached):
        invoke(selection, selection["selected_specs"])
    assert entered == [{"max_workers": 1}]
    manifest = json.loads((root/"run/manifest.json").read_bytes())
    assert manifest["specs"] == selected_specs(chosen) and len(manifest["specs"]) == 4
    assert manifest["selection_sha256"] == hashlib.sha256((root/"selection.json").read_bytes()).hexdigest()
    assert json.loads((root/"selection.json").read_bytes())["specs"] == SPECS
    assert (root/"run/records").is_dir() and not list((root/"run/records").iterdir())


def test_actual_legacy_full_five_arm_stress_release_is_unchanged(cli):
    invoke, source, entered, root = cli
    selection = selection_for(source)
    del selection["selected_specs"]
    selection.pop("passed")  # This was not a legacy runner prerequisite.
    with pytest.raises(ExecutionBoundaryReached):
        invoke(selection, SPECS, stage="stress", start=625301, count=84)
    manifest = json.loads((root/"run/manifest.json").read_bytes())
    assert len(entered) == 1 and manifest["specs"] == SPECS
    assert manifest["seeds"] == list(range(625301, 625385))


@pytest.mark.parametrize("mutation", ["unpassed", "truthy_passed", "missing_selected", "missing_baseline",
    "changed_kwargs", "extra_arm", "not_strict", "wrong_source", "wrong_seeds", "wrong_input", "null_subset"])
def test_actual_cli_refuses_invalid_release_before_executor_or_directory(cli, mutation):
    invoke, source, entered, root = cli
    selection = selection_for(source)
    if mutation == "unpassed":
        selection["passed"] = False
    elif mutation == "truthy_passed":
        selection["passed"] = 1
    elif mutation == "missing_selected":
        del selection["selected_specs"][selection["selected"]]
    elif mutation == "missing_baseline":
        del selection["selected_specs"][runner.BASE]
    elif mutation == "changed_kwargs":
        selection["selected_specs"][selection["selected"]]["kwargs"]["max_expansions"] = 201
    elif mutation == "extra_arm":
        selection["selected_specs"]["compact_unregistered"] = copy.deepcopy(runner.BASE_SPEC)
    elif mutation == "not_strict":
        selection["selected_specs"] = copy.deepcopy(SPECS)
    elif mutation == "wrong_source":
        selection["source_sha256"]["src/sentinel.py"] = "changed"
    elif mutation == "wrong_seeds":
        selection["reserved_seeds"]["confirmation"][0] = 625100
    elif mutation == "null_subset":
        selection["selected_specs"] = None
    specs = selected_specs(CANDIDATES[1]) if mutation == "wrong_input" else selected_specs(CANDIDATES[0])
    with pytest.raises(ValueError):
        invoke(selection, specs)
    assert not entered and not (root/"run").exists()


def test_legacy_release_returns_original_object():
    assert runner.release_specs({"specs": SPECS}) is SPECS
