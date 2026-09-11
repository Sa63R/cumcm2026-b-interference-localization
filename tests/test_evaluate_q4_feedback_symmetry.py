"""Synthetic JSON/row validation only: no scenario generator or simulator."""
import copy
import gzip
import json
from pathlib import Path
import zipfile

import pytest

from experiments import evaluate_q4_feedback_symmetry as e


def good():
    return dict(all_clear=True, mean_saved_s=10., saving_ci95_s=[1., 19.], mean_reduction_fraction=.005, p95_ratio=1.05)


@pytest.mark.parametrize("phase,field,value", [
    ("development", "mean_saved_s", 0.), ("development", "all_clear", False),
    ("development", "p95_ratio", 1.050001), ("confirmation", "saving_ci95_s", [0., 19.]),
    ("confirmation", "mean_reduction_fraction", .004999), ("confirmation", "all_clear", False)])
def test_fixed_gate_rejects(phase, field, value):
    random = good(); random[field] = value
    assert not e.candidate_passes(random, good(), phase)


def test_stress_nonworse_and_two_candidates():
    stress = good(); stress["mean_saved_s"] = 0.
    assert e.candidate_passes(good(), stress, "development")
    assert e.candidate_passes(good(), stress, "confirmation")
    stress["mean_saved_s"] = -1e-9
    assert not e.candidate_passes(good(), stress, "development")
    assert e.CANDIDATES == ("compact_feedback_mean", "compact_feedback_centers") and e.REFERENCE == "compact_joint_continuation"
    assert len(e.SPECS) == 5 and "clear_before_probe_audit.json" not in e.AUDITS
    with pytest.raises(ValueError): e.candidate_passes(good(), good(), "unknown")


def row(seed, label, stage="pilot"):
    return dict(seed=seed, strategy=label, stage=stage, case_id=f"metadata-only-{seed}", case_sha256=f"case{seed}",
        successful=True, all_cleared=True, virtual_time_s=100., penalized_time_s=100., common_lower_bound_s=50.,
        time_over_lower_bound=2., penalized_time_over_lower_bound=2., program_runtime_s=.01,
        movement_s=70., detection_s=20., switching_s=2., optical_s=3., removal_s=5.)


@pytest.mark.parametrize("count", [24, 14, 128, 84])
def test_exact_complete_development_matrix_sizes(count):
    seeds = list(range(count)); rows = [row(s, label) for s in seeds for label in e.SPECS]
    e.validate_matrix(rows, seeds, e.SPECS, "pilot")
    with pytest.raises(ValueError): e.validate_matrix(rows[:-1], seeds, e.SPECS, "pilot")
    with pytest.raises(ValueError): e.validate_matrix(rows[:-1]+[rows[0]], seeds, e.SPECS, "pilot")
    with pytest.raises(ValueError): e.validate_matrix(rows+[row(count, e.CANDIDATES[0])], seeds, e.SPECS, "pilot")


@pytest.mark.parametrize("key,value", [("case_sha256", "different"), ("common_lower_bound_s", 51.),
    ("stage", "stress"), ("penalized_time_s", 1.), ("virtual_time_s", float("nan")), ("time_over_lower_bound", 9.)])
def test_full_case_stage_and_penalty_match(key, value):
    rows = [row(1, label) for label in e.SPECS]; rows[-1][key] = value
    with pytest.raises(ValueError): e.validate_matrix(rows, [1], e.SPECS, "pilot")


def test_failed_row_is_retained_with_fixed_penalty():
    rows = [row(1, label) for label in e.SPECS]; rows[-1].update(successful=False, all_cleared=False, penalized_time_s=360000., penalized_time_over_lower_bound=7200.)
    e.validate_matrix(rows, [1], e.SPECS, "pilot")
    rows[-1]["penalized_time_s"] = rows[-1]["virtual_time_s"]
    with pytest.raises(ValueError): e.validate_matrix(rows, [1], e.SPECS, "pilot")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def gz(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream: json.dump(value, stream)


@pytest.fixture
def metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(e, "ROOT", tmp_path)
    monkeypatch.setattr(e, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(e, "RESEARCH", tmp_path / "research")
    monkeypatch.setattr(e, "RL_ROOT", tmp_path / "rl")
    monkeypatch.setattr(e, "DEVELOPMENT", {"development": [1]})
    directory = e.RESULTS / "development"; directory.mkdir(parents=True)
    specs = e.SPECS
    manifest = dict(source_sha256={}, specs=specs, seeds=[1], stage="pilot", failure_penalty_s=360000, selection_sha256=None)
    save(directory / "manifest.json", manifest); save(directory / "freeze.json", {"manifest_sha256": e.digest(manifest)})
    with zipfile.ZipFile(directory / "source.zip", "w"): pass
    rows = sorted([row(1, label) for label in specs], key=lambda r: (r["seed"], r["strategy"]))
    report = e.report_rows(rows); save(directory / "summary.json", report)
    for r in rows: gz(directory / "records" / f"{r['strategy']}-1.json.gz", {"row": r, "spec": specs[r["strategy"]]})
    generic = dict(all_passed=True, records=5, passed_records=5, errors=[],
        audits=[dict(passed=True, **{k:r[k] for k in ('strategy','case_id','successful','virtual_time_s','common_lower_bound_s','time_over_lower_bound')}) for r in rows])
    save(directory / "independent_audit.json", generic)
    for name, label in e.AUDITS.items():
        if label is None: continue
        paths = [f"records/{label}-1.json.gz", "manifest.json", "freeze.json", "independent_audit.json"]
        save(directory / name, dict(all_passed=True, records=1, passed_records=1, errors=[],
            audits=[dict(strategy=label, seed=1, passed=True)], input_sha256={p:e.sha(directory / p) for p in paths}))
    rl = e.RESULTS / "development-rl"; rl.mkdir()
    rl_spec = {"entrypoint":"fake-metadata-only", "kwargs":{}}
    identities = {"research/q4_joint_visibility/rl_reference/reference.json":"reference",
        "experiments/q4_frozen_rl_worker.py":"worker", "experiments/run_q4_frozen_rl_compare.py":"runner"}
    save(e.RL_ROOT / "research/q4_joint_visibility/rl_reference/reference.json", {"spec":rl_spec})
    rf = dict(reference_sha256="reference", worker_sha256="worker", runner_sha256="runner", reference_arm=False,
        state_manifest_sha256=e.sha(directory / "manifest.json"), state_summary_sha256=e.sha(directory / "summary.json"),
        cases=[{k:rows[0][k] for k in ("case_sha256","common_lower_bound_s","stage")}], spec=rl_spec)
    save(rl / "freeze.json", rf)
    rl_row = row(1, e.RL_LABEL)
    combined=e.report_rows(sorted(rows+[rl_row], key=lambda r: (r["seed"], r["strategy"])))
    combined.update(rl_audits_all_passed=True, freeze_sha256=e.sha(rl / "freeze.json")); save(rl / "comparison.json", combined)
    gz(rl / "worker-0/record.json.gz", dict(row=rl_row,spec=rl_spec,audit={"passed":True},policy_reference_sha256="reference"))
    return directory, rl, dict(source_sha256={}, rl_identity=identities)


def test_complete_reader_accepts_synthetic_five_plus_rl(metadata):
    directory, rl, frozen = metadata
    evidence = {}; report, combined, _ = e.read_set("development", evidence, frozen)
    assert len(report["rows"]) == 5 and len(combined["rows"]) == 6
    assert "results/development-rl/worker-0/record.json.gz" in evidence


@pytest.mark.parametrize("mutation", ["audit_count", "audit_seed", "record_bytes", "missing_rl", "rl_failed", "rl_case", "rl_source", "state_spec"])
def test_reader_rejects_stale_or_incomplete_evidence(metadata, mutation):
    directory, rl, frozen = metadata
    if mutation in {"audit_count", "audit_seed"}:
        p=directory / "feedback_symmetry_mean_audit.json"; a=e.read(p)
        if mutation=="audit_count": a["records"]=128
        else: a["audits"][0]["seed"]=2
        save(p,a)
    elif mutation=="record_bytes":
        p=directory / "records" / f"{e.CANDIDATES[0]}-1.json.gz"
        with p.open("ab") as stream: stream.write(b"\0")
    elif mutation=="missing_rl": (rl / "worker-0/record.json.gz").unlink()
    elif mutation=="rl_failed":
        p=rl / "comparison.json"; a=e.read(p);a["rl_audits_all_passed"]=False;save(p,a)
    elif mutation=="rl_case":
        p=rl / "freeze.json";a=e.read(p);a["cases"][0]["case_sha256"]="wrong";save(p,a)
        p=rl / "comparison.json";a=e.read(p);a["freeze_sha256"]=e.sha(rl / "freeze.json");save(p,a)
    elif mutation=="rl_source": frozen["rl_identity"]["experiments/q4_frozen_rl_worker.py"]="changed"
    else:
        p=directory / "manifest.json";a=e.read(p);a["specs"][e.CANDIDATES[0]]["kwargs"]["max_expansions"]=201;save(p,a)
    with pytest.raises((ValueError, OSError, EOFError)): e.read_set("development", {}, frozen)


@pytest.mark.parametrize("key", ["source_sha256", "rl_identity", "evaluator_sha256", "evaluator_dependencies_sha256", "protocol_sha256", "specs_sha256"])
def test_identity_changes_refused_before_reading_cases(tmp_path, monkeypatch, key):
    monkeypatch.setattr(e,"RESEARCH",tmp_path)
    identity={k:{"x":"old"} if k.endswith("identity") or k.endswith("dependencies_sha256") or k=="source_sha256" else "old"
              for k in ("source_sha256","rl_identity","evaluator_sha256","evaluator_dependencies_sha256","protocol_sha256","specs_sha256")}
    identity["specs"]=e.SPECS
    save(tmp_path / "development-freeze.json",identity)
    changed=copy.deepcopy(identity);changed[key]="changed"
    monkeypatch.setattr(e,"development_identity",lambda:changed)
    monkeypatch.setattr(e,"read_set",lambda *a:pytest.fail("Must reject before reading any dataset"))
    with pytest.raises(ValueError,match="Frozen"):e.main("development")


def test_existing_decision_cannot_be_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(e,"RESEARCH",tmp_path);save(tmp_path / "development-decision.json",{"passed":False})
    monkeypatch.setattr(e,"development_identity",lambda:pytest.fail("Preserve decision before any work"))
    with pytest.raises(ValueError,match="Preserve"):e.main("development")


def test_selection_prefers_simpler_only_inside_fixed_five_second_band():
    cap, bounded = e.CANDIDATES
    assert e.choose_candidate([], {}) is None
    assert e.choose_candidate([bounded], {bounded: 90.}) == bounded
    assert e.choose_candidate([cap, bounded], {cap: 100., bounded: 95.001}) == cap
    assert e.choose_candidate([cap, bounded], {cap: 100., bounded: 95.}) == bounded
    assert e.choose_candidate([cap, bounded], {cap: 95., bounded: 100.}) == cap
    with pytest.raises(ValueError): e.choose_candidate([cap, cap], {cap: 1.})
    with pytest.raises(ValueError): e.choose_candidate(['unregistered'], {'unregistered': 1.})


@pytest.mark.parametrize('selected', e.CANDIDATES)
def test_independent_matrix_contains_only_selected_candidate(tmp_path, monkeypatch, selected):
    monkeypatch.setattr(e, 'RESEARCH', tmp_path)
    chosen = e.selected_specs(selected)
    assert len(chosen) == 4 and set(chosen) & set(e.CANDIDATES) == {selected}
    save(tmp_path / 'selection.json', dict(selected=selected, selected_specs=chosen))
    assert e.stage_specs('confirmation') == chosen
    assert e.stage_specs('stress') == chosen
    assert e.stage_specs('development') == e.SPECS
    save(tmp_path / 'selection.json', dict(selected=selected, selected_specs=e.SPECS))
    with pytest.raises(ValueError, match='specifications'): e.stage_specs('confirmation')
    with pytest.raises(ValueError): e.selected_specs('unregistered')


@pytest.mark.parametrize('selected', e.CANDIDATES)
def test_complete_independent_four_state_plus_rl_reader(metadata, monkeypatch, selected):
    old_directory, old_rl, frozen = metadata
    directory, rl = e.RESULTS / 'confirmation', e.RESULTS / 'confirmation-rl'
    old_directory.rename(directory); old_rl.rename(rl)
    monkeypatch.setattr(e, 'INDEPENDENT', {'confirmation': [1]})
    specs = e.selected_specs(selected)
    save(e.RESEARCH / 'selection.json', dict(selected=selected, selected_specs=specs))
    manifest = e.read(directory / 'manifest.json')
    manifest.update(specs=specs, stage='confirmation', selection_sha256=e.sha(e.RESEARCH / 'selection.json'))
    save(directory / 'manifest.json', manifest)
    save(directory / 'freeze.json', {'manifest_sha256': e.digest(manifest)})
    for path in (directory / 'records').glob('*.json.gz'):
        with gzip.open(path, 'rt', encoding='utf8') as stream: record = json.load(stream)
        if record['row']['strategy'] not in specs: path.unlink()
    rows = sorted([row(1, label, 'confirmation') for label in specs], key=lambda r: r['strategy'])
    for r in rows: gz(directory / 'records' / f"{r['strategy']}-1.json.gz", {'row':r, 'spec':specs[r['strategy']]})
    report = e.report_rows(rows); save(directory / 'summary.json', report)
    save(directory / 'independent_audit.json', dict(all_passed=True, records=4, passed_records=4, errors=[],
        audits=[dict(passed=True, **{k:r[k] for k in ('strategy','case_id','successful','virtual_time_s','common_lower_bound_s','time_over_lower_bound')}) for r in rows]))
    for name, label in e.AUDITS.items():
        if label is None: continue
        if label not in specs:
            (directory / name).unlink()
            continue
        paths = [f'records/{label}-1.json.gz', 'manifest.json', 'freeze.json', 'independent_audit.json']
        save(directory / name, dict(all_passed=True, records=1, passed_records=1, errors=[],
            audits=[dict(strategy=label, seed=1, passed=True)], input_sha256={p:e.sha(directory / p) for p in paths}))
    rf = e.read(rl / 'freeze.json')
    rf.update(state_manifest_sha256=e.sha(directory / 'manifest.json'), state_summary_sha256=e.sha(directory / 'summary.json'),
              cases=[{k:rows[0][k] for k in ('case_sha256','common_lower_bound_s','stage')}])
    save(rl / 'freeze.json', rf)
    rr = row(1, e.RL_LABEL, 'confirmation')
    combined = e.report_rows(sorted(rows + [rr], key=lambda r: r['strategy']))
    combined.update(rl_audits_all_passed=True, freeze_sha256=e.sha(rl / 'freeze.json'))
    save(rl / 'comparison.json', combined)
    gz(rl / 'worker-0/record.json.gz', dict(row=rr, spec=rf['spec'], audit={'passed':True}, policy_reference_sha256='reference'))
    state, all_methods, _ = e.read_set('confirmation', {}, frozen)
    assert len(state['rows']) == 4 and len(all_methods['rows']) == 5


def test_independent_cannot_switch_to_other_development_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(e, 'RESEARCH', tmp_path)
    identity = dict(source_sha256={}, rl_identity={}, evaluator_sha256='e', evaluator_dependencies_sha256={},
                    protocol_sha256='p', specs_sha256='s', specs=e.SPECS)
    save(tmp_path / 'development-freeze.json', identity)
    cap, bounded = e.CANDIDATES
    save(tmp_path / 'development-decision.json', dict(passed=True, selected=cap))
    chosen = e.selected_specs(bounded)
    save(tmp_path / 'selection-specs.json', chosen)
    save(tmp_path / 'selection.json', dict(passed=True, selected=bounded, selected_specs=chosen, **identity,
        reserved_seeds=e.INDEPENDENT, development_decision_sha256=e.sha(tmp_path / 'development-decision.json'),
        development_freeze_sha256=e.sha(tmp_path / 'development-freeze.json')))
    monkeypatch.setattr(e, 'development_identity', lambda: identity)
    monkeypatch.setattr(e, 'read_set', lambda *args: pytest.fail('Reject candidate switch before reading independent data'))
    with pytest.raises(ValueError, match='selection'): e.main('confirmation')


@pytest.mark.parametrize('key,changed', [('successful',False), ('virtual_time_s',120.),
    ('common_lower_bound_s',51.), ('time_over_lower_bound',2.4)])
def test_same_case_generic_audit_cannot_reuse_stale_numbers(metadata, key, changed):
    directory, _, frozen = metadata
    path = directory / 'independent_audit.json'
    audit = e.read(path); audit['audits'][0][key] = changed; save(path, audit)
    with pytest.raises(ValueError, match='stale'): e.read_set('development', {}, frozen)
