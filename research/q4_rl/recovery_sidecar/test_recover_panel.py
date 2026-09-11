"""Small offline fixtures only: no simulator, training, network or GPU."""
from concurrent.futures import Future
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

spec = importlib.util.spec_from_file_location("recovery", Path(__file__).with_name("recover_panel.py"))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class ImmediatePool:
    def __init__(self, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def submit(self, function, *args, **kwargs):
        future = Future()
        future.set_result(function(*args, **kwargs))
        return future


class RecoveryTests(unittest.TestCase):
    CAP = 1000000000
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.old, self.new, self.control = [self.root / p for p in ("old", "new", "control")]
        for p in (self.old, self.control, self.old / "records"):
            p.mkdir()
        self.requests = [dict(seed=8101000, split="development", family="random", source_mode=m)
            for m in ("mixed", "all_directional")]
        self.specs = {label: {"entrypoint": "fake:run"} for label in ("r8", "macro_v2_ppo512", "r9_probe")}
        r.put(self.control / "specs.json", self.specs)
        (self.root / "source.py").write_text("# frozen fixture\n")
        (self.root / "protocol.json").write_text("{}\n")
        self.sources = {"source.py": r.sha(self.root / "source.py")}
        with zipfile.ZipFile(self.old / "source.zip", "w") as archive:
            archive.write(self.root / "source.py", "source.py")
        self.e = types.SimpleNamespace(ROOT=self.root, PROTOCOL_PATH=self.root / "protocol.json",
            source_hashes=lambda: self.sources, artifact_hashes=lambda specs: {}, validate_specs=lambda specs: specs,
            case_requests=lambda *args, **kwargs: self.requests, build_case=self.case, _set_cpu_environment=lambda: None)
        self.manifest = dict(source_sha256=self.sources, artifact_sha256={}, specs=self.specs,
            raw_specs_sha256=r.sha(self.control / "specs.json"), protocol_sha256=r.sha(self.e.PROTOCOL_PATH),
            requests=self.requests, stage="development", reference="r9_probe", bootstrap_samples=5000,
            bootstrap_seed=4260911, workers=10, numerical_threads_per_worker=1, selection_sha256=None,
            formal_simulator=False, practice_simulator=False)
        r.put(self.old / "manifest.json", self.manifest)
        r.put(self.old / "freeze.json", {"manifest_sha256": r.canonical(self.manifest)})
        (self.old / "specs.original.json").write_bytes((self.control / "specs.json").read_bytes())
        r.put(self.control / "plan.json", dict(panels=[dict(run="fixed", start=8101000, count=1,
            families=["random"], scenarios=2)], arms=list(self.specs), models=[], prior_rl_reference="macro_v2_ppo512",
            bootstrap_samples=5000, control_sha256={"specs.json": r.sha(self.control / "specs.json")}))
        self.status = self.root / "supervisor.json"
        r.put(self.status, dict(run="fixed", final_sync_ok=True, termination_requested=True,
            child_returncode=-15, stop_reason="disk_free_below_minimum", elapsed_s=5.))
        self.matrix = r.expected_matrix(self.manifest, self.e)
        self.names = list(self.matrix)

    @staticmethod
    def case(**request):
        return types.SimpleNamespace(case_id="fixture-" + request["source_mode"], evaluation_config=lambda: request)

    def record(self, name, successful=True):
        item = self.matrix[name]
        request = self.requests[item["request_index"]]
        evaluation = dict(case_id=item["case_id"], ground_truth=request, virtual_time_s=1000., source_total=10,
            cleared_total=10 if successful else 9, measurement_count=2, failed_clear_count=0, action_count=12,
            all_cleared=successful, time_breakdown_s={"movement_s": 900., "detection_s": 100.})
        row = dict(**{k:v for k,v in evaluation.items() if k not in ("ground_truth", "time_breakdown_s")},
            **evaluation["time_breakdown_s"], case_sha256=item["case_sha256"], strategy=item["label"], problem=4,
            seed=request["seed"], stage=request["split"], family=request["family"], source_mode=request["source_mode"],
            successful=successful, completion_certified=successful, accepted_exit=True, audit_passed=successful,
            errors=[] if successful else ["fixture incomplete"], common_lower_bound_s=100., common_lower_bound_rounded_s=100.,
            penalized_time_s=1000. if successful else 360000., time_over_lower_bound=10. if successful else None,
            penalized_time_over_lower_bound=10. if successful else 3600., process_cpu_s=.1)
        return dict(row=row, evaluation=evaluation, request=request, spec=self.specs[item["label"]],
            history=[], evaluation_phase="after_policy_termination", audit={"passed": successful},
            common_lower_bound={"common_lower_bound_s": 100., "common_lower_bound_rounded_s":100.})

    def save(self, name, record):
        (self.old / "records" / name).write_bytes(gzip.compress(json.dumps(record).encode()))

    def prepare(self):
        return r.prepare(self.root, self.old, self.new, self.control, "fixed", self.status, self.e,
            r.sha(self.old / "manifest.json"), r.sha(self.control / "plan.json"), r.sha(self.status), self.CAP)

    def test_failure_retained_and_exact_missing_matrix(self):
        self.save(self.names[0], self.record(self.names[0], False))
        saved = self.prepare()
        self.assertEqual(saved["retained_failed_count"], 1)
        self.assertEqual([x["record_name"] for x in saved["missing_matrix"]], self.names[1:])
        self.assertEqual(r.sha(self.old / "records" / self.names[0]), r.sha(self.new / "records" / self.names[0]))
        self.assertTrue((self.old / "records" / self.names[0]).samefile(self.new / "records" / self.names[0]))

    def test_truncated_gzip_preserved_and_missing(self):
        payload = gzip.compress(json.dumps(self.record(self.names[0])).encode())[:-6]
        (self.old / "records" / self.names[0]).write_bytes(payload)
        saved = self.prepare()
        self.assertEqual(len(saved["administrative_records"]), 1)
        self.assertEqual(len(saved["missing_matrix"]), 6)
        self.assertEqual((self.new / "administrative_records" / self.names[0]).read_bytes(), payload)

    def test_decodable_wrong_identity_is_not_retryable(self):
        record = self.record(self.names[0]); record["row"]["strategy"] = "other"
        self.save(self.names[0], record)
        with self.assertRaisesRegex(ValueError, "row identity"):
            self.prepare()
        self.assertFalse(self.new.exists())

    def test_wrong_spec_rejected(self):
        record = self.record(self.names[0]); record["spec"] = {"entrypoint": "changed:run"}
        self.save(self.names[0], record)
        with self.assertRaisesRegex(ValueError, "request/spec"):
            self.prepare()

    def test_changed_source_rejected(self):
        self.e.source_hashes = lambda: {"source.py": "0" * 64}
        with self.assertRaisesRegex(ValueError, "Current source"):
            self.prepare()

    def test_changed_model_rejected(self):
        self.e.artifact_hashes = lambda specs: {"model.pt": "0" * 64}
        with self.assertRaisesRegex(ValueError, "model/config"):
            self.prepare()

    def test_bad_freeze_rejected(self):
        (self.old / "freeze.json").write_text('{"manifest_sha256":"wrong"}')
        with self.assertRaisesRegex(ValueError, "Manifest/freeze"):
            self.prepare()

    def test_duplicate_matrix_rejected(self):
        altered = copy.deepcopy(self.manifest); altered["requests"].append(self.requests[0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            r.expected_matrix(altered, self.e)

    def test_failed_sync_rejected(self):
        status = r.read(self.status); status["final_sync_ok"] = False
        self.status.write_text(json.dumps(status))
        with self.assertRaisesRegex(ValueError, "disk-stop/sync"):
            self.prepare()

    def test_execute_only_missing_and_preserves_failed(self):
        self.save(self.names[0], self.record(self.names[0], False))
        self.prepare()
        called, reports = [], []
        def run_one(request, label, spec, **kwargs):
            name = self.case(**request).case_id + "--" + label + ".json.gz"
            called.append(name)
            return self.record(name)
        def report_rows(rows, **kwargs):
            reports.append(kwargs)
            self.assertEqual(len(rows), 6)
            return {"rows": rows, "reference": kwargs["reference"]}
        self.e.run_one, self.e.report_rows = run_one, report_rows
        supervisor = types.SimpleNamespace(require_outer_supervisor=lambda root: None)
        with patch.object(r, "ProcessPoolExecutor", ImmediatePool), patch.dict(sys.modules, {"scripts.q4_training_pair": supervisor}):
            code = r.execute(self.root, self.new, r.sha(self.new / "recovery.json"), self.e, self.CAP)
        self.assertEqual(code, 1)  # The old complete failure is never rerun away.
        self.assertEqual(called, self.names[1:])
        self.assertEqual([x["reference"] for x in reports], ["r9_probe", "r8", "macro_v2_ppo512"])
        self.assertTrue(all(x["samples"] == 5000 and x["bootstrap_seed"] == 4260911 for x in reports))
        self.assertEqual(r.read(self.new / "panel_complete.json")["complete_rows"], 6)

    def test_retained_bytes_change_blocks_execution(self):
        self.save(self.names[0], self.record(self.names[0]))
        self.prepare()
        # Simulate a mutation via the old hard link; execute must detect it.
        self.save(self.names[0], self.record(self.names[0], False))
        with self.assertRaisesRegex(ValueError, "Retained record"):
            r.execute(self.root, self.new, r.sha(self.new / "recovery.json"), self.e, self.CAP)

    def test_missing_record_never_produces_partial_summary(self):
        self.prepare()
        self.assertFalse((self.new / "summary.json").exists())
        self.assertEqual(len(r.read(self.new / "recovery.json")["missing_matrix"]), 6)

    def test_cap_frozen_and_mismatch_rejected(self):
        saved = self.prepare()
        self.assertEqual(saved["max_new_artifact_bytes"], self.CAP)
        with self.assertRaisesRegex(ValueError, "cap differs"):
            r.execute(self.root, self.new, r.sha(self.new / "recovery.json"), self.e, self.CAP + 1)

    def test_writer_exact_boundary_and_stop_record_inside_cap(self):
        self.new.mkdir()
        budget = r.ArtifactBudget(self.new, 8192, outer_reserve=0)
        with budget.writer(self.new / "partial.bin") as stream:
            stream.write(b"a" * 4096)
            with self.assertRaises(r.ArtifactLimit):
                stream.write(b"b")
        self.assertEqual((self.new / "partial.bin").stat().st_size, 4096)
        budget.stop()
        self.assertLessEqual(budget.scan(), 8192)
        self.assertTrue(r.read(self.new / "administrative_stop.json")["administrative_interruption"])

    def test_old_hardlinks_excluded_new_partial_counted(self):
        self.new.mkdir()
        old = self.root / "old-record.bin"; old.write_bytes(b"x" * 12000)
        linked = self.new / "retained.bin"; r.link(old, linked)
        budget = r.ArtifactBudget(self.new, 8192, [linked], outer_reserve=0)
        self.assertEqual(budget.used, 0)
        with budget.writer(self.new / "inflight.bin") as stream:
            stream.write(b"x" * 4000)
        self.assertEqual(budget.scan(), 4000)
        with self.assertRaises(r.ArtifactLimit):
            budget.put(self.new / "too-big.json", {"data": "x" * 200})
        self.assertEqual(budget.scan(), 4000)

    def test_cap_during_raw_write_preserves_inflight_without_publication(self):
        self.new.mkdir(); (self.new / "records").mkdir(); (self.new / "inflight").mkdir()
        budget = r.ArtifactBudget(self.new, 4196, outer_reserve=0)
        with self.assertRaises(r.ArtifactLimit):
            r.write_record(self.new, self.names[0], self.record(self.names[0]), budget)
        self.assertFalse((self.new / "records" / self.names[0]).exists())
        self.assertTrue((self.new / "inflight" / self.names[0]).exists())
        self.assertLessEqual(budget.scan(), 100)

    def test_parent_log_reserve_guard(self):
        parent = self.root / "run"; parent.mkdir()
        output = parent / "evaluation"; output.mkdir()
        budget = r.ArtifactBudget(output, 16384, outer_reserve=100)
        (parent / "job.log").write_bytes(b"x" * 101)
        with self.assertRaisesRegex(r.ArtifactLimit, "outer_log"):
            budget.put(output / "data.json", {})
        budget.stop("outer_log_reserve_exceeded")
        self.assertFalse((output / "data.json").exists())
        self.assertEqual(r.read(output / "administrative_stop.json")["observed_parent_run_bytes"], 101)

    def test_report_cap_never_publishes_partial_performance_summary(self):
        self.prepare()
        self.e.run_one = lambda request, label, spec, **kwargs: self.record(self.case(**request).case_id + "--" + label + ".json.gz")
        self.e.report_rows = lambda rows, **kwargs: {"rows": rows}
        original_put = r.ArtifactBudget.put
        def limited_put(budget, path, value, **kwargs):
            if path.name == "summary_vs_r8.json":
                raise r.ArtifactLimit("max_new_artifact_bytes")
            return original_put(budget, path, value, **kwargs)
        supervisor = types.SimpleNamespace(require_outer_supervisor=lambda root: None)
        with patch.object(r, "ProcessPoolExecutor", ImmediatePool), patch.object(r.ArtifactBudget, "put", limited_put), patch.dict(sys.modules, {"scripts.q4_training_pair": supervisor}):
            code = r.execute(self.root, self.new, r.sha(self.new / "recovery.json"), self.e, self.CAP)
        self.assertEqual(code, 75)
        self.assertEqual(len(list((self.new / "records").glob("*.gz"))), 6)
        self.assertTrue((self.new / "inflight-reports/summary.json").exists())
        self.assertFalse((self.new / "summary.json").exists())
        self.assertFalse((self.new / "panel_complete.json").exists())

    def test_parent_logs_already_over_reserve_stop_before_execution(self):
        self.prepare()
        with patch.object(r.ArtifactBudget, "outer_bytes", return_value=r.OUTER_LOG_RESERVE_BYTES + 1):
            code = r.execute(self.root, self.new, r.sha(self.new / "recovery.json"), self.e, self.CAP)
        self.assertEqual(code, 75)
        self.assertFalse((self.new / "execution.started.json").exists())
        self.assertEqual(r.read(self.new / "administrative_stop.json")["stop_reason"], "outer_log_reserve_exceeded")

    def test_postpublication_resource_stop_reports_actual_files(self):
        self.prepare()
        self.e.run_one = lambda request, label, spec, **kwargs: self.record(self.case(**request).case_id + "--" + label + ".json.gz")
        self.e.report_rows = lambda rows, **kwargs: {"rows": rows}
        real_audit, calls = r.ArtifactBudget.audit, []
        def audit(budget):
            calls.append(True)
            if len(calls) == 2:
                raise r.ArtifactLimit("outer_log_reserve_exceeded")
            return real_audit(budget)
        supervisor = types.SimpleNamespace(require_outer_supervisor=lambda root: None)
        with patch.object(r, "ProcessPoolExecutor", ImmediatePool), patch.object(r.ArtifactBudget, "audit", audit), patch.dict(sys.modules, {"scripts.q4_training_pair": supervisor}):
            code = r.execute(self.root, self.new, r.sha(self.new / "recovery.json"), self.e, self.CAP)
        stopped = r.read(self.new / "administrative_stop.json")
        self.assertEqual(code, 75)
        self.assertTrue(stopped["performance_report_published"])
        self.assertTrue(stopped["panel_complete_marker_exists"])
        self.assertEqual(len(stopped["published_report_files"]), 3)
        self.assertFalse(stopped["resource_exit_75_qualifies_for_promotion"])


if __name__ == "__main__":
    unittest.main()
