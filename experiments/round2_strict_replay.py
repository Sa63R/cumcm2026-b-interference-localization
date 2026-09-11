"""Validate frozen Q3 code against exact prefixes of an offline practice snapshot.

This is not a counterfactual simulator or an off-policy performance estimator.
Run only AFTER freezing the candidate. Never use this validation data to train,
choose parameters, infer a distribution, or manufacture unrecorded feedback.

Manifest fields: validation_role='validation_only', database_sha256,
source_root, source_sha256 (all src/**/*.py), spec (entrypoint and kwargs),
spec_sha256 (canonical JSON), and optional asset_sha256 (relative file paths).
CLI: python experiments/round2_strict_replay.py --database SNAPSHOT.sqlite3
     --manifest frozen.json --output NEW_RESULTS.json
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import time
from types import SimpleNamespace


class ReplayDivergence(RuntimeError):
    """No real feedback is available for the requested action after this prefix."""


class ReplayEvidenceError(RuntimeError):
    """The saved evidence cannot support even this matching transition."""


class RecordedRejection(RuntimeError):
    """Exact saved rejection; HTTP status/transport timing are not reconstructed."""

    def __init__(self, response):
        self.response = copy.deepcopy(response)
        super().__init__("Recorded action was rejected; state and cost are unchanged")


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def action_key(action, parameters):
    """Exact numeric coordinates: no rounding, distance tolerance or interpolation."""
    if action not in {"enter", "measure", "clear", "exit"}:
        raise ReplayEvidenceError("Unsupported action type")
    if action in {"enter", "exit"}:
        return (action,)
    from simulator_client.state import Position
    channel = parameters.get("channel")
    if type(channel) is not int or channel not in range(1, 21):
        raise ReplayEvidenceError("Invalid channel")
    point = parameters.get("position", {})
    if not isinstance(point, dict) or set(point) != {"x", "y"}:
        raise ReplayEvidenceError("Invalid position fields")
    position = Position(point["x"], point["y"])
    # Numeric equality identifies only identical binary values, except signed
    # zero and integer/float spellings representing the very same coordinate.
    return action, channel, position.x, position.y


class CausalGeometryAudit:
    """Conservative Q3 constraints from already returned responses only."""

    def __init__(self):
        self.regions = {}
        self.cleared = set()
        self.empty_updates = 0
        self.clear_certificates = 0
        self.uncertified_clear_attempts = 0
        self.failed_clears = 0

    def update(self, action, parameters, response):
        if action not in {"measure", "clear"} or not response["accepted"]:
            return
        from geometry import clip_polygon, disk_halfplanes
        from localization.omni import OmniCandidateRegion
        channel = parameters["channel"]
        position = (parameters["position"]["x"], parameters["position"]["y"])
        if channel in self.cleared:
            if action == "measure" and response.get("measure_result") != "no_signal":
                raise ReplayEvidenceError("Recorded signal follows successful removal")
            if action == "clear" and response.get("clear_result") == "success":
                raise ReplayEvidenceError("Recorded source is removed twice")
            if action == "clear":
                self.failed_clears += 1
            return
        region = self.regions.setdefault(channel, OmniCandidateRegion())
        radius = None
        if action == "clear":
            certified = bool(region.vertices) and all(
                math.dist(vertex, position) <= 20.0 for vertex in region.vertices)
            self.clear_certificates += int(certified)
            self.uncertified_clear_attempts += int(not certified)
            if response["clear_result"] == "success":
                radius = 20.0
                self.cleared.add(channel)
            else:
                self.failed_clears += 1
        elif response["measure_result"] == "direction":
            region.observe(position, response["svd_deg"])
        elif response["measure_result"] == "no_signal":
            region.observe_no_signal(position)
        else:
            radius = 5.0
        if radius is not None:
            for plane in disk_halfplanes(position, radius, 128, outer=True):
                region.vertices = clip_polygon(region.vertices, plane)
            region._circle = None
        self.empty_updates += int(not region.vertices)

    def summary(self):
        return {"empty_outer_region_updates": self.empty_updates,
                "clear_attempts_with_prior_certificate": self.clear_certificates,
                "uncertified_clear_attempts": self.uncertified_clear_attempts,
                "recorded_failed_clears_in_prefix": self.failed_clears,
                "semantics": "conservative_prefix_constraints_not_hidden_truth_validation"}


class StrictPrefixTape:
    """Lazily read at most the next recorded step; permanently lock on divergence."""

    def __init__(self, records):
        self._records = iter(records)
        self.matched_steps = 0
        self.loaded_steps = 0
        self.accepted_steps = 0
        self.rejected_steps = 0
        self.recorded_retry_attempts = 0
        self.first_unsupported = None
        self.evidence_error = None
        self.terminal_matched = False
        self.last_record = None

    def _stop(self, requested, expected, reason):
        self.first_unsupported = {
            "step_index": self.matched_steps, "reason": reason,
            "requested_action": requested[0],
            "recorded_action": expected[0] if expected else None,
            "channel_matches": (requested[1] == expected[1]
                                if expected and len(requested) > 1 and len(expected) > 1 else None),
            "position_matches": (requested[2:] == expected[2:]
                                 if expected and len(requested) > 1 and len(expected) > 1 else None)}
        raise ReplayDivergence("unsupported_counterfactual_action")

    def exchange(self, path, parameters):
        if self.first_unsupported is not None:
            raise ReplayDivergence("Replay is locked after the first unsupported action")
        if self.evidence_error is not None:
            raise ReplayEvidenceError("Replay is locked after invalid evidence")
        requested = action_key(path.lstrip("/"), parameters)
        if self.terminal_matched:
            self._stop(requested, None, "request_after_recorded_exit")
        try:
            record = next(self._records)
        except StopIteration:
            self._stop(requested, None, "recorded_trajectory_exhausted")
        except Exception as error:
            self.evidence_error = "record_decode_error"
            raise ReplayEvidenceError(self.evidence_error) from error
        self.loaded_steps += 1
        try:
            expected = action_key(record["action"], record["action_parameters"])
            if (type(record["attempts"]) is not int or record["attempts"] < 1
                    or type(record["step_index"]) is not int):
                raise ReplayEvidenceError("Invalid logical step metadata")
            response = copy.deepcopy(record["response"])
        except (ValueError, KeyError, TypeError, ReplayEvidenceError) as error:
            self.evidence_error = "record_metadata_error"
            raise ReplayEvidenceError(self.evidence_error) from error
        if requested != expected:
            self._stop(requested, expected, "action_channel_or_exact_coordinate_differs")
        if record["step_index"] != self.matched_steps:
            self.evidence_error = "non_contiguous_step_index"
            raise ReplayEvidenceError(self.evidence_error)
        self.last_record = record
        return response

    def commit(self, response):
        self.matched_steps += 1
        self.accepted_steps += int(response["accepted"])
        self.rejected_steps += int(not response["accepted"])
        self.recorded_retry_attempts += max(0, self.last_record["attempts"] - 1)
        self.terminal_matched = bool(response["accepted"] and self.last_record["action"] == "exit")


def _check_costs(record, before, expected, response):
    """Validate physical increments, including unchanged rejected actions."""
    if (type(record["accepted"]) not in {bool, int} or record["accepted"] not in (0, 1)
            or bool(record["accepted"]) != response["accepted"]):
        raise ReplayEvidenceError("Accepted flag disagrees with recorded response")
    after = float(response["virtual_time_s"])
    for saved, actual in ((record["virtual_time_before_s"], before),
                          (record["virtual_time_after_s"], after),
                          (record["delta_virtual_time_s"], after - before)):
        if not math.isclose(saved, actual, rel_tol=1e-10, abs_tol=2e-5):
            raise ReplayEvidenceError("Recorded transition time disagrees with causal state")
    saved = record["cost_components"]
    for name in set(saved) | set(expected):
        if not math.isclose(saved.get(name, 0.0), expected.get(name, 0.0),
                            rel_tol=1e-10, abs_tol=2e-5):
            raise ReplayEvidenceError("Recorded physical cost component mismatch")
    if not response["accepted"] and after != before:
        raise ReplayEvidenceError("Rejected action changed virtual time")


def _make_client(tape, clock):
    from simulation.engine import MemoryClient

    class AuditedMemoryClient(MemoryClient):
        def __init__(self):
            super().__init__(tape.exchange, clock=clock)
            self.geometry_audit = CausalGeometryAudit()

        def _new_action(self, path, fields):
            self._sequence += 1
            action = SimpleNamespace(path=path, payload=fields, first_sent_at=self._clock())
            response = tape.exchange(path, fields)
            try:
                self._validate_common(response)
                costs = self._increments(action, response) if response["accepted"] else {}
                _check_costs(tape.last_record, self.state.virtual_time_s, costs, response)
                if response["accepted"]:
                    self._validate_accepted(action, response)
                    self.geometry_audit.update(path.lstrip("/"), fields, response)
                    self._apply_accepted(action, response)
                tape.commit(response)
            except (ValueError, KeyError, TypeError, ReplayEvidenceError) as error:
                tape.evidence_error = type(error).__name__
                raise ReplayEvidenceError("Matching recorded transition failed evidence audit") from error
            if not response["accepted"]:
                raise RecordedRejection(response)
            return response

    return AuditedMemoryClient()


class PrefixReplayClient:
    """Policy-facing whitelist. No tape, source labels or episode metadata API."""

    __slots__ = ("__client",)

    def __init__(self, client):
        self.__client = client

    @property
    def state(self):
        return copy.deepcopy(self.__client.state)

    @property
    def remaining_real_time_s(self):
        return self.__client.remaining_real_time_s

    @property
    def pending_request(self):
        return None

    def enter(self):
        return self.__client.enter()

    def measure(self, position, channel):
        return self.__client.measure(position, channel)

    def clear(self, position, channel):
        return self.__client.clear(position, channel)

    def exit(self):
        return self.__client.exit()


def validate_policy(records, solver, kwargs=None, *, clock=time.monotonic):
    """Run a callable on a causal prefix. Cleanup never submits a synthetic exit."""
    tape = StrictPrefixTape(records)
    client = _make_client(tape, clock)
    policy_error = None
    started = time.perf_counter()
    try:
        solver(PrefixReplayClient(client), **(kwargs or {}))
    except ReplayDivergence:
        pass
    except RecordedRejection:
        policy_error = "recorded_rejection_stopped_policy"
    except ReplayEvidenceError:
        policy_error = "recorded_evidence_audit_error"
    except Exception as error:
        # Arbitrary exception strings may accidentally disclose saved values.
        policy_error = type(error).__name__
    finally:
        client.close()  # close is local; it never advances the tape.
    if tape.first_unsupported:
        status = "unsupported_counterfactual_action"
    elif tape.evidence_error:
        status = "invalid_recorded_evidence"
    elif policy_error:
        status = "policy_stopped_on_prefix"
    elif tape.terminal_matched:
        status = "recorded_trajectory_matched"
    else:
        status = "policy_returned_before_recorded_exit"
    return {"status": status, "matched_steps": tape.matched_steps,
            "loaded_steps": tape.loaded_steps, "accepted_steps": tape.accepted_steps,
            "rejected_steps": tape.rejected_steps,
            "recorded_retry_attempts_not_resimulated": tape.recorded_retry_attempts,
            "first_unsupported": tape.first_unsupported, "policy_error": policy_error,
            "matched_prefix_virtual_time_s": client.state.virtual_time_s,
            "matched_prefix_costs": asdict(client.state.time_breakdown),
            "wall_time_s": time.perf_counter() - started,
            "geometry_audit": client.geometry_audit.summary(),
            "performance_comparison_supported": False,
            "truncation_is_policy_failure": False}


def sqlite_steps(connection, episode_id):
    """No labels, final counts, state BLOBs or future observations are selected."""
    query = """SELECT step_index,action,action_json,response_json,accepted,attempts,
        virtual_time_before_s,virtual_time_after_s,delta_virtual_time_s,cost_components_json
        FROM steps WHERE episode_id=? ORDER BY step_index"""
    for row in connection.execute(query, (episode_id,)):
        yield {"step_index": row[0], "action": row[1],
               "action_parameters": json.loads(row[2]), "response": json.loads(row[3]),
               "accepted": row[4], "attempts": row[5], "virtual_time_before_s": row[6],
               "virtual_time_after_s": row[7], "delta_virtual_time_s": row[8],
               "cost_components": json.loads(row[9])}


def verify_manifest(manifest, database):
    if manifest.get("validation_role") != "validation_only":
        raise ValueError("Manifest must preserve the entire snapshot as validation-only")
    if file_hash(database) != manifest.get("database_sha256"):
        raise ValueError("Database is not the frozen snapshot")
    if ("validator_sha256" in manifest
            and manifest["validator_sha256"] != file_hash(__file__)):
        raise ValueError("Frozen validator hash mismatch")
    root = Path(manifest["source_root"]).resolve(strict=True)
    actual = {p.relative_to(root).as_posix(): file_hash(p)
              for p in (root / "src").rglob("*.py")}
    if not actual or actual != manifest.get("source_sha256"):
        raise ValueError("Frozen source manifest does not match complete source tree")
    spec = manifest["spec"]
    if canonical_hash(spec) != manifest.get("spec_sha256"):
        raise ValueError("Frozen policy spec hash mismatch")
    if not isinstance(spec.get("kwargs", {}), dict) or ":" not in spec.get("entrypoint", ""):
        raise ValueError("Invalid frozen policy spec")
    for relative, digest in manifest.get("asset_sha256", {}).items():
        asset = (root / relative).resolve(strict=True)
        if not asset.is_relative_to(root) or file_hash(asset) != digest:
            raise ValueError("Frozen policy asset mismatch or external asset path")
    return root, spec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Output already exists; keep prior validation evidence")
    manifest_digest = file_hash(args.manifest)
    validator_digest = file_hash(__file__)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root, spec = verify_manifest(manifest, args.database)
    sys.path.insert(0, str(root / "src"))
    module_name, function_name = spec["entrypoint"].split(":")
    module = importlib.import_module(module_name)
    if not Path(module.__file__).resolve().is_relative_to(root / "src"):
        raise ValueError("Policy module was imported outside its frozen source root")
    solver = getattr(module, function_name)
    rows = []
    with sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        # Include all Q3 episodes, including incomplete recordings; never select
        # according to internal train/test labels or a final success outcome.
        ids = [row[0] for row in connection.execute("SELECT id FROM episodes WHERE problem=3 ORDER BY id")]
        for ordinal, episode_id in enumerate(ids):
            row = validate_policy(sqlite_steps(connection, episode_id), solver, spec.get("kwargs", {}))
            rows.append({"episode_ordinal": ordinal, **row})
    if file_hash(args.database) != manifest["database_sha256"]:
        raise ValueError("Snapshot changed during validation; results must not be used")
    # Catch a concurrent source/asset change rather than attributing results to
    # an earlier revision. The evaluator itself is fingerprinted separately.
    verify_manifest(manifest, args.database)
    if file_hash(args.manifest) != manifest_digest or file_hash(__file__) != validator_digest:
        raise ValueError("Manifest or validator changed during validation")
    result = {"validation_role": "validation_only", "problem": 3,
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "manifest_sha256": manifest_digest,
              "snapshot_sha256": manifest["database_sha256"],
              "validator_sha256": validator_digest, "episodes": rows,
              "summary": {status: sum(row["status"] == status for row in rows)
                          for status in sorted({row["status"] for row in rows})},
              "warning": "Exact-prefix validation only; no counterfactual time or all-clear comparison"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
    print(json.dumps({"episodes": len(rows), "statuses": result["summary"]}))
    return int(any(row["status"] == "invalid_recorded_evidence" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
