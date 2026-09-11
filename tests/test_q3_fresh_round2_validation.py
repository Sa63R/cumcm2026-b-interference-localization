"""Only synthetic temporary SQLite fixtures; never open the research dataset."""

import hashlib
import json
import sqlite3

import pytest

from experiments.export_q3_fresh_validation import compatible, export
from experiments.export_q3_fresh_round2_validation import (
    DEFAULT_EXCLUDE_REPORT, SELECTION_SALT, export_round2, main,
)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture
def fake_data(tmp_path):
    database = tmp_path / "synthetic.sqlite3"
    rows = [{"id": i + 1, "case_code": f"fake-case-{i:03d}", "evidence": digest(f"fake-evidence-{i}")}
            for i in range(30)]
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            CREATE TABLE episodes (id INTEGER PRIMARY KEY, case_code TEXT UNIQUE, problem INTEGER,
                complete INTEGER, cleared_count INTEGER, source_total INTEGER, trajectory_complete INTEGER,
                evidence_sha256 TEXT, virtual_time_s REAL, policy TEXT);
            CREATE TABLE source_estimates (episode_id INTEGER, channel INTEGER, label_json TEXT);
            CREATE TABLE steps (episode_id INTEGER, accepted INTEGER, step_index INTEGER, channel INTEGER,
                x_m REAL, y_m REAL, response_json TEXT, action TEXT);
        """)
        for row in rows:
            connection.execute("INSERT INTO episodes VALUES (?,?,3,1,10,10,1,?,4000,'synthetic-policy')",
                               (row["id"], row["case_code"], row["evidence"]))
            for channel in range(1, 11):
                clear_index = 3 * channel
                label = dict(channel=channel, x_m=1100, y_m=0, radius_m=2,
                             outer_vertices=[[1099, -1], [1101, -1], [1101, 1], [1099, 1]],
                             geometry_consistent=True, cleared=True, clear_step_index=clear_index)
                connection.execute("INSERT INTO source_estimates VALUES (?,?,?)",
                                   (row["id"], channel, json.dumps(label)))
                for step_index, x, response, action in (
                    (clear_index - 2, 0, {"measure_result": "direction", "svd_deg": 0}, "measure"),
                    (clear_index - 1, -100, {"measure_result": "no_signal"}, "measure"),
                    (clear_index, 1100, {"clear_result": "success"}, "clear"),
                ):
                    connection.execute("INSERT INTO steps VALUES (?,1,?,?,?,0,?,?)",
                                       (row["id"], step_index, channel, x, json.dumps(response), action))
            connection.execute("INSERT INTO steps VALUES (?,1,31,20,0,0,?,'measure')",
                               (row["id"], json.dumps({"measure_result": "no_signal"})))
    old_rows = sorted(rows, key=lambda row: digest("fresh-q3-v1:" + row["case_code"]))[:12]
    exclusion_report = tmp_path / "fake-round1-report.json"
    excluded = [row["evidence"] for row in old_rows]
    exclusion_report.write_text(json.dumps({"metadata": {"evidence_sha256": excluded}}), encoding="utf-8")
    return database, exclusion_report, rows, excluded


def test_exact_salted_selection_after_exclusion_and_twelve_groups(fake_data):
    database, exclusion_report, rows, excluded = fake_data
    expected = sorted([row for row in rows if row["evidence"] not in excluded],
                      key=lambda row: digest("fresh-q3-v2:" + row["case_code"]))[:12]
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    result = export_round2(database, exclude_report=exclusion_report)
    metadata = result["metadata"]
    assert metadata["evidence_sha256"] == [row["evidence"] for row in expected]
    assert metadata["new_evidence_sha256"] == metadata["evidence_sha256"]
    assert metadata["selected_evidence_sha256"] == metadata["evidence_sha256"]
    assert not set(metadata["evidence_sha256"]) & set(excluded)
    assert metadata["excluded_evidence_sha256"] == sorted(excluded)
    assert metadata["excluded_report_sha256"] == hashlib.sha256(exclusion_report.read_bytes()).hexdigest()
    assert metadata["selection_salt"] == SELECTION_SALT == "fresh-q3-v2:"
    assert metadata["original_group_count"] == metadata["requested_groups"] == 12
    assert metadata["reconstructed_case_count"] == len(result["cases"]) == 24
    assert metadata["eligible_before_exclusion"] == 30
    assert metadata["excluded_eligible_episodes"] == 12
    assert metadata["eligible_episodes"] == 18
    assert [group["case_code"] for group in metadata["original_groups"]] == [row["case_code"] for row in expected]
    assert all(case["case_id"].startswith("validation2-") for case in result["cases"])
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


def test_same_witness_and_two_interior_radii_preserve_history(fake_data):
    database, exclusion_report, _, _ = fake_data
    result = export_round2(database, count=2, exclude_report=exclusion_report)
    assert result["metadata"]["radius_fractions"] == [0.25, 0.75]
    for case, fraction in zip(result["cases"], [0.25, 0.75] * 2):
        group = case["case_id"].rsplit("-radius-", 1)[0]
        assert case["case_id"] in next(item["case_ids"] for item in result["metadata"]["original_groups"]
                                       if item["group_id"] == group)
        anchors = result["metadata"]["anchors"][case["case_id"]]
        assert any(item["channel"] == 20 for item in anchors)
        for source in case["sources"]:
            assert (source["x"], source["y"]) == (1100, 0)
            events = [dict(x_m=item["position"][0], y_m=item["position"][1],
                           response_json=json.dumps(item["response"]))
                      for item in anchors if item["channel"] == source["channel"]]
            interval = compatible((source["x"], source["y"]), events)
            assert interval == pytest.approx((1100, 1200 - 1e-7))
            assert source["reception_radius_m"] == pytest.approx(1100 + fraction * (100 - 1e-7))
            assert interval[0] <= source["reception_radius_m"] <= interval[1]
            assert result["metadata"]["uncertainty_radii"][case["case_id"]][str(source["channel"])] == 2


def test_evaluated_report_manifest_metadata_structure(fake_data):
    database, exclusion_report, _, excluded = fake_data
    direct = export_round2(database, count=2, exclude_report=exclusion_report)
    exclusion_report.write_text(json.dumps({
        "manifest": {"metadata": {"evidence_sha256": excluded}},
        "summary": {}, "rows": [],
    }), encoding="utf-8")
    wrapped = export_round2(database, count=2, exclude_report=exclusion_report)
    assert wrapped["cases"] == direct["cases"]
    assert wrapped["metadata"]["excluded_evidence_sha256"] == sorted(excluded)
    assert wrapped["metadata"]["new_evidence_sha256"] == direct["metadata"]["new_evidence_sha256"]


def test_round_one_defaults_preserve_selection_seeds_and_ids(fake_data):
    database, _, rows, _ = fake_data
    original = export(database, 3)
    explicit = export(database, 3, selection_salt="fresh-q3-v1:", excluded_evidence=(),
                      case_prefix="validation")
    assert original == explicit
    expected = sorted(rows, key=lambda row: digest("fresh-q3-v1:" + row["case_code"]))[:3]
    assert original["metadata"]["evidence_sha256"] == [row["evidence"] for row in expected]
    assert [case["case_id"] for case in original["cases"]] == [
        f"validation-{i:03d}-{suffix}" for i in range(3) for suffix in ("radius-low", "radius-high")]
    assert [case["seed"] for case in original["cases"]] == [923000, 923000, 923001, 923001, 923002, 923002]


def test_failed_reconstruction_is_not_silently_replaced(fake_data):
    database, exclusion_report, rows, excluded = fake_data
    first = min((row for row in rows if row["evidence"] not in excluded),
                key=lambda row: digest(SELECTION_SALT + row["case_code"]))
    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM source_estimates WHERE episode_id=?", (first["id"],))
    with pytest.raises(ValueError, match="Only 1 of 2 selected groups reconstructed"):
        export_round2(database, count=2, exclude_report=exclusion_report)


def test_insufficient_new_groups_fails(fake_data):
    database, exclusion_report, _, _ = fake_data
    with pytest.raises(ValueError, match="Only 18 eligible new groups"):
        export_round2(database, count=19, exclude_report=exclusion_report)


@pytest.mark.parametrize("evidence", [None, [], ["bad"] * 12, ["a" * 64] * 12])
def test_invalid_exclusion_report_fails_before_opening_database(tmp_path, evidence):
    report = tmp_path / "invalid-report.json"
    report.write_text(json.dumps({"metadata": {"evidence_sha256": evidence}}), encoding="utf-8")
    with pytest.raises(ValueError, match="12 unique"):
        export_round2(tmp_path / "never-open.sqlite3", exclude_report=report)
    assert not (tmp_path / "never-open.sqlite3").exists()


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_invalid_count_fails_before_any_input_read(tmp_path, count):
    with pytest.raises(ValueError, match="positive integer"):
        export_round2(tmp_path / "not-opened.sqlite3", count, tmp_path / "not-opened.json")


def test_cli_uses_explicit_paths_and_refuses_overwrite(fake_data, tmp_path, capsys):
    database, exclusion_report, _, _ = fake_data
    output = tmp_path / "new" / "validation2.json"
    arguments = ["--database", str(database), "--out", str(output), "--count", "2",
                 "--exclude-report", str(exclusion_report)]
    main(arguments)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["metadata"]["original_group_count"] == 2
    assert json.loads(capsys.readouterr().out)["reconstructed_case_count"] == 4
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        main(arguments)
    assert output.read_bytes() == before
    assert DEFAULT_EXCLUDE_REPORT.as_posix().endswith("results/q3_fresh/validation/results.json")
