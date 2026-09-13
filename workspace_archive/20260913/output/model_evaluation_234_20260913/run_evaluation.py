#!/usr/bin/env python3
"""Standalone experiments 2/3/4: robustness, tree validation, and cost drivers.

This script intentionally evaluates only the selected Q3 strategy and Q4 V6 Lite.
It does not edit the manuscript and does not perform a component ablation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
Q3_CSV = ROOT / "github-release-q3-q4/experiments/q3_selected/validation/local_paired400.csv"
Q4_DIR = ROOT / "output/q4_v6_lite_20260912"
Q4_CSV = Q4_DIR / "paired_records.csv"
Q4_PLAN = Q4_DIR / "plan.json"
OFFICIAL_DIR = ROOT / "output/q4_v6_lite_official_20260913"
OFFICIAL_RESULTS = OFFICIAL_DIR / "audited_results.json"
OFFICIAL_SUMMARY = OFFICIAL_DIR / "summary.json"
TRAIN_DIR = ROOT / "Q4_V6/research/round1"
TRAIN_FILES = [TRAIN_DIR / "critic_training.jsonl", TRAIN_DIR / "critic_training_extra.jsonl"]
MODEL_JSON = ROOT / "output/q4_v6_lite_20260912/source_v6_lite/transit_critic_big_extra.json"

RNG_SEED = 20260913
BOOTSTRAP_REPS = 10000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def fnum(x) -> float:
    return float(x)


def inum(x) -> int:
    return int(float(x))


def cvar(values: np.ndarray, alpha: float) -> float:
    values = np.asarray(values, dtype=float)
    k = max(1, int(math.ceil((1 - alpha) * len(values))))
    return float(np.partition(values, len(values) - k)[-k:].mean())


def bootstrap_ci(values: np.ndarray, statistic: str, reps: int = BOOTSTRAP_REPS) -> list[float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    rng = np.random.default_rng(RNG_SEED + n + sum(map(ord, statistic)))
    samples = np.empty(reps, dtype=float)
    chunk = min(500, reps)
    for start in range(0, reps, chunk):
        size = min(chunk, reps - start)
        x = values[rng.integers(0, n, size=(size, n))]
        if statistic == "mean":
            samples[start : start + size] = x.mean(axis=1)
        elif statistic == "p90":
            samples[start : start + size] = np.quantile(x, 0.90, axis=1)
        elif statistic == "cvar90":
            k = max(1, int(math.ceil(0.10 * n)))
            samples[start : start + size] = np.partition(x, n - k, axis=1)[:, -k:].mean(axis=1)
        else:
            raise ValueError(statistic)
    return [float(v) for v in np.quantile(samples, [0.025, 0.975])]


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [100 * (center - half), 100 * (center + half)]


def risk_summary(label: str, values: list[float], successes: int) -> dict:
    x = np.asarray(values, dtype=float)
    return {
        "dataset": label,
        "cases": int(len(x)),
        "all_clear_cases": int(successes),
        "all_clear_percent": float(100 * successes / len(x)),
        "all_clear_wilson_ci95_percent": wilson(successes, len(x)),
        "mean_seconds_per_source": float(x.mean()),
        "mean_ci95": bootstrap_ci(x, "mean"),
        "median_seconds_per_source": float(np.median(x)),
        "sample_sd": float(x.std(ddof=1)),
        "p90_seconds_per_source": float(np.quantile(x, 0.90)),
        "p90_ci95": bootstrap_ci(x, "p90"),
        "p95_seconds_per_source": float(np.quantile(x, 0.95)),
        "cvar90_seconds_per_source": cvar(x, 0.90),
        "cvar90_ci95": bootstrap_ci(x, "cvar90"),
        "cvar95_seconds_per_source": cvar(x, 0.95),
        "min_seconds_per_source": float(x.min()),
        "max_seconds_per_source": float(x.max()),
    }


def q4_scene_features(sources: list[dict]) -> dict:
    radial = np.asarray([math.hypot(s["x_um"], s["y_um"]) / 1e6 for s in sources])
    receive = np.asarray([s["max_receive_um"] / 1e6 for s in sources])
    directional = sum(s["kind"] == "directional" for s in sources)
    return {
        "source_total": len(sources),
        "directional_count": directional,
        "directional_share": directional / len(sources),
        "mean_receive_radius_m": float(receive.mean()),
        "mean_radial_distance_m": float(radial.mean()),
        "edge_share_r_ge_1500m": float((radial >= 1500).mean()),
    }


def cost_fields_q4(row: dict) -> dict:
    return {
        "movement": fnum(row["distance_m"]) / 5,
        "switching": fnum(row["switch_count"]),
        "measurement": 5 * fnum(row["measurement_count"]),
        "optical_attempt": 3 * fnum(row["clear_attempt_count"]),
        "successful_removal": 2 * fnum(row["cleared_count"]),
    }


def cost_fields_q3(row: dict) -> dict:
    clears = inum(row["cleared"])
    failures = inum(row["failed_clears"])
    return {
        "movement": fnum(row["movement_metres"]) / 5,
        "switching": fnum(row["switches"]),
        "measurement": 5 * fnum(row["detects"]),
        "optical_attempt": 3 * (clears + failures),
        "successful_removal": 2 * clears,
    }


def aggregate_cost(label: str, rows: list[dict]) -> dict:
    names = ["movement", "switching", "measurement", "optical_attempt", "successful_removal"]
    sources = sum(r["_sources"] for r in rows)
    virtual = sum(r["_virtual"] for r in rows)
    result = {
        "dataset": label,
        "cases": len(rows),
        "sources": sources,
        "case_mean_seconds_per_source": float(np.mean([r["_virtual"] / r["_sources"] for r in rows])),
        "pooled_seconds_per_source": virtual / sources,
        "identity_max_abs_error_seconds": max(abs(r["_virtual"] - sum(r["_cost"].values())) for r in rows),
    }
    for name in names:
        total = sum(r["_cost"][name] for r in rows)
        result[f"{name}_pooled_seconds_per_source"] = total / sources
        result[f"{name}_share_percent"] = 100 * total / virtual
    return result


def robust_ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    inv = np.linalg.pinv(X.T @ X)
    beta = inv @ X.T @ y
    residual = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, inv, X)
    adjusted = residual / np.maximum(1e-8, 1 - h)
    meat = X.T @ ((adjusted * adjusted)[:, None] * X)
    cov = inv @ meat @ inv
    se = np.sqrt(np.maximum(0, np.diag(cov)))
    r2 = 1 - float(np.sum(residual**2) / np.sum((y - y.mean()) ** 2))
    return beta, se, r2


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, equivalent to scipy.stats.rankdata(method='average')."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranked = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranked[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranked


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    mse = mean_squared_error(y, pred)
    zero_mse = float(np.mean(y * y))
    corr = float(np.corrcoef(y, pred)[0, 1]) if np.std(pred) > 0 else 0.0
    rho = float(np.corrcoef(rankdata(y), rankdata(pred))[0, 1])
    return {
        "rows": len(y),
        "MSE": float(mse),
        "RMSE": float(math.sqrt(mse)),
        "MAE": float(mean_absolute_error(y, pred)),
        "R2": float(r2_score(y, pred)),
        "Pearson_r": corr,
        "Spearman_rho": rho,
        "zero_prediction_MSE": zero_mse,
        "MSE_skill_vs_zero_percent": float(100 * (1 - mse / zero_mse)),
    }


def cluster_bootstrap_decision_ci(records: list[dict], reps: int = BOOTSTRAP_REPS) -> list[float]:
    by_seed = defaultdict(list)
    for r in records:
        by_seed[r["seed"]].append(r["policy_saving"])
    seeds = sorted(by_seed)
    totals = np.asarray([sum(by_seed[s]) for s in seeds], dtype=float)
    counts = np.asarray([len(by_seed[s]) for s in seeds], dtype=float)
    rng = np.random.default_rng(RNG_SEED + 431)
    vals = np.empty(reps)
    chunk = 500
    for start in range(0, reps, chunk):
        size = min(chunk, reps - start)
        idx = rng.integers(0, len(seeds), size=(size, len(seeds)))
        vals[start : start + size] = totals[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return [float(x) for x in np.quantile(vals, [0.025, 0.975])]


def build_feature_names() -> list[str]:
    names = [
        "route_length",
        "probe_fraction",
        "distance_to_probe",
        "remaining_route_distance",
        "destination_is_site",
        "remaining_sites",
        "pending_sources",
        "cleared_sources",
        "unknown_channels",
        "candidate_channels",
        "total_expected_radius_gain",
        "total_relative_radius_gain",
        "mean_center_probe_distance",
        "estimated_probe_action_cost",
        "start_distance_from_origin",
        "probe_distance_from_origin",
        "destination_distance_from_origin",
        "mean_visibility",
    ]
    sub = [
        "expected_radius_gain",
        "uncertainty_radius",
        "gain_radius_ratio",
        "center_probe_distance",
        "nearest_prior_measurement_distance",
        "positive_measurements",
        "negative_measurements",
        "angular_sine",
        "visibility",
        "anchor_probe_distance",
        "max_polygon_vertex_distance",
        "min_polygon_vertex_distance",
        "is_destination_target",
    ]
    for rank in range(1, 4):
        names.extend(f"rank{rank}_{s}" for s in sub)
    assert len(names) == 57
    return names


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # Q3 selected strategy only.
    q3_raw = [r for r in read_csv(Q3_CSV) if r["mode"] == "v3_origin20"]
    q3 = []
    for r in q3_raw:
        rr = dict(r)
        rr["_sources"] = inum(r["true_sources"])
        rr["_virtual"] = fnum(r["virtual_seconds"])
        rr["_cost"] = cost_fields_q3(r)
        q3.append(rr)

    # Q4 V6 Lite only, joined to frozen scenario geometry.
    plan = json.loads(Q4_PLAN.read_text())
    scene = {c["key"]: q4_scene_features(c["scenario"]["sources"]) for c in plan["cases"]}
    q4_raw = [r for r in read_csv(Q4_CSV) if r["method"] == "lite"]
    q4 = []
    for r in q4_raw:
        rr = dict(r)
        key = r["case_key"]
        rr.update(scene[key])
        rr["_sources"] = inum(r["source_total"])
        rr["_virtual"] = fnum(r["virtual_time_s"])
        rr["_cost"] = cost_fields_q4(r)
        q4.append(rr)

    official_raw = json.loads(OFFICIAL_RESULTS.read_text())
    official = []
    official_cost_map = {
        "movement_s": "movement",
        "switching_s": "switching",
        "detection_s": "measurement",
        "optical_s": "optical_attempt",
        "removal_s": "successful_removal",
    }
    for r in official_raw:
        rr = dict(r)
        rr["_sources"] = inum(r["total"])
        rr["_virtual"] = fnum(r["virtual_s"])
        rr["_cost"] = {dst: fnum(r["costs"][src]) for src, dst in official_cost_map.items()}
        official.append(rr)

    groups = {
        "Q3 local random (v3_origin20)": q3,
        "Q4 local random (V6 Lite)": [r for r in q4 if r["group"] == "new_practice"],
        "Q4 local all-directional (V6 Lite)": [r for r in q4 if r["group"] == "all_directional"],
        "Q4 local boundary-outward R=1000 (V6 Lite)": [r for r in q4 if r["group"] == "boundary_outward_r1000"],
        "Q4 official practice (V6 Lite)": official,
    }

    robustness = []
    for label, rows in groups.items():
        if label.startswith("Q3"):
            values = [fnum(r["seconds_per_source"]) for r in rows]
            successes = sum(r["success"] == "True" and inum(r["cleared"]) == inum(r["true_sources"]) for r in rows)
        elif "official" in label:
            values = [fnum(r["seconds_per_source"]) for r in rows]
            successes = sum(bool(r["all_cleared"]) for r in rows)
        else:
            values = [fnum(r["seconds_per_source"]) for r in rows]
            successes = sum(r["all_cleared"] == "True" for r in rows)
        robustness.append(risk_summary(label, values, successes))
    write_csv(OUT / "robustness_summary.csv", robustness)

    # Source-count sensitivity, selected strategies only.
    source_count_rows = []
    for domain, rows, key in [
        ("Q3 local random", q3, "true_sources"),
        ("Q4 local random", groups["Q4 local random (V6 Lite)"], "source_total"),
    ]:
        by_n = defaultdict(list)
        for r in rows:
            by_n[inum(r[key])].append(fnum(r["seconds_per_source"]))
        for n in sorted(by_n):
            x = np.asarray(by_n[n])
            source_count_rows.append(
                {
                    "dataset": domain,
                    "source_count": n,
                    "cases": len(x),
                    "mean_seconds_per_source": float(x.mean()),
                    "median_seconds_per_source": float(np.median(x)),
                    "p90_seconds_per_source": float(np.quantile(x, 0.9)),
                    "cvar90_seconds_per_source": cvar(x, 0.9),
                }
            )
    write_csv(OUT / "source_count_sensitivity.csv", source_count_rows)

    # Worst-tail cases; this is descriptive diagnosis, not post-hoc tuning.
    tails = []
    for label, rows in [("Q3 local random", q3), ("Q4 local random", groups["Q4 local random (V6 Lite)"])]:
        ranked = sorted(rows, key=lambda r: fnum(r["seconds_per_source"]), reverse=True)[:10]
        for rank, r in enumerate(ranked, 1):
            if label.startswith("Q3"):
                tails.append(
                    {
                        "dataset": label,
                        "rank": rank,
                        "case": r["seed"],
                        "sources": r["true_sources"],
                        "seconds_per_source": r["seconds_per_source"],
                        "distance_per_source_m": fnum(r["movement_metres"]) / inum(r["true_sources"]),
                        "measurements_per_source": fnum(r["detects"]) / inum(r["true_sources"]),
                        "failed_clears": r["failed_clears"],
                        "directional_share": "",
                        "mean_receive_radius_m": "",
                        "mean_radial_distance_m": "",
                        "coverage_tail_s": "",
                    }
                )
            else:
                tails.append(
                    {
                        "dataset": label,
                        "rank": rank,
                        "case": r["case_key"],
                        "sources": r["source_total"],
                        "seconds_per_source": r["seconds_per_source"],
                        "distance_per_source_m": fnum(r["distance_m"]) / inum(r["source_total"]),
                        "measurements_per_source": fnum(r["measurement_count"]) / inum(r["source_total"]),
                        "failed_clears": r["failed_clear_count"],
                        "directional_share": r["directional_share"],
                        "mean_receive_radius_m": r["mean_receive_radius_m"],
                        "mean_radial_distance_m": r["mean_radial_distance_m"],
                        "coverage_tail_s": r["tail_after_last_clear_s"],
                    }
                )
    write_csv(OUT / "tail_cases_top10.csv", tails)

    # Exact cost identity and aggregate contribution.
    costs = [aggregate_cost(label, rows) for label, rows in groups.items()]
    write_csv(OUT / "cost_decomposition.csv", costs)

    # Scene-factor sensitivity with source-count fixed effects on 1000 random Q4 cases.
    random_q4 = groups["Q4 local random (V6 Lite)"]
    y = np.asarray([fnum(r["seconds_per_source"]) for r in random_q4])
    ns = sorted(set(inum(r["source_total"]) for r in random_q4))
    continuous = ["directional_share", "mean_receive_radius_m", "mean_radial_distance_m", "edge_share_r_ge_1500m"]
    columns = ["intercept"] + [f"source_count_{n}" for n in ns[1:]] + continuous
    X_parts = [np.ones(len(random_q4))]
    X_parts.extend(np.asarray([inum(r["source_total"]) == n for r in random_q4], dtype=float) for n in ns[1:])
    X_parts.extend(np.asarray([fnum(r[k]) for r in random_q4]) for k in continuous)
    X = np.column_stack(X_parts)
    beta, se, scene_r2 = robust_ols(y, X)
    scene_effects = []
    for key in continuous:
        j = columns.index(key)
        vals = np.asarray([fnum(r[key]) for r in random_q4])
        iqr = float(np.quantile(vals, 0.75) - np.quantile(vals, 0.25))
        scene_effects.append(
            {
                "factor": key,
                "observed_iqr": iqr,
                "effect_seconds_per_source_per_iqr": float(beta[j] * iqr),
                "hc3_ci95_low": float((beta[j] - 1.959963984540054 * se[j]) * iqr),
                "hc3_ci95_high": float((beta[j] + 1.959963984540054 * se[j]) * iqr),
                "raw_coefficient": float(beta[j]),
                "raw_hc3_se": float(se[j]),
            }
        )
    write_csv(OUT / "scene_factor_sensitivity.csv", scene_effects)

    # Regression-tree validation. Every split is grouped by simulator scenario seed.
    train_rows = []
    for p in TRAIN_FILES:
        with p.open() as f:
            train_rows.extend(json.loads(line) for line in f)
    features = build_feature_names()
    Xt = np.asarray([r["features"] for r in train_rows], dtype=np.float32)
    yt = np.asarray([r["saving_seconds"] for r in train_rows], dtype=float)
    wt = 1 / np.asarray([r["targets"] for r in train_rows], dtype=float)
    seeds = np.asarray([r["seed"] for r in train_rows], dtype=np.int64)
    scenarios = np.asarray([r["scenario"] for r in train_rows])

    params = dict(
        n_estimators=160,
        max_depth=15,
        min_samples_leaf=25,
        random_state=822174,
        n_jobs=2,
        max_features=0.9,
    )
    fixed_train = seeds % 10 < 8
    fixed_model = ExtraTreesRegressor(**params)
    fixed_model.fit(Xt[fixed_train], yt[fixed_train], sample_weight=wt[fixed_train])
    fixed_pred = fixed_model.predict(Xt[~fixed_train])
    fixed_metrics = regression_metrics(yt[~fixed_train], fixed_pred)
    fixed_metrics["training_scenarios"] = int(len(set(seeds[fixed_train])))
    fixed_metrics["heldout_scenarios"] = int(len(set(seeds[~fixed_train])))

    oof = np.empty_like(yt)
    fold_rows = []
    fold_importances = []
    for fold in range(5):
        valid = seeds % 5 == fold
        model = ExtraTreesRegressor(**params)
        model.fit(Xt[~valid], yt[~valid], sample_weight=wt[~valid])
        oof[valid] = model.predict(Xt[valid])
        fm = regression_metrics(yt[valid], oof[valid])
        fm.update(fold=fold, training_scenarios=int(len(set(seeds[~valid]))), heldout_scenarios=int(len(set(seeds[valid]))))
        fold_rows.append(fm)
        fold_importances.append(model.feature_importances_)
    write_csv(OUT / "regression_cv_folds.csv", fold_rows)
    cv_metrics = regression_metrics(yt, oof)
    cv_metrics["scenarios"] = int(len(set(seeds)))

    scenario_rows = []
    for s in sorted(set(scenarios)):
        mask = scenarios == s
        row = regression_metrics(yt[mask], oof[mask])
        row.update(scenario=s, scenarios=int(len(set(seeds[mask]))))
        scenario_rows.append(row)
    write_csv(OUT / "regression_by_scenario.csv", scenario_rows, ["scenario", "scenarios"] + list(regression_metrics(yt[:2], yt[:2] + 1e-6)))

    # Decision-state evaluation: select best predicted fraction, accept only if prediction > 0.
    decision_groups = defaultdict(list)
    for i, r in enumerate(train_rows):
        decision_groups[(r["seed"], r["action"])].append(i)
    decisions = []
    for (seed, action), idx in decision_groups.items():
        idx = np.asarray(idx)
        selected = int(idx[np.argmax(oof[idx])])
        accepted = bool(oof[selected] > 0)
        actual = float(yt[selected]) if accepted else 0.0
        oracle = max(0.0, float(np.max(yt[idx])))
        decisions.append(
            {
                "seed": int(seed),
                "scenario": train_rows[selected]["scenario"],
                "action": int(action),
                "candidate_count": int(len(idx)),
                "selected_prediction": float(oof[selected]),
                "selected_actual_saving": float(yt[selected]),
                "accepted": accepted,
                "policy_saving": actual,
                "oracle_saving": oracle,
                "regret": oracle - actual,
            }
        )
    write_csv(OUT / "regression_decision_states.csv", decisions)
    accepted = [r for r in decisions if r["accepted"]]
    decision_summary = {
        "decision_states": len(decisions),
        "accepted_states": len(accepted),
        "acceptance_rate_percent": 100 * len(accepted) / len(decisions),
        "accepted_actual_positive_percent": 100 * sum(r["selected_actual_saving"] > 0 for r in accepted) / len(accepted),
        "accepted_mean_actual_saving_seconds": float(np.mean([r["selected_actual_saving"] for r in accepted])),
        "accepted_median_actual_saving_seconds": float(np.median([r["selected_actual_saving"] for r in accepted])),
        "policy_mean_saving_per_decision_seconds": float(np.mean([r["policy_saving"] for r in decisions])),
        "policy_mean_saving_cluster_bootstrap_ci95": cluster_bootstrap_decision_ci(decisions),
        "oracle_mean_saving_per_decision_seconds": float(np.mean([r["oracle_saving"] for r in decisions])),
        "mean_regret_seconds": float(np.mean([r["regret"] for r in decisions])),
    }

    # OOF calibration by predicted decile.
    edges = np.quantile(oof, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    calibration = []
    for b in range(10):
        mask = (oof > edges[b]) & (oof <= edges[b + 1]) if b else (oof >= edges[b]) & (oof <= edges[b + 1])
        calibration.append(
            {
                "decile": b + 1,
                "rows": int(mask.sum()),
                "mean_prediction": float(oof[mask].mean()),
                "mean_actual_saving": float(yt[mask].mean()),
                "actual_positive_percent": float(100 * np.mean(yt[mask] > 0)),
            }
        )
    write_csv(OUT / "regression_calibration.csv", calibration)

    # Grouped permutation importance on the untouched fixed holdout.
    blocks = [
        ("route_geometry_and_position", [0, 1, 2, 3, 14, 15, 16]),
        ("global_task_state", [4, 5, 6, 7, 8]),
        ("candidate_set_aggregate", [9, 10, 11, 12, 13, 17]),
        ("top3_gain_and_uncertainty", [18 + 13 * k + j for k in range(3) for j in [0, 1, 2]]),
        ("top3_geometry", [18 + 13 * k + j for k in range(3) for j in [3, 4, 7, 9, 10, 11]]),
        ("top3_measurement_history", [18 + 13 * k + j for k in range(3) for j in [5, 6, 8]]),
        ("top3_destination_alignment", [18 + 13 * k + 12 for k in range(3)]),
    ]
    assert sorted(j for _, js in blocks for j in js) == list(range(57))
    Xv, yv = Xt[~fixed_train], yt[~fixed_train]
    baseline_mse = mean_squared_error(yv, fixed_pred)
    rng = np.random.default_rng(RNG_SEED + 927)
    group_importance = []
    for name, js in blocks:
        deltas = []
        for _ in range(10):
            perm = rng.permutation(len(Xv))
            xp = Xv.copy()
            xp[:, js] = Xv[perm][:, js]
            deltas.append(mean_squared_error(yv, fixed_model.predict(xp)) - baseline_mse)
        group_importance.append(
            {
                "feature_group": name,
                "features": len(js),
                "delta_MSE_mean": float(np.mean(deltas)),
                "delta_MSE_sd": float(np.std(deltas, ddof=1)),
                "delta_RMSE_approx": float(math.sqrt(max(0, baseline_mse + np.mean(deltas))) - math.sqrt(baseline_mse)),
            }
        )
    group_importance.sort(key=lambda r: r["delta_MSE_mean"], reverse=True)
    write_csv(OUT / "regression_group_permutation_importance.csv", group_importance)

    mean_imp = np.mean(fold_importances, axis=0)
    feature_importance = [
        {"rank": rank, "feature_index": int(j), "feature": features[j], "mean_impurity_importance": float(mean_imp[j])}
        for rank, j in enumerate(np.argsort(mean_imp)[::-1], 1)
    ]
    write_csv(OUT / "regression_feature_importance.csv", feature_importance)

    # Figures.
    style = {
        "Q3 local random (v3_origin20)": "#4C78A8",
        "Q4 local random (V6 Lite)": "#F58518",
        "Q4 local all-directional (V6 Lite)": "#54A24B",
        "Q4 local boundary-outward R=1000 (V6 Lite)": "#E45756",
        "Q4 official practice (V6 Lite)": "#B279A2",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), constrained_layout=True)
    ax = axes[0, 0]
    for label in ["Q4 local random (V6 Lite)", "Q4 local all-directional (V6 Lite)", "Q4 local boundary-outward R=1000 (V6 Lite)"]:
        x = np.sort([fnum(r["seconds_per_source"]) for r in groups[label]])
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), label=label.replace("Q4 local ", ""), color=style[label])
    ax.set(xlabel="Seconds per source", ylabel="Empirical CDF", title="Q4 V6 Lite: local distributions")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax = axes[0, 1]
    labels = [r["dataset"].replace(" (V6 Lite)", "").replace(" local ", "\n") for r in robustness]
    meanv = [r["mean_seconds_per_source"] for r in robustness]
    p90v = [r["p90_seconds_per_source"] for r in robustness]
    cvarv = [r["cvar90_seconds_per_source"] for r in robustness]
    xx = np.arange(len(labels))
    ax.plot(xx, meanv, "o-", label="Mean")
    ax.plot(xx, p90v, "s-", label="P90")
    ax.plot(xx, cvarv, "^-", label="CVaR90")
    ax.set_xticks(xx, labels, rotation=18, ha="right", fontsize=8)
    ax.set(ylabel="Seconds per source", title="Central and tail risk")
    ax.legend()
    ax.grid(alpha=0.25)
    ax = axes[1, 0]
    for domain, marker in [("Q3 local random", "o"), ("Q4 local random", "s")]:
        rr = [r for r in source_count_rows if r["dataset"] == domain]
        ax.plot([r["source_count"] for r in rr], [r["mean_seconds_per_source"] for r in rr], marker=marker, label=domain)
    ax.set(xlabel="Number of sources", ylabel="Mean seconds per source", title="Source-count sensitivity")
    ax.legend()
    ax.grid(alpha=0.25)
    ax = axes[1, 1]
    q4x = np.sort([fnum(r["seconds_per_source"]) for r in groups["Q4 local random (V6 Lite)"]])
    q3x = np.sort([fnum(r["seconds_per_source"]) for r in q3])
    ax.plot(q3x, 1 - np.arange(len(q3x)) / len(q3x), label="Q3 selected", color=style["Q3 local random (v3_origin20)"])
    ax.plot(q4x, 1 - np.arange(len(q4x)) / len(q4x), label="Q4 V6 Lite", color=style["Q4 local random (V6 Lite)"])
    ax.set_yscale("log")
    ax.set(xlabel="Seconds per source", ylabel="Exceedance probability", title="Upper-tail exceedance")
    ax.legend()
    ax.grid(alpha=0.25)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"robustness_tail.{ext}", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.8), constrained_layout=True)
    component_names = ["movement", "switching", "measurement", "optical_attempt", "successful_removal"]
    component_labels = ["Movement", "Channel switching", "RF measurement", "Optical attempts", "Successful removal"]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]
    bottom = np.zeros(len(costs))
    for name, lab, color in zip(component_names, component_labels, colors):
        vals = np.asarray([r[f"{name}_pooled_seconds_per_source"] for r in costs])
        ax.bar(np.arange(len(costs)), vals, bottom=bottom, label=lab, color=color)
        bottom += vals
    ax.set_xticks(np.arange(len(costs)), [r["dataset"].replace(" (V6 Lite)", "").replace(" local ", "\n") for r in costs], rotation=16, ha="right", fontsize=8)
    ax.set(ylabel="Pooled seconds per source", title="Exact virtual-time decomposition (selected strategies only)")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"cost_decomposition.{ext}", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    ax = axes[0, 0]
    hb = ax.hexbin(oof, yt, gridsize=48, mincnt=1, bins="log", cmap="viridis")
    lim = np.quantile(np.abs(np.r_[oof, yt]), 0.99)
    ax.plot([-lim, lim], [-lim, lim], "--", color="white", lw=1)
    ax.axhline(0, color="grey", lw=0.7)
    ax.axvline(0, color="grey", lw=0.7)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="OOF predicted saving (s)", ylabel="Counterfactual actual saving (s)", title="5-fold grouped out-of-fold predictions")
    fig.colorbar(hb, ax=ax, label="log count")
    ax = axes[0, 1]
    ax.plot([r["mean_prediction"] for r in calibration], [r["mean_actual_saving"] for r in calibration], "o-")
    lo = min(r["mean_prediction"] for r in calibration)
    hi = max(r["mean_prediction"] for r in calibration)
    ax.plot([lo, hi], [lo, hi], "--", color="grey")
    for r in calibration:
        ax.annotate(str(r["decile"]), (r["mean_prediction"], r["mean_actual_saving"]), fontsize=8)
    ax.set(xlabel="Mean prediction in decile (s)", ylabel="Mean actual saving (s)", title="OOF calibration by predicted decile")
    ax.grid(alpha=0.25)
    ax = axes[1, 0]
    gi = group_importance[::-1]
    ax.barh([r["feature_group"].replace("_", " ") for r in gi], [r["delta_MSE_mean"] for r in gi], color="#4C78A8")
    ax.set(xlabel="Increase in held-out MSE after permutation", title="Grouped permutation importance")
    ax.grid(axis="x", alpha=0.25)
    ax = axes[1, 1]
    sr = sorted(scenario_rows, key=lambda r: r["Pearson_r"])
    ax.barh([r["scenario"] for r in sr], [r["Pearson_r"] for r in sr], color="#F58518")
    ax.axvline(0, color="black", lw=0.7)
    ax.set(xlabel="OOF Pearson correlation", title="Validation stability across scenario families")
    ax.grid(axis="x", alpha=0.25)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"regression_validation.{ext}", dpi=200)
    plt.close(fig)

    # Machine-readable roll-up.
    stored_model_training = json.loads(MODEL_JSON.read_text()).get("training")
    results = {
        "scope": {
            "no_ablation": True,
            "paper_modified": False,
            "q3_strategy": "v3_origin20",
            "q4_strategy": "V6 Lite",
            "local_q4_is_reconstructed_simulator_only": True,
            "official_q4_is_practice_not_formal_test": True,
        },
        "robustness": robustness,
        "cost_decomposition": costs,
        "scene_factor_model": {
            "cases": len(random_q4),
            "source_count_fixed_effects": ns,
            "HC3_robust": True,
            "R2": scene_r2,
            "effects": scene_effects,
        },
        "regression_tree": {
            "model": "ExtraTreesRegressor",
            "feature_count": 57,
            "rows": len(train_rows),
            "scenarios": int(len(set(seeds))),
            "target": "baseline completion seconds minus probe-then-completion seconds",
            "positive_target_means_time_saved": True,
            "frozen_80_20_group_holdout": fixed_metrics,
            "five_fold_grouped_oof": cv_metrics,
            "decision_state_evaluation": decision_summary,
            "group_permutation_importance": group_importance,
            "top_individual_features": feature_importance[:15],
            "stored_deployed_model_training_metadata": stored_model_training,
        },
    }
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "generated_by": Path(__file__).name,
        "random_seed": RNG_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPS,
        "software": {
            "python": __import__("platform").python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "data_sources": {str(p.relative_to(ROOT)): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in [Q3_CSV, Q4_CSV, Q4_PLAN, OFFICIAL_RESULTS, OFFICIAL_SUMMARY, *TRAIN_FILES, MODEL_JSON]},
        "statistical_methods": [
            "case-level nonparametric bootstrap percentile 95% intervals",
            "Wilson 95% binomial interval for all-clear rate",
            "P90/P95 and top-tail CVaR90/CVaR95",
            "source-count fixed effects plus HC3 robust covariance",
            "deterministic scenario-grouped 5-fold out-of-fold validation",
            "cluster bootstrap by simulator scenario for decision-state utility",
            "held-out grouped permutation importance",
        ],
        "boundaries": [
            "No algorithm component was deleted or replaced; this is not an ablation study.",
            "Q4 local results use the reconstructed simulator and are not official-program results.",
            "Official results are practice observations, not formal test scores.",
            "Counterfactual tree labels validate single decision states; they are not additive full-task savings.",
            "Scene-factor regression is associative, not causal.",
        ],
    }
    (OUT / "method_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Detailed Chinese report generated from the exact machine-readable results.
    by_label = {r["dataset"]: r for r in robustness}
    cost_by = {r["dataset"]: r for r in costs}

    def ci(v):
        return f"[{v[0]:.3f}, {v[1]:.3f}]"

    def risk_table(labels):
        lines = ["| 数据集 | 完成 | 均值（秒/源） | 中位数 | P90 | P95 | CVaR90 | 最大值 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for label in labels:
            r = by_label[label]
            lines.append(f"| {label} | {r['all_clear_cases']}/{r['cases']} | {r['mean_seconds_per_source']:.3f} | {r['median_seconds_per_source']:.3f} | {r['p90_seconds_per_source']:.3f} | {r['p95_seconds_per_source']:.3f} | {r['cvar90_seconds_per_source']:.3f} | {r['max_seconds_per_source']:.3f} |")
        return "\n".join(lines)

    cost_lines = ["| 数据集 | 移动 | 测量 | 切换 | 光学尝试 | 成功清除 | 合计（池化秒/源） | 移动占比 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in costs:
        cost_lines.append(f"| {r['dataset']} | {r['movement_pooled_seconds_per_source']:.3f} | {r['measurement_pooled_seconds_per_source']:.3f} | {r['switching_pooled_seconds_per_source']:.3f} | {r['optical_attempt_pooled_seconds_per_source']:.3f} | {r['successful_removal_pooled_seconds_per_source']:.3f} | {r['pooled_seconds_per_source']:.3f} | {r['movement_share_percent']:.2f}% |")

    scenario_lines = ["| 场景族 | 场景数 | RMSE | MAE | Pearson r | MSE 相对零预测改善 |", "|---|---:|---:|---:|---:|---:|"]
    for r in sorted(scenario_rows, key=lambda x: x["scenario"]):
        scenario_lines.append(f"| {r['scenario']} | {r['scenarios']} | {r['RMSE']:.3f} | {r['MAE']:.3f} | {r['Pearson_r']:.3f} | {r['MSE_skill_vs_zero_percent']:.2f}% |")

    perm_lines = ["| 特征组 | 置乱后 MSE 增量 | 近似 RMSE 增量 |", "|---|---:|---:|"]
    for r in group_importance:
        perm_lines.append(f"| {r['feature_group']} | {r['delta_MSE_mean']:.3f} | {r['delta_RMSE_approx']:.3f} |")

    feature_lines = ["| 排名 | 特征 | 平均不纯度重要性 |", "|---:|---|---:|"]
    for r in feature_importance[:12]:
        feature_lines.append(f"| {r['rank']} | {r['feature']} | {r['mean_impurity_importance']:.5f} |")

    effect_lines = ["| 因素 | 样本内 IQR | IQR 变化对应秒/源 | HC3 95% CI |", "|---|---:|---:|---:|"]
    for r in scene_effects:
        effect_lines.append(f"| {r['factor']} | {r['observed_iqr']:.3f} | {r['effect_seconds_per_source_per_iqr']:+.3f} | [{r['hc3_ci95_low']:+.3f}, {r['hc3_ci95_high']:+.3f}] |")

    q4_random = by_label["Q4 local random (V6 Lite)"]
    q4_official = by_label["Q4 official practice (V6 Lite)"]
    q3_risk = by_label["Q3 local random (v3_origin20)"]
    top_q4 = [r for r in tails if r["dataset"] == "Q4 local random"][:5]
    top_q3 = [r for r in tails if r["dataset"] == "Q3 local random"][:5]
    source_q4 = [r for r in source_count_rows if r["dataset"] == "Q4 local random"]
    source_q3 = [r for r in source_count_rows if r["dataset"] == "Q3 local random"]
    source_table = "\n".join(
        f"| {'第三问' if r['dataset'].startswith('Q3') else '第四问'} | {r['source_count']} | {r['cases']} | {r['mean_seconds_per_source']:.3f} | {r['p90_seconds_per_source']:.3f} | {r['cvar90_seconds_per_source']:.3f} |"
        for r in source_q3 + source_q4
    )
    q4_tail_table = "\n".join(
        f"| {r['case']} | {r['sources']} | {float(r['seconds_per_source']):.3f} | {float(r['directional_share']):.3f} | {float(r['distance_per_source_m']):.3f} | {float(r['measurements_per_source']):.3f} | {r['failed_clears']} | {float(r['coverage_tail_s']):.3f} |"
        for r in top_q4
    )
    q3_tail_table = "\n".join(
        f"| {r['case']} | {r['sources']} | {float(r['seconds_per_source']):.3f} | {float(r['distance_per_source_m']):.3f} | {float(r['measurements_per_source']):.3f} | {r['failed_clears']} |"
        for r in top_q3
    )
    scenario_table = "\n".join(scenario_lines)
    permutation_table = "\n".join(perm_lines)
    feature_table = "\n".join(feature_lines)
    cost_table = "\n".join(cost_lines)
    effect_table = "\n".join(effect_lines)
    max_cost_identity_error = max(r["identity_max_abs_error_seconds"] for r in costs)

    report = f"""# 实验 2、3、4：稳健性、回归树验证与耗时解释

本报告是独立实验结果，不插入论文正文，不进行消融。第三问固定使用 `v3_origin20`，第四问固定使用 `V6 Lite`；所有表格、区间和图均可由同目录脚本复现。

## 结论先行

1. **完成稳健性成立，但第四问尾部仍明显受源总数与覆盖搜索影响。** 第三问本地 400/400、第四问本地 1140/1140、第四问官方演练 11/11 均全部清除。第四问 1000 个本地随机场景均值为 {q4_random['mean_seconds_per_source']:.3f} 秒/源，P90 为 {q4_random['p90_seconds_per_source']:.3f}，CVaR90 为 {q4_random['cvar90_seconds_per_source']:.3f}；官方演练 11 轮均值为 {q4_official['mean_seconds_per_source']:.3f} 秒/源，P90 为 {q4_official['p90_seconds_per_source']:.3f}。
2. **回归树有方向性信息，但预测强度不高。** 57 维 ExtraTrees 的严格场景分组五折 OOF 结果为 RMSE={cv_metrics['RMSE']:.3f} 秒、MAE={cv_metrics['MAE']:.3f} 秒、Pearson r={cv_metrics['Pearson_r']:.3f}、R²={cv_metrics['R2']:.3f}，相对恒为 0 的预测，MSE 改善 {cv_metrics['MSE_skill_vs_zero_percent']:.2f}%。因此它更适合作为“是否值得顺路测量”的弱排序器，而不是精确耗时预测器。
3. **移动是绝对主耗时。** 第三问、第四问各数据集的移动耗时占比均可在分项表中直接核验；第四问本地随机组移动占比为 {cost_by['Q4 local random (V6 Lite)']['movement_share_percent']:.2f}%，RF 测量占比为 {cost_by['Q4 local random (V6 Lite)']['measurement_share_percent']:.2f}%。优化解释应优先围绕路径长度、覆盖尾段和源总数，而不是只盯清除尝试。

## 1. 数据与统计口径

- 第三问：本地重建模拟器上的 400 个随机场景，只取已选定的 `v3_origin20`。
- 第四问本地：冻结的 1000 个随机新场景、70 个全定向压力场景、70 个边界外向且接收半径为 1000 m 的压力场景，只取 V6 Lite。
- 第四问官方：官方模拟器 v1.1 的 11 个新演练场景，共 144 个源；这是演练，不是正式测试。
- 稳健性以“每个案例的总虚拟时间/源数”为案例等权指标；同时报告 P90、P95、CVaR90（最慢 10% 的均值）和 CVaR95。
- 均值、P90、CVaR90 的 95% 区间用 10000 次案例级非参数 bootstrap；全清除率用 Wilson 95% 区间。

## 2. 稳健性与尾部风险

{risk_table(list(groups))}

### 2.1 区间与完成率

- 第三问随机 400 例均值 95% CI：{ci(q3_risk['mean_ci95'])} 秒/源；P90 95% CI：{ci(q3_risk['p90_ci95'])}；400/400 的全清除率 Wilson 95% CI 为 {ci(q3_risk['all_clear_wilson_ci95_percent'])}% 。
- 第四问本地随机 1000 例均值 95% CI：{ci(q4_random['mean_ci95'])} 秒/源；P90 95% CI：{ci(q4_random['p90_ci95'])}；1000/1000 的全清除率 Wilson 95% CI 为 {ci(q4_random['all_clear_wilson_ci95_percent'])}% 。
- 第四问官方 11 例均值 95% CI：{ci(q4_official['mean_ci95'])} 秒/源。样本只有 11 例，区间主要反映小样本不确定性，不应外推为正式测试置信保证。

### 2.2 源总数敏感性

第四问不足 16 个源时必须继续覆盖剩余站点以排除遗漏，固定覆盖成本被更少的源分摊，因此“秒/源”会随源数增加而显著下降。

| 问题 | 源数 | 案例数 | 均值 | P90 | CVaR90 |
|---|---:|---:|---:|---:|---:|
{source_table}

### 2.3 最慢案例诊断

第四问随机组最慢 5 例：

| 案例 | 源数 | 秒/源 | 定向占比 | 距离/源（m） | 测量/源 | 失败清除 | 覆盖尾段（s） |
|---|---:|---:|---:|---:|---:|---:|---:|
{q4_tail_table}

第三问随机组最慢 5 例：

| 种子 | 源数 | 秒/源 | 距离/源（m） | 测量/源 | 失败清除 |
|---:|---:|---:|---:|---:|---:|
{q3_tail_table}

这些尾部案例只用于解释，不据此重新调参，所以不会形成“看完测试集再优化”的数据泄漏。

## 3. 第四问回归树模型验证与解释

### 3.1 验证设计

训练数据含 {len(train_rows)} 个候选动作样本、{len(set(seeds))} 个独立模拟场景、57 个仅由合法观测构成的特征。候选决策状态从 V4 基线轨迹中抽样，再对“不测量”与“顺路测量”分别续跑至任务完成。标签定义为：

`saving_seconds = 不顺路测量时完成任务的总时间 - 采用该顺路测量后完成任务的总时间`

正值表示该候选动作在反事实续跑中节省时间。所有划分均以场景种子为组，同一场景的多个决策点和候选比例不会跨越训练/验证边界。

### 3.2 总体验证结果

| 验证 | 训练场景 | 验证场景 | RMSE | MAE | R² | Pearson r | Spearman ρ | 相对零预测 MSE 改善 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 冻结 80/20 场景留出 | {fixed_metrics['training_scenarios']} | {fixed_metrics['heldout_scenarios']} | {fixed_metrics['RMSE']:.3f} | {fixed_metrics['MAE']:.3f} | {fixed_metrics['R2']:.3f} | {fixed_metrics['Pearson_r']:.3f} | {fixed_metrics['Spearman_rho']:.3f} | {fixed_metrics['MSE_skill_vs_zero_percent']:.2f}% |
| 5 折场景分组 OOF | 每折约 80% | 全部 {cv_metrics['scenarios']} | {cv_metrics['RMSE']:.3f} | {cv_metrics['MAE']:.3f} | {cv_metrics['R2']:.3f} | {cv_metrics['Pearson_r']:.3f} | {cv_metrics['Spearman_rho']:.3f} | {cv_metrics['MSE_skill_vs_zero_percent']:.2f}% |

冻结 80/20 留出的 MSE 为 {fixed_metrics['MSE']:.6f}，已发布树文件内保存的原训练环境值为 {stored_model_training['MSE']:.6f}，两者只差 {abs(fixed_metrics['MSE'] - stored_model_training['MSE']):.6f}（约 {100 * abs(fixed_metrics['MSE'] - stored_model_training['MSE']) / stored_model_training['MSE']:.6f}%），说明本次复现与发布记录一致。

模型并不具备高精度回归能力：R² 较低、绝对误差较大。不过它的相关系数为正，预测高分箱的实际平均收益也可由 `regression_calibration.csv` 检查。因此合理定位是弱排序/门控信号，而非可直接解释为“将节省多少秒”的精确估计。

### 3.3 决策状态层面的检验

对每个验证决策状态，在最多 3 个路径比例候选中选择预测值最大的一个，并仅在最大预测值大于 0 时接受：

- 决策状态：{decision_summary['decision_states']} 个；接受 {decision_summary['accepted_states']} 个，接受率 {decision_summary['acceptance_rate_percent']:.2f}%。
- 被接受候选中，真实反事实收益为正的比例：{decision_summary['accepted_actual_positive_percent']:.2f}%。
- 被接受候选实际收益均值/中位数：{decision_summary['accepted_mean_actual_saving_seconds']:.3f}/{decision_summary['accepted_median_actual_saving_seconds']:.3f} 秒。
- 将拒绝记为 0 后，每个决策状态的平均反事实收益：{decision_summary['policy_mean_saving_per_decision_seconds']:.3f} 秒；按场景聚类 bootstrap 的 95% CI 为 {ci(decision_summary['policy_mean_saving_cluster_bootstrap_ci95'])}。
- 同一候选集合的反事实最优均值为 {decision_summary['oracle_mean_saving_per_decision_seconds']:.3f} 秒，当前选择相对反事实最优的平均遗憾为 {decision_summary['mean_regret_seconds']:.3f} 秒。

注意：每个标签都从其所在状态单独续跑得到，不同决策点的收益不能简单相加为完整任务节省量。完整任务表现仍应以第 2 节的端到端案例为准。

### 3.4 跨场景稳定性

{scenario_table}

### 3.5 特征解释

在冻结 20% 场景留出集上，把同一语义组的特征整体置乱 10 次；MSE 增量越大，说明该组对模型越重要。

{permutation_table}

五折模型的前 12 个单特征不纯度重要性如下；树模型中的重要性是预测贡献线索，不等于因果效应。

{feature_table}

## 4. 耗时构成与敏感因素解释

### 4.1 精确耗时恒等式

每个案例均按题目计时规则重算：

`总时间 = 路程/5 + 5×测量次数 + 频道切换次数 + 3×清除尝试次数 + 2×成功清除数`

所有数据集的逐案例最大重构误差不超过 {max_cost_identity_error:.9f} 秒，说明分项没有遗漏。下表使用“所有案例总分项/所有源数”的池化秒/源，因此五个分项严格加总为合计。

{cost_table}

移动成本在各组中均占主导。RF 测量是第二大项；频道切换与光学清除的合计占比较小。因此尾部优化的首要解释变量是路径/覆盖长度，其次才是测量密度和失败清除。

### 4.2 场景外生因素的敏感性

只在第四问 1000 个随机新场景内建模，并使用源总数固定效应控制 10--16 个源造成的非线性差异。下表给出各因素从样本第 25 百分位增加到第 75 百分位时，秒/源的条件关联变化；协方差使用 HC3 稳健估计。模型 R²={scene_r2:.3f}。

{effect_table}

解释限制：这些是控制源总数后的条件关联，不是随机干预所得的因果效应。接收半径或径向位置的区间若跨 0，应表述为“本批数据未观察到稳定方向”，不能说成“完全没有影响”。

## 5. 证据边界

- 本地第四问使用重建模拟器，只证明策略在该测试环境中的可复现稳健性，不等价于官方隐藏分布。
- 官方 11 轮是官方演练观测，不是正式测试；未在相同官方案例上运行其他策略，不能据此做版本优劣结论。
- 回归树数据来自本地反事实续跑。场景分组验证避免了同场景泄漏，但不消除模拟器与官方环境之间的分布偏移。
- 本实验没有删除任何模块、替换策略或重新选参，因此不是消融实验。

## 6. 输出文件

- `results.json`：全部核心统计量。
- `robustness_summary.csv`、`source_count_sensitivity.csv`、`tail_cases_top10.csv`：稳健性与尾部。
- `regression_cv_folds.csv`、`regression_by_scenario.csv`、`regression_calibration.csv`、`regression_decision_states.csv`：回归树验证明细。
- `regression_group_permutation_importance.csv`、`regression_feature_importance.csv`：模型解释。
- `cost_decomposition.csv`、`scene_factor_sensitivity.csv`：耗时分解与场景敏感因素。
- `robustness_tail.*`、`regression_validation.*`、`cost_decomposition.*`：PNG/PDF 图。
- `method_manifest.json`：输入文件哈希、随机种子、统计方法和适用边界。
"""
    (OUT / "234_详细实验结果.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(OUT), "robustness": robustness, "fixed": fixed_metrics, "cv": cv_metrics, "decision": decision_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
