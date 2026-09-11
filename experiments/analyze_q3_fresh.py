"""Build portable tables and figures for the first GPT Pro handoff."""

import csv
import gzip
import json
import math
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/q3_fresh"
OUT = ROOT / "research/q3_fresh_round1"


def paired(rows, a, b, group_episodes=False):
    first = {r["case_id"]: r for r in rows if r["strategy"] == a}
    second = {r["case_id"]: r for r in rows if r["strategy"] == b}
    differences = {k: second[k]["virtual_time_s"] - v["virtual_time_s"] for k, v in first.items()}
    sample = list(differences.values())
    if group_episodes:
        groups = {}
        for k, d in differences.items():
            groups.setdefault(k.rsplit("-radius-", 1)[0], []).append(d)
        sample = [statistics.mean(v) for v in groups.values()]
    rng = random.Random(940000)
    means = sorted(statistics.mean(rng.choices(sample, k=len(sample))) for _ in range(5000))
    return dict(mean_delta_s=statistics.mean(sample), descriptive_bootstrap_95_s=[means[125], means[4874]],
                resampling_units=len(sample), wins=sum(d < 0 for d in sample),
                losses=sum(d > 0 for d in sample), max_regression_s=max(sample),
                worst_case=max(differences, key=differences.get),
                best_case=min(differences, key=differences.get))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    docs = {name: json.loads((DATA / name / "results.json").read_text(encoding="utf-8"))
            for name in ("development", "scheduling", "holdout", "validation")}
    audit, analysis = {}, {}
    for name, doc in docs.items():
        assert doc["manifest"]["status"] == "completed" and doc["manifest"]["source_unchanged"]
        assert all(r["error"] is None and r["all_cleared"] and r["completion_certified"] for r in doc["rows"])
        audit[name] = dict(rows=len(doc["rows"]), certified_full_clear=sum(r["all_cleared"] for r in doc["rows"]))
        checked = 0
        for file in sorted((DATA/name).glob("trace-*.json.gz")):
            traces=json.loads(gzip.open(file,"rt",encoding="utf-8").read())
            for trace in traces.values():
                p, channel, total = (0,0), 1, 0.0
                seen, removed = set(), set()
                sources={s["channel"]: s for s in trace["scenario"]["sources"]}
                for action in trace["search"]["action_history"]:
                    q, c = action["position"], action["channel"]
                    total += math.dist(p,q)/5
                    source=sources.get(c) if c not in removed else None
                    d=math.inf if source is None else math.dist(q,(source["x"],source["y"]))
                    if action["action"] == "measure":
                        assert (c,tuple(q)) not in seen
                        seen.add((c,tuple(q)))
                        total += 5 + (c != channel)
                        channel = c
                        if action["result"] == "direction":
                            assert 5 < d <= source["reception_radius_m"] + 1e-7
                            bearing=math.degrees(math.atan2(source["y"]-q[1],source["x"]-q[0]))%360
                            assert abs((bearing-action["bearing_deg"]+180)%360-180) <= 1.005 + 1e-7
                        elif action["result"] == "near":
                            assert d <= 5 + 1e-7
                        else:
                            assert source is None or d > source["reception_radius_m"] - 1e-7
                    else:
                        assert action["result"] == "success" and d <= 20 + 1e-7
                        total += 5
                        removed.add(c)
                    assert abs(total-action["virtual_time_s"]) < 1e-3
                    p=q
                assert removed == set(sources)
                assert abs(total-trace["evaluation"]["virtual_time_s"]) < 1e-3
                checked += 1
        assert checked == len(doc["rows"])
        audit[name]["independently_audited_traces"] = checked
        analysis[name] = dict(summary=doc["summary"], comparisons={})
        for a, b in (("v0", "v1"), ("v0", "v1_selective"), ("v1", "v1_selective")):
            if b in doc["summary"]:
                analysis[name]["comparisons"][f"{b}_minus_{a}"] = paired(doc["rows"], a, b, name == "validation")
        analysis[name]["worst_ratios_v1b"] = sorted(
            [r for r in doc["rows"] if r["strategy"] == "v1_selective"],
            key=lambda r: r["time_over_physical_lower_bound"], reverse=True)[:5]
    for doc in docs.values():
        for r in doc["rows"]:
            assert abs(sum(r["time_breakdown_s"].values()) - r["virtual_time_s"]) < 1e-4
            assert r["virtual_time_s"] >= r["certified_lower_bound_s"]
    validation = docs["validation"]["rows"]
    analysis["validation"]["robust_summary"] = {
        name: dict(mean_observation_robust_lb_s=statistics.mean(r["observation_robust_physical_lower_bound_s"] for r in validation if r["strategy"] == name),
                   ratio=statistics.mean(r["virtual_time_s"] for r in validation if r["strategy"] == name) /
                         statistics.mean(r["observation_robust_physical_lower_bound_s"] for r in validation if r["strategy"] == name))
        for name in docs["validation"]["summary"]}
    for suffix in ("low", "high"):
        subset = [r for r in validation if r["case_id"].endswith(suffix)]
        analysis["validation"]["radius_"+suffix] = {name: dict(
            mean_s=statistics.mean(r["virtual_time_s"] for r in subset if r["strategy"] == name),
            mean_lb_s=statistics.mean(r["physical_lower_bound_s"] for r in subset if r["strategy"] == name),
            ratio=statistics.mean(r["virtual_time_s"] for r in subset if r["strategy"] == name) /
                  statistics.mean(r["physical_lower_bound_s"] for r in subset if r["strategy"] == name),
            mean_case_ratio=statistics.mean(r["time_over_physical_lower_bound"] for r in subset if r["strategy"] == name))
            for name in docs["validation"]["summary"]}
    (OUT / "analysis.json").write_text(json.dumps(dict(audit=audit, analysis=analysis), indent=2), encoding="utf-8")
    columns = ["batch", "case_id", "strategy", "source_total", "all_cleared", "completion_certified",
               "virtual_time_s", "physical_lower_bound_s", "time_over_physical_lower_bound",
               "certified_lower_bound_s", "time_over_certified_lower_bound", "measurement_count",
               "failed_clear_count", "confirmation_tail_s", "policy_wall_s"]
    with (OUT / "all_runs.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for name, doc in docs.items():
            writer.writerows(dict(batch=name, **r) for r in doc["rows"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 160, "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    labels = ["V0", "V1", "V1b"]
    names = ["v0", "v1", "v1_selective"]
    colors = ["#405b85", "#dc9944", "#348b76"]
    for ax, batch, title in zip(axes, ["holdout", "validation"], ["32 new synthetic cases", "12 practice episodes, 2 reconstructions each"]):
        s = docs[batch]["summary"]
        bottom = [0.,0.,0.]
        for key, label, color in [("movement_s","Movement",colors[0]), ("detection_s","Measurements",colors[1]),
                                  ("switching_s","Switching",colors[2]), ("clear_s","Successful clearance","#8794a6")]:
            vals = [s[n]["mean_components"][key] if key != "clear_s" else
                    s[n]["mean_components"]["optical_s"]+s[n]["mean_components"]["removal_s"] for n in names]
            ax.bar(labels,vals,bottom=bottom,label=label,color=color,width=.58)
            bottom=[a+b for a,b in zip(bottom,vals)]
        lb=(s["v0"]["mean_physical_lb_s"] if batch == "holdout" else
            analysis["validation"]["robust_summary"]["v0"]["mean_observation_robust_lb_s"])
        ax.axhline(lb,color="#ae4050",linestyle="--",label=f"Physical lower bound: {lb:.1f}s")
        for i,n in enumerate(names):
            ax.text(i,bottom[i]+60,f'{bottom[i]:.1f}s\n{bottom[i]/lb:.3f}x LB',ha="center",fontsize=9)
        ax.set(ylim=(0,5100),ylabel="Mean total time (seconds)",title=title)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].legend(loc="upper right",fontsize=8)
    fig.savefig(OUT / "time_breakdown.png")
    plt.close(fig)
    # Every trajectory is paired to the same hypothetical world, truth shown
    # only in this post-session evaluator figure.
    worst = analysis["holdout"]["worst_ratios_v1b"][0]["case_id"]
    trace = next(json.loads(gzip.open(p,"rt",encoding="utf-8").read())
                 for p in sorted((DATA/"holdout").glob("trace-*.json.gz"))
                 if json.loads(gzip.open(p,"rt",encoding="utf-8").read())["v0"]["scenario"]["case_id"] == worst)
    fig, axes = plt.subplots(1,3,figsize=(14,4.7),constrained_layout=True)
    for ax,name,label in zip(axes,names,labels):
        t=trace[name]
        pts=[(0.,0.)]+[a["position"] for a in t["search"]["action_history"]]
        ax.plot([p[0] for p in pts],[p[1] for p in pts],color="#405b85",linewidth=.85,alpha=.8)
        sources=t["scenario"]["sources"]
        ax.scatter([s["x"] for s in sources],[s["y"] for s in sources],c="#bd4446",marker="x",s=28,label="Source truth (evaluation only)")
        ax.scatter([0],[0],marker="*",s=90,color="#dc9944")
        row=next(r for r in docs["holdout"]["rows"] if r["case_id"]==worst and r["strategy"]==name)
        ax.set(title=f'{label}: {row["virtual_time_s"]:.1f}s / {row["physical_lower_bound_s"]:.1f}s = {row["time_over_physical_lower_bound"]:.3f}x',
               xlabel="x (m)",ylabel="y (m)",xlim=(-1900,1900),ylim=(-1900,1900),aspect="equal")
    fig.suptitle("Highest V1b lower-bound ratio in holdout: "+worst)
    fig.savefig(OUT / "worst_holdout_routes.png")
    plt.close(fig)
    print(json.dumps({k: v["comparisons"] for k,v in analysis.items()}, indent=2))
    print(json.dumps(analysis["validation"]["robust_summary"],indent=2))
    print("worst_holdout", worst)


if __name__ == "__main__":
    main()
