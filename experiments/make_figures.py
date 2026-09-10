"""Generate paper figures from geometry and saved synthetic observations."""

from __future__ import annotations
import argparse
import gzip
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import numpy as np
from geometry import minimum_enclosing_circle, polygon_diameter
from localization import CandidateRegion, select_next_point, second_point_candidates
from planning import coverage_points

BLUE, RED, GREEN, GRAY = "#235789", "#bc4749", "#25806f", "#78828d"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.facecolor": "white"})


def save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


def polygon(ax, vertices, **kwargs):
    ax.add_patch(Polygon(vertices, closed=True, **kwargs))


def plane(ax, limit=None):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.grid(alpha=.16)
    if limit is not None:
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)


def geometry_figures(output):
    source = (900., 500.)
    stations = ((0., 0.), (0., 1000.))
    region = CandidateRegion()
    for s in stations:
        region.observe(s, math.degrees(math.atan2(source[1]-s[1], source[0]-s[0])))
    circle = region.enclosing_disk()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")
    ax = axes[0]
    polygon(ax, region.vertices, facecolor=BLUE, alpha=.20, edgecolor=BLUE)
    ax.add_patch(Circle(circle.center, circle.radius, fill=False, edgecolor=GREEN, linewidth=1.5))
    ax.plot(*source, "x", color=RED, ms=8, label="示意真值")
    ax.plot(*circle.center, "+", color=GREEN, ms=10, label="包围圆圆心")
    ax.set_xlim(840, 960); ax.set_ylim(440, 560)
    plane(ax)
    ax.set_title(f"(a) 两次测向后的外包定位区域\nD={region.diameter:.2f} m  r*={circle.radius:.2f} m")
    ax.legend(loc="lower right", fontsize=8)
    ax = axes[1]
    vertices = [(0., 0.), (36., 0.), (18., 18*math.sqrt(3))]
    mec = minimum_enclosing_circle(vertices)
    polygon(ax, vertices, facecolor=BLUE, edgecolor=BLUE, alpha=.15)
    ax.add_patch(Circle((18, 0), 18, fill=False, edgecolor=RED, linestyle="--", label="以一条直径为直径的圆"))
    ax.add_patch(Circle(mec.center, mec.radius, fill=False, edgecolor=GREEN, label="最小包围圆"))
    ax.scatter(*zip(*vertices), color=BLUE, s=20)
    ax.set_xlim(-7, 43); ax.set_ylim(-22, 38)
    plane(ax)
    ax.set_title("(b) 直径不足以保证 20 m 清除\n等边三角形 D=36 m  r*=20.785 m")
    ax.legend(loc="lower center", fontsize=8)
    save(fig, output/"fig01_geometry.png")

    first = CandidateRegion().observe((0, 0), 0)
    safe = first.guaranteed_detection_region()
    choice = select_next_point(first, (0, 0))
    candidates = second_point_candidates(first, (0, 0))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")
    ax = axes[0]
    polygon(ax, safe.vertices, facecolor=GREEN, alpha=.16, edgecolor=GREEN, label="保证再接收域的内近似")
    polygon(ax, first.vertices, facecolor=BLUE, alpha=.45, edgecolor=BLUE, label="首测候选源区域")
    ax.scatter(*zip(*candidates), s=12, color=GRAY, label="可选检测点")
    ax.scatter(*choice.position, marker="*", s=130, color=RED, label="极小极大代理指标选点")
    ax.scatter(0,0,c="black", s=25)
    plane(ax); ax.set_xlim(-100, 1600); ax.set_ylim(-750,750)
    ax.set_title("(a) 首测 (0,0)，示向度 0°")
    ax.legend(fontsize=7, loc="upper left")
    ax=axes[1]
    targets = [250,500,1000,1400]
    approaches = {"沿示向度前移至中心": first.enclosing_disk().center,
                  "中心横向偏移 150 m": (750.,150.), "代理指标选点": choice.position}
    widths=.24
    values={}
    for k,(label,second) in enumerate(approaches.items()):
        radii=[]
        for x in targets:
            candidate = first.copy()
            angle = math.degrees(math.atan2(-second[1],x-second[0]))
            # Ground truth is used only for this controlled post-choice evaluation.
            candidate.observe(second,angle)
            radii.append(candidate.enclosing_disk().radius)
        values[label] = radii
        ax.bar(np.arange(4)+(k-1)*widths,radii,widths,label=label,color=[BLUE,GREEN,RED][k])
    ax.set_xticks(np.arange(4),[str(x) for x in targets]);ax.set_xlabel("用于事后评价的真距离 / m")
    ax.set_ylabel("第二次测向后的包围圆半径 / m")
    ax.set_title("(b) 不同真距离下的示意比较")
    ax.legend(fontsize=7)
    ax.grid(axis="y",alpha=.15)
    save(fig,output/"fig02_second_point.png")
    (output/"geometry_values.json").write_text(json.dumps({
        "localization_diameter_m":region.diameter,"localization_mec_radius_m":circle.radius,
        "counterexample_diameter_m":polygon_diameter(vertices),"counterexample_mec_radius_m":mec.radius,
        "selected_second_point":choice.position,"controlled_zero_error_comparison":values,
        "comparison_note":"All second measurements use zero angular error; source x-axis positions are evaluation-only; receiver radius assumed 1500m for diagram comparison."
    },ensure_ascii=False,indent=2),encoding="utf-8")


def study_figures(study, output, selected):
    groups=json.loads((study/"summary.json").read_text(encoding="utf-8"))["groups"]
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout="constrained")
    for ax,problem in zip(axes,(3,4)):
        variant=selected[problem].replace("adaptive_center","adaptive")
        points=coverage_points(problem,variant=variant)
        ax.add_patch(Circle((0,0),1800,fill=False,color="black",lw=1.3,label="目标圆域"))
        xy=np.array([(p.x,p.y) for p in points])
        ax.plot(xy[:,0],xy[:,1],color=BLUE,alpha=.5,lw=.8)
        ax.scatter(xy[:,0],xy[:,1],s=20,color=BLUE,label=f"{len(points)} 个覆盖检测点")
        if problem==3:
            for p in points:
                ax.add_patch(Circle((p.x,p.y),1000,facecolor=GREEN,alpha=.06,edgecolor=GREEN))
        plane(ax,3000 if problem==3 else 2900)
        ax.set_title(f"({chr(94+problem)}) 问题 {problem} 的全域覆盖")
        ax.legend(fontsize=8,loc="upper right")
    save(fig,output/"fig03_coverage.png")

    fig,axes=plt.subplots(1,2,figsize=(10,4.3),layout="constrained")
    for ax,problem in zip(axes,(3,4)):
        case=f"q{problem}-hard-boundary_outward"
        path=study/"traces"/f"{case}--{selected[problem]}.json.gz"
        with gzip.open(path,"rt",encoding="utf-8") as stream:trace=json.load(stream)
        trajectory=np.array(trace["search"]["trajectory"])
        ax.plot(trajectory[:,0],trajectory[:,1],color=BLUE,alpha=.6,lw=.65,label="动作位置轨迹")
        truth=trace["evaluation"]["ground_truth"]["sources"]
        ax.scatter([s["x"] for s in truth],[s["y"] for s in truth],marker="x",s=38,color=RED,label="源真值（事后读取）")
        clears=[a["position"] for a in trace["search"]["action_history"] if a["action"]=="clear" and a["result"]=="success"]
        ax.scatter(*zip(*clears),facecolors="none",edgecolors=GREEN,s=48,label="成功清除动作位置")
        estimates=[s["center"] for s in trace["search"]["source_estimates"].values() if "center" in s]
        ax.scatter(*zip(*estimates),marker="+",c=GRAY,s=28,label="最终几何估计")
        for s in truth:
            if s["orientation_deg"] is not None:
                angle=math.radians(s["orientation_deg"])
                ax.arrow(s["x"],s["y"],250*math.cos(angle),250*math.sin(angle),color=RED,width=6,head_width=55)
        ax.add_patch(Circle((0,0),1800,fill=False,color="black",alpha=.4))
        plane(ax,2700)
        ax.set_title(f"问题 {problem} 边界案例\n清除 {len(clears)}/{len(truth)}，总虚拟时间 {trace['evaluation']['virtual_time_s']:.1f} s")
        ax.legend(fontsize=6.8,loc="upper left")
    save(fig,output/"fig05_trajectories.png")

    fig,axes=plt.subplots(1,2,figsize=(10,4.1),layout="constrained")
    labels={"baseline":"基准", "adaptive_center":"即时定位", "adaptive_minimax":"代理选点", "triangular":"三角覆盖", "deferred":"延迟定位"}
    for ax,problem in zip(axes,(3,4)):
        subset=[g for g in groups if g["problem"]==problem and g["case_kind"]=="random"]
        names=[labels.get(g["strategy"],g["strategy"]) for g in subset]
        bottom=np.zeros(len(subset))
        for key,label,color in [("movement_s","移动",BLUE),("detection_s","检测",GREEN),
                                ("switching_s","切换",GRAY),("optical_s","光学定位",RED),("removal_s","激光清除","#d9b44a")]:
            values=[g["mean_time_breakdown_s"][key] for g in subset]
            ax.bar(names,values,bottom=bottom,color=color,label=label)
            bottom+=values
        ax.set_title(f"问题 {problem} 同案例对照  n={subset[0]['runs']}")
        ax.set_ylabel("平均总虚拟时间 / s")
        ax.grid(axis="y",alpha=.15)
        for i,total in enumerate(bottom):ax.text(i,total+80,f"{total:.0f}",ha="center",fontsize=8)
        ax.set_ylim(0,max(bottom)*1.2)
        ax.legend(ncol=3,fontsize=7,loc="upper left")
    save(fig,output/"fig04_time_comparison.png")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,default=ROOT/"results"/"study")
    parser.add_argument("--output",type=Path,default=ROOT/"论文"/"figures")
    parser.add_argument("--q4-strategy",default="triangular")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    geometry_figures(args.output)
    study_figures(args.study,args.output,{3:"adaptive_center",4:args.q4_strategy})


if __name__=="__main__":main()
