"""Render the result-linked Chinese manuscript and complete source appendix."""

from __future__ import annotations
import argparse
import csv
from html import escape
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image,
                               Table, TableStyle, Preformatted, KeepTogether)
from localization import CandidateRegion, select_next_point
from planning import coverage_points


def table(headers, rows):
    return "\n".join(["| " + " | ".join(map(str, headers)) + " |",
                      "|" + "---|" * len(headers)] +
                     ["| " + " | ".join(map(str,row)) + " |" for row in rows])


def source_files():
    return sorted([p for directory in ("src", "experiments", "tests", "scripts")
                   for p in (ROOT/directory).rglob("*") if p.suffix in {".py", ".ps1"}] + list(ROOT.glob("*.cmd")))


def prepare(study):
    summary=json.loads((study/"summary.json").read_text(encoding="utf-8"))
    manifest=json.loads((study/"manifest.json").read_text(encoding="utf-8"))
    groups=summary["groups"]
    def group(problem,strategy,kind="random"):
        return next(g for g in groups if g["problem"]==problem and g["strategy"]==strategy and g["case_kind"]==kind)
    selected={3:"adaptive_center",4:"triangular"}
    q3,q4=(group(p,selected[p]) for p in (3,4))
    count=manifest["random_cases_per_problem"]
    labels={"baseline":"基准扫描", "adaptive_center":"即时中心定位", "adaptive_minimax":"即时代理选点",
            "deferred":"方格延迟定位", "triangular":"三角延迟定位"}
    replacements={
        "ABSTRACT_RESULTS": f"对每问 {count} 个本地随机案例与 7 个困难案例进行成对验证。所选问题三策略在随机案例中的完整清除为 {q3['all_clear_runs']}/{count}，问题四为 {q4['all_clear_runs']}/{count}；各局平均定位清除时间的均值分别为 {q3['mean_case_time_per_cleared_s']:.2f} s/源与 {q4['mean_case_time_per_cleared_s']:.2f} s/源，相比各自基准总虚拟时间均值分别变化 {(-q3['relative_mean_time_reduction_pct']):+.2f}% 与 {(-q4['relative_mean_time_reduction_pct']):+.2f}%。所有数字来自本地构造场景，官方演练和正式表需以官方模拟器实测补齐。",
        "REALIZABLE_COUNTEREXAMPLE": "上述三角形也可由合法窄角观测产生。取真实源为重心 (18,6√3)，第一检测点为 (-900,0)，示向度为 1°；将检测点绕重心旋转 120° 和 240°，对应示向度分别为 121° 和 241°。三个站点到源的距离均约 918.059 m，示向误差均在 ±1° 内；每个扇形的一条边界恰为三角形的一边，另一条边界不切入三角形，因此三个半角 1° 的正向扇形相交正好得到此三角形。",
        "SECOND_POINT_EXAMPLE": "例如首测位置 (0,0)、示向度 0°，按默认参数选得第二点约 (857.729,514.103) m。仅用于事后评价，设真源为 (1000,0)，第二读数按零潜在误差舍入为 285.47°，更新外包区域的包围圆半径约 22.812 m；共线对照第二点 (500,0)、读数 0° 时约 500.154 m。横向交会显著缩域，但前者仍大于 20 m，不能据此直接声称保证清除。",
        "Q4_IMPLEMENTATION": f"实现按三角形到原点的最短距离筛选与圆域相交的单元，并保留其全部顶点。去重后，方格方案共有 {len(coverage_points(4,variant='baseline'))} 个站点，三角方案共有 {len(coverage_points(4,variant='triangular'))} 个站点。在最近邻覆盖路线基础上，使用固定原点起点、终点自由的开路2-opt：只在逆转一段路径能严格缩短长度时接受交换。为减少覆盖阶段的往返，三角策略先扫描全部覆盖点，再按当前位置到候选区估计中心的距离处理源队列，并使用有限次中心主动检测及同一光学兜底。对照中的方格延迟策略采用相同路线优化和处理时机，使网格变更的效果可以相对单独比较。",
        "EXPERIMENT_DESIGN": f"随机实验每问取 {count} 个场景，种子为 {manifest['start_seed']} 至 {manifest['start_seed']+count-1}；两问另各构造 7 个困难场景。源数量从 10 至 16 等概率抽取，位置按圆内面积均匀，接收半径在 1000 至 1500 m 均匀，频道无放回抽取；问题四随机指定至少一个全向源和至少一个定向源，定向角均匀。位置与频道经固定种子哈希得到地点固定误差，示向度量化到 0.01°；若量化越过题设误差上界则向真值方向回调 0.01°。这套明确的实验分布不代表官方生成规律。先前用于排查实现的少量种子不纳入本表。",
    }
    rows=[]
    for g in groups:
        if g["case_kind"]=="random":
            rows.append([g["problem"],labels[g["strategy"]],f"{g['all_clear_runs']}/{g['runs']}",
                         f"{g['mean_virtual_time_s']:.2f}",f"{g['mean_case_time_per_cleared_s']:.2f}",
                         f"{g['mean_runtime_s']:.3f}"])
    replacements["RANDOM_TABLE"]=table(["问题","策略","全清局数","均值T / s","均值T/K / s","均值τ / s"],rows)
    findings=[]
    for p in (3,4):
        g=group(p,selected[p]); ci=g["paired_mean_saving_ci95_s"]
        findings.append(f"问题{p}选用{labels[selected[p]]}。相对基准，平均总虚拟时间降低 {g['relative_mean_time_reduction_pct']:.2f}%，成对平均节省 {g['mean_paired_seconds_saved']:.2f} s，其案例级自助抽样95%区间为 [{ci[0]:.2f},{ci[1]:.2f}] s。该策略合并口径 ΣT/ΣK 为 {g['pooled_time_per_cleared_s']:.2f} s/源；它与表中各局 T/K 均值分别统计。")
    gm=group(3,"adaptive_minimax")
    findings.append(f"问题二代理选点策略在问题三全局任务中的平均总时间为 {gm['mean_virtual_time_s']:.2f} s，中心策略为 {q3['mean_virtual_time_s']:.2f} s。这说明改善交会几何可能同时增加移动成本，不能仅据单源区域缩小就认定完整任务更快。最终运行入口默认中心策略。")
    replacements["RANDOM_FINDINGS"]="\n\n".join(findings)
    rows=[]
    for p in (3,4):
        for strategy in (["baseline",selected[p]]):
            g=group(p,strategy,"hard")
            rows.append([p,labels[strategy],f"{g['all_clear_runs']}/{g['runs']}",
                         f"{100*g['pooled_clear_fraction']:.1f}%",f"{g['mean_case_time_per_cleared_s']:.2f}",g["max_actions"]])
    replacements["HARD_TABLE"]=table(["问题","策略","全清局数","合并清除比例","均值T/K / s","最大动作数"],rows)
    validation=ROOT/"results"/"validation"/"summary.json"
    if validation.exists():
        checks=json.loads(validation.read_text(encoding="utf-8"))
        replacements["VALIDATION_RESULTS"]=f"当前自动化回归检查共 {checks['passed']} 项通过。其范围覆盖几何、候选点、空间覆盖、物理边界、客户端协议、完整搜索与正式证据登记；测试原始输出与研究源文件摘要随材料保存。"
    else:
        replacements["VALIDATION_RESULTS"]="几何、空间覆盖、物理边界、协议、搜索及证据登记测试的完整输出见支撑材料中的 results/validation/pytest.txt。"
    with (ROOT/"支撑材料"/"正式测试登记.csv").open(encoding="utf-8-sig",newline="") as stream: ledger=list(csv.DictReader(stream))
    formal=[]
    for p in (3,4):
        formal.append(f"### 问题{p}正式测试结果")
        records=[]
        for row in ledger:
            if row["problem"]==str(p):
                records.append([row.get("case_code") or f"测试{row['slot']} 待官方测试",
                                row.get("cleared_count") or "待实测", row.get("average_clear_time_s") or "待实测",
                                row.get("program_runtime_s") or "待实测"])
        formal.append(table(["测试案例编码","清除干扰源个数","平均定位清除时间 / s","程序运行时间 / s"],records))
    replacements["FORMAL_TABLES"]="\n\n".join(formal)
    replacements["SUPPORT_FILES"]=table(["路径","用途"],[
        ["src/","几何、主动定位、覆盖、策略、协议、独立仿真和正式登记完整代码"],
        ["tests/","各模块和端到端回归验证"],["experiments/","成对试验、绘图、论文和支撑包生成程序"],
        ["scripts/","Windows环境、单局运行与材料登记入口"],
        ["results/study/","随机与困难案例逐局CSV、汇总JSON、源摘要及压缩动作记录"],
        ["results/validation/","自动化测试输出与检查摘要"],
        ["论文/","可编辑正文、模板、核心插图与图形数值"],
        ["支撑材料/正式测试登记.csv","六次正式登记及结果来源"],
        ["支撑材料/正式日志/","原名原内容正式加密日志，尚待实际导出登记"],
        ["MANIFEST.sha256","压缩包内各文件内容摘要"]])
    replacements["REPRODUCTION"]=(f"自动化测试命令：python -m pytest -q。\n\n"
        f"本地研究命令：python -m experiments.run_study --output results/reproduced --random-cases {count} --start-seed {manifest['start_seed']}。"
        "输出目录必须尚不存在，以保留已有证据；相同种子和源程序应重现虚拟时间与动作结果，现实耗时允许依机器变化。\n\n"
        "图表命令：python -m experiments.make_figures --study results/study。论文命令：python -m experiments.build_paper。"
        "归档命令：python -m experiments.package_support。正式登记与审计命令见 scripts/materials.ps1 及支撑材料说明。")
    template=(ROOT/"论文"/"正文模板.md").read_text(encoding="utf-8")
    for key,value in replacements.items():template=template.replace("{{"+key+"}}",value)
    if re.search(r"\{\{[A-Z_]+\}\}",template):raise ValueError("Unresolved manuscript placeholder")
    return template


def render(text, destination):
    fonts=Path("C:/Windows/Fonts")
    if fonts.exists():
        pdfmetrics.registerFont(TTFont("CN",str(fonts/"simsun.ttc"),subfontIndex=0))
        pdfmetrics.registerFont(TTFont("CNBold",str(fonts/"simhei.ttf")))
    else:
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        # Font aliases allow rendering without copying proprietary font files.
        pdfmetrics.registerFontFamily("STSong-Light",normal="STSong-Light",bold="STSong-Light")
    cn="CN" if "CN" in pdfmetrics.getRegisteredFontNames() else "STSong-Light"
    bold="CNBold" if "CNBold" in pdfmetrics.getRegisteredFontNames() else cn
    width,height=A4
    usable=width-5*cm
    body=ParagraphStyle("Body",fontName=cn,fontSize=10.5,leading=17.0,firstLineIndent=21,rightIndent=10.5,
                        spaceAfter=7,wordWrap="CJK",alignment=TA_JUSTIFY)
    h1=ParagraphStyle("Title",fontName=bold,fontSize=18,leading=25,alignment=TA_CENTER,spaceAfter=20)
    h2=ParagraphStyle("Section",fontName=bold,fontSize=13,leading=19,spaceBefore=12,spaceAfter=9,keepWithNext=True)
    h3=ParagraphStyle("Subsection",fontName=bold,fontSize=11.3,leading=17,spaceBefore=8,spaceAfter=6,keepWithNext=True)
    cell=ParagraphStyle("Cell",fontName=cn,fontSize=8.2,leading=12.5,wordWrap="CJK")
    caption=ParagraphStyle("Caption",fontName=cn,fontSize=9,leading=13,alignment=TA_CENTER,spaceAfter=9)
    code=ParagraphStyle("Code",fontName=cn,fontSize=6.8,leading=8.5,spaceAfter=1)
    def formatted(value):
        value=escape(value)
        for old,new in {"ᵢ":"i","ⱼ":"j","₁":"1","₂":"2","ₖ":"k"}.items():
            value=value.replace(old,"<sub>"+new+"</sub>")
        return value
    doc=SimpleDocTemplate(str(destination),pagesize=A4,rightMargin=2.5*cm,leftMargin=2.5*cm,
                          topMargin=2.5*cm,bottomMargin=2.5*cm,title=text.splitlines()[0][2:],
                          author="",subject="CUMCM 2026 Problem B",pageCompression=1)
    flow=[]
    lines=text.splitlines();i=0;page_markers=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line=="<!-- PAGE -->":
            page_markers+=1
            if page_markers==1:flow.append(PageBreak())
            i+=1;continue
        if line.startswith("|"):
            block=[]
            while i<len(lines) and lines[i].strip().startswith("|"):
                values=[c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[-: ]+",c) for c in values):block.append(values)
                i+=1
            n=len(block[0]);ratios=[1]*n
            if n==2:ratios=[.9,2.1]
            elif n==3:ratios=[1,2.3,.7]
            elif n==4:ratios=[1.5,1,1.2,1.2]
            elif n==6:ratios=[.5,1.35,.8,1.05,1.05,1]
            widths=[usable*r/sum(ratios) for r in ratios]
            data=[[Paragraph(formatted(c),cell) for c in row] for row in block]
            t=Table(data,colWidths=widths,repeatRows=1,hAlign="CENTER")
            t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#E8EDF1")),
                ("VALIGN",(0,0),(-1,-1),"TOP"),("LINEABOVE",(0,0),(-1,0),.7,colors.black),
                ("LINEBELOW",(0,0),(-1,0),.4,colors.black),("LINEBELOW",(0,-1),(-1,-1),.7,colors.black),
                ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
                ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5)]))
            flow.extend([t,Spacer(1,10)]);continue
        match=re.match(r"!\[(.*?)\]\((.*?)\)",line)
        if match:
            from PIL import Image as PILImage
            path=ROOT/"论文"/match.group(2)
            with PILImage.open(path) as im:w,h=im.size
            image=Image(str(path),width=usable,height=usable*h/w)
            flow.append(KeepTogether([image,Paragraph(escape(match.group(1)),caption)]));i+=1;continue
        level=len(line)-len(line.lstrip("#"))
        if level:
            if line[level:].strip().startswith(("附录A","附录B")):flow.append(PageBreak())
            flow.append(Paragraph(formatted(line[level:].strip()),{1:h1,2:h2}.get(level,h3)))
        else:flow.append(Paragraph(formatted(line),body))
        i+=1
    flow.append(Spacer(1,8))
    for path in source_files():
        name=str(path.relative_to(ROOT)).replace("\\","/")
        flow.append(Paragraph(escape(name),h3))
        source=path.read_text(encoding="utf-8-sig").splitlines()
        output=[]
        for number,line in enumerate(source,1):
            prefix=f"{number:4d}  "
            line=line.expandtabs(4)
            for old in ("ᵢ","ⱼ","₁","₂","ₖ"):
                line=line.replace(old,"\\u"+format(ord(old),"04x"))
            if not line:output.append(prefix);continue
            while line:
                take=len(line)
                while pdfmetrics.stringWidth(prefix+line[:take],cn,6.8)>usable-4:take-=1
                if take<=0:raise ValueError("Code line does not fit")
                output.append(prefix+line[:take]);line=line[take:];prefix="      > "
        # Chunks permit page breaking without keeping entire files together.
        for first in range(0,len(output),20):
            flow.append(Preformatted("\n".join(output[first:first+20]),code))
        flow.append(Spacer(1,8))
    def footer(canvas,doc):
        canvas.saveState();canvas.setFont(cn,9)
        canvas.drawCentredString(width/2,1.6*cm,str(doc.page));canvas.restoreState()
    doc.build(flow,onFirstPage=footer,onLaterPages=footer)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,default=ROOT/"results"/"study")
    args=parser.parse_args()
    text=prepare(args.study)
    (ROOT/"论文"/"B题论文.md").write_text(text,encoding="utf-8")
    render(text,ROOT/"论文"/"B题论文.pdf")
    print("Created manuscript and PDF with complete source appendix.")


if __name__=="__main__":main()
