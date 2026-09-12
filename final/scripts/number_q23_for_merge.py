"""Convert split Q2/Q3 chapters to automatic references without changing math."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "overleaf-merged"
SEC = OUT / "sections"
NAMES = ["q2.tex", "q3.tex", "symbols_q23.tex", "references.tex", "appendix.tex"]
before = {name: (SEC / name).read_text(encoding="utf-8") for name in NAMES}
after = dict(before)
audit = {"formula_labels": [], "figure_labels": [], "table_labels": [], "input_sha256": {}}
for name, text in before.items():
    audit["input_sha256"][name] = hashlib.sha256(text.encode("utf-8")).hexdigest()

def equation(match):
    math, question, number = match.group(1), match.group(2), match.group(3)
    label = f"eq:q{question}-{number}"
    audit["formula_labels"].append(label)
    return r"\begin{equation}\label{" + label + "}\n" + math.strip() + "\n" + r"\end{equation}"

for name in NAMES:
    text = after[name]
    text = re.sub(r"\\\[\s*(.*?)\s*\\tag\{([23])-(\d+)\}\s*\\\]", equation, text, flags=re.S)
    text = re.sub(r"式\(([23])-(\d+)\)", lambda m: r"式\eqref{eq:q" + m.group(1) + "-" + m.group(2) + "}", text)
    # Remove typed image numbers from both visible and accessibility captions.
    text = re.sub(r"alt=\{图[1-8]\s+", "alt={", text)
    def caption(match):
        label = f"fig:q23-{match.group(1)}"
        audit["figure_labels"].append(label)
        return r"\caption{" + match.group(2) + "}\n" + r"\label{" + label + "}"
    text = re.sub(r"(?m)^\\caption\{图([1-8])\s+(.*)\}$", caption, text)
    text = re.sub(r"图([1-8])(?!\d)", lambda m: r"图\ref{fig:q23-" + m.group(1) + "}", text)
    text = text.replace("第3.3节", r"第\ref{sec:q3-3}节")
    after[name] = text

table_labels = {"1": "tab:q23-symbols", "2": "tab:q2-example",
                "3": "tab:q3-comparison", "4": "tab:q23-evidence"}
table_pattern = re.compile(
    r"\\begin\{center\}\\small 表([1-4]) ([^\n]+)\\end\{center\}\s*"
    r"\\nopagebreak\s*\\begin\{longtable\}(.*?)\s*"
    r"\\toprule\s*(.*?)\\midrule\s*\\endfirsthead\s*"
    r".*?\\endhead\s*\\bottomrule\s*\\endlastfoot\s*(.*?)\\end\{longtable\}", re.S)

def table(match):
    number, title, spec, header, rows = match.groups()
    label = table_labels[number]
    audit["table_labels"].append(label)
    return (r"\begin{center}" + "\n" +
            r"\begin{minipage}{\linewidth}" + "\n" +
            r"\centering\small" + "\n" +
            r"\captionof{table}{" + title + "}\n" +
            r"\label{" + label + "}\n" +
            r"\vspace{4pt}" + "\n" +
            r"\begin{tabular}" + spec.strip() + "\n" +
            r"\toprule" + "\n" + header.strip() + "\n" +
            r"\midrule" + "\n" + rows.strip() + "\n" +
            r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" +
            r"\end{minipage}" + "\n" + r"\end{center}")

for name in NAMES:
    after[name] = table_pattern.sub(table, after[name])

algorithm_2 = r"""\par\noindent\begin{minipage}{\linewidth}
\refstepcounter{algorithm}\label{alg:q2-probe}
\textbf{算法\thealgorithm：单源主动检测点选择}

\small
\textbf{输入：}当前点 \(x_0\)、首测方向 \(\theta_1\)、源外包区域
\(\widehat C_f\)、同频道历史测点。
\begin{enumerate}[label=\arabic*.,ref=\arabic*,leftmargin=2em,itemsep=2pt,topsep=4pt]
  \item 求 \(\widehat C_f\) 的包围圆心及长轴，先生成基础集合 \(A_0\)。
  \item 基础集合有有效新点时再扩展 \(A_1\)，去重并核验保证接收条件。
  \item 构造名义源点集合 \(H_f\)；对每个 \(x\in A\) 计算式\eqref{eq:q2-8}。
  \item 返回分值最小的 \(x\)；基础集合无新点时尝试圆心与偏移回退。
  \item 执行真实测量，以真实反馈更新区域或触发近场清除。
\end{enumerate}
\end{minipage}\par"""

algorithm_3 = r"""\par\noindent\begin{minipage}{\linewidth}
\refstepcounter{algorithm}\label{alg:q3-search}
\textbf{算法\thealgorithm：多源滚动搜索与清除}

\small
\textbf{输入：}目标圆域 \(\Omega\)、20 个频道、源数范围 10---16 和设备动作费用。
\begin{enumerate}[label=\arabic*.,ref=\arabic*,leftmargin=2em,itemsep=2pt,topsep=4pt]
  \item 从原点进入环境，扫描频道并初始化区域、已发现集与未来覆盖站。
  \item\label{step:q3-finish-check} 若已有 16 次不同频道的真实成功清除，确认完成并退出。
  \item 构造剩余覆盖任务和已知源任务，冻结代表位置及代理服务费。
  \item 用状态压缩 A* 求有限路线；允许一次经全域核验的未来站移动。
  \item 若首任务为覆盖扫描：逐频道检查清除证书、静默证书和数量上限；
        对仍需查询的频道实际测量，只登记已完成的真实覆盖证据。
  \item 若首任务为源处理：调用算法\ref{alg:q2-probe}，达到清除尺度即执行清除；
        有限探测不足时执行光学网格；完成后尝试有价值的共享测量。
  \item 按实际反馈更新信息状态；检查完整覆盖及全部已知源清除条件。
  \item 未完成且预算允许则返回步骤\ref{step:q3-finish-check}；
        否则记录完成或中止原因并退出。
\end{enumerate}
\end{minipage}\par"""

for name, replacement, number in [("q2.tex", algorithm_2, "1"), ("q3.tex", algorithm_3, "2")]:
    pattern = re.compile(r"\\par\\noindent\\begin\{minipage\}\{\\linewidth\}\s*"
                         + r"\\textbf\{算法" + number + r"：.*?\\end\{minipage\}\\par", re.S)
    after[name], count = pattern.subn(lambda m: replacement, after[name])
    assert count == 1, (name, count)

assert len(audit["formula_labels"]) == 25
assert len(audit["figure_labels"]) == 8
assert len(audit["table_labels"]) == 4
for name in NAMES:
    old_math = re.findall(r"\\\[\s*(.*?)\s*\\tag\{[23]-\d+\}\s*\\\]", before[name], re.S)
    new_math = re.findall(r"\\begin\{equation\}\\label\{eq:q[23]-\d+\}\s*(.*?)\s*\\end\{equation\}", after[name], re.S)
    assert old_math == new_math, name
    assert not re.search(r"\\tag\{|式\([23]-\d+\)|图[1-8](?!\d)|表[1-4](?!\d)|第3\.3节|\\begin\{verbatim\}", after[name]), name
    (SEC / name).write_text(after[name], encoding="utf-8", newline="\n")

audit["display_math_contents_identical"] = True
audit["tables"] = "All 4 short tables converted from repeated-header longtable to captionof+tabular inside minipage, preserving first header and all data rows; avoids duplicate table-counter increments."
audit["algorithms"] = {"q2_steps": 5, "q3_steps": 8, "labels": ["alg:q2-probe", "alg:q3-search"], "meaning_unchanged": True}
audit["bibliography"] = "Five reference entries and existing textual citations unchanged; global bibliography integration belongs to main-manuscript work."
audit["dependencies"] = ["amsmath (equation, aligned, eqref)", "caption (captionof)", "graphicx (including existing alt option)", "booktabs", "array", "calc (real)", "enumitem", "needspace", "placeins", "hyperref recommended", "algorithm counter reset by section, thealgorithm=section-arabic, defined in main preamble"]
audit["output_sha256"] = {name: hashlib.sha256((SEC / name).read_bytes()).hexdigest() for name in NAMES}
(OUT / "q23_automatic_numbering_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"equations": 25, "figures": 8, "tables": 4, "algorithms": 2, "math_identical": True}, ensure_ascii=False))
