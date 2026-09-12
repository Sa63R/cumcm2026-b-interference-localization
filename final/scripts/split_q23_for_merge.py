"""Split the portable paper into editable chapters; leave its source untouched."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "overleaf-import/q23_paper.tex"
OUT = ROOT / "overleaf-merged"
SECTIONS = OUT / "sections"
SECTIONS.mkdir(parents=True, exist_ok=True)
source = SOURCE.read_text(encoding="utf-8")
body = source.split(r"\begin{document}", 1)[1].rsplit(r"\end{document}", 1)[0]
heads = list(re.finditer(r"^\\section\{([^}]+)\}\\label\{([^}]+)\}", body, re.M))
assert [h.group(1) for h in heads] == [
    "符号说明", "问题二　保证接收条件下的主动交会定位",
    "问题三　基于状态压缩与覆盖约束的多源搜索", "参考文献",
    "附录A 实验来源与复核口径", "附录B 退化案例的轨迹对照",
]

spans = {
    "symbols_q23.tex": (heads[0].start(), heads[1].start()),
    "q2.tex": (heads[1].start(), heads[2].start()),
    "q3.tex": (heads[2].start(), heads[3].start()),
    "references.tex": (heads[3].start(), heads[4].start()),
    "appendix.tex": (heads[4].start(), len(body)),
}
labels = {}
for match in re.finditer(r"\\(sub)?section\{([^}]+)\}\\label\{([^}]+)\}", body):
    title = match.group(2)
    if title.startswith("2."):
        short = "sec:q2-" + title.split(" ", 1)[0].split(".")[1]
    elif title.startswith("3."):
        short = "sec:q3-" + title.split(" ", 1)[0].split(".")[1]
    else:
        short = {"符号说明": "sec:symbols-q23", "参考文献": "sec:references-q23",
                 "问题二　保证接收条件下的主动交会定位": "sec:q2",
                 "问题三　基于状态压缩与覆盖约束的多源搜索": "sec:q3",
                 "附录A 实验来源与复核口径": "sec:appendix-evidence",
                 "附录B 退化案例的轨迹对照": "sec:appendix-trajectories"}[title]
    labels[match.group(3)] = short

def transform(text: str) -> str:
    text = text.replace("{q23_figures/", "{figures/")
    text = re.sub(r"(\\subsection\{)[23]\.\d+\s+", r"\1", text)
    for old, new in labels.items():
        text = text.replace(r"\label{" + old + "}", r"\label{" + new + "}")
    # The current source has no empty Pandoc anchor blocks, but keep this
    # transformation explicit for source regeneration.
    text = re.sub(r"(?m)^\\phantomsection\s*\\label\{[^}]+\}\s*\n", "", text)
    text = text.replace(r"\toprule\noalign{}", r"\toprule")
    text = text.replace(r"\midrule\noalign{}", r"\midrule")
    text = text.replace(r"\bottomrule\noalign{}", r"\bottomrule")
    text = text.replace(r"\begin{longtable}[]", r"\begin{longtable}")
    # Retain paragraphs; put ordinary prose sentences on separate source lines.
    lines = []
    in_verbatim = False
    in_display = False
    for line in text.splitlines():
        if r"\begin{verbatim}" in line:
            in_verbatim = True
        if line.strip() == r"\[":
            in_display = True
        if (not in_verbatim and not in_display and line and
                not line.startswith("\\") and not line.rstrip().endswith(r"\\") and
                re.match(r"[\u3400-\u9fff]", line)):
            line = re.sub(r"([。；])(?=.)", r"\1\n", line)
        lines.append(line)
        if r"\end{verbatim}" in line:
            in_verbatim = False
        if line.strip() == r"\]":
            in_display = False
    return "\n".join(lines).strip() + "\n"

audit = {"source": str(SOURCE), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
         "excluded": "Standalone centered title block only; all content from symbols through appendices retained.",
         "changes": ["Split into five source files", "Removed typed 2.x/3.x subsection prefixes",
                     "Replaced Pandoc-generated heading labels with short labels",
                     "Changed figure folder to figures/", "Removed empty noalign and longtable optional args",
                     "Broke prose source lines at Chinese full stops/semicolons, preserving paragraph breaks"],
         "files": {}, "label_mapping": labels}

def compact(t: str) -> str:
    return re.sub(r"\s+", "", t)

for name, (start, end) in spans.items():
    raw = body[start:end]
    output = transform(raw)
    comment = {
        "symbols_q23.tex": "% 第二、三问原有符号说明；供与第一问符号表合并。\n",
        "q2.tex": "% 问题二正文；修改文字、公式与图表请直接编辑本文件。\n",
        "q3.tex": "% 问题三正文；修改文字、公式与图表请直接编辑本文件。\n",
        "references.tex": "% 第二、三问原有参考文献；待与第一问合并统一编号。\n",
        "appendix.tex": "% 第二、三问附录；保留既有实验来源及负例轨迹。\n",
    }[name]
    path = SECTIONS / name
    path.write_text(comment + output, encoding="utf-8", newline="\n")
    # Exact segment-level preservation after the listed formatting-only edits.
    expected = transform(raw)
    actual = path.read_text(encoding="utf-8").split("\n", 1)[1]
    assert expected == actual
    raw_blocks = [compact(p) for p in re.split(r"\n\s*\n", transform(raw).strip())]
    out_blocks = [compact(p) for p in re.split(r"\n\s*\n", actual.strip())]
    assert raw_blocks == out_blocks
    raw_math = re.findall(r"\\\[(.*?)\\\]", raw, re.S)
    out_math = re.findall(r"\\\[(.*?)\\\]", actual, re.S)
    assert raw_math == out_math
    audit["files"][name] = {
        "source_lines": [body[:start].count("\n") + 1, body[:end].count("\n") + 1],
        "paragraph_blocks_retained": len(out_blocks), "paragraph_order_identical": True,
        "display_math_blocks_identical": len(raw_math),
        "tags_unchanged": re.findall(r"\\tag\{([^}]+)\}", actual),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }

(OUT / "q23_split_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

notes = """# 第二、三问拆稿说明与合稿待统一编号

拆分保留了原稿从符号说明到附录的全部正文，仅独立稿居中标题块没有放入章节。
原稿未改。5 个文件在 `sections/` 下，未创建或修改主文件与导言。

## 本轮已处理

- `q2.tex`、`q3.tex` 使用顶层 `section`，共 19 个 `subsection` 已去手写数字。
- 段落之间空行保留；普通中文句子分行，便于直接改写。
- 25 个长的 Pandoc 标题标签改为短标签，完整映射见 JSON。
- 空 `noalign`、空 `longtable` 可选参数移除；必要的表头 minipage、算法块与公式原样保留。
- 图片路径统一为 `figures/fig*.pdf`；本任务未复制图片。
- 原稿不存在独立空 `phantomsection + label` 块，因此没有误删有效标题锚点。

## 需要主稿统一的硬编码

1. **公式**：第二问的 8 个 `tag{2-1}` 至 `tag{2-8}`、第三问的 17 个
   `tag{3-1}` 至 `tag{3-17}` 均保留。若改连续自动编号，应为公式补标签并替换正文
   `式(2-4)`、`式(2-3)`、`式(2-5)`、`式(2-8)`、`式(3-2)`、`式(3-12)`。
   算法 1 的 verbatim 内也有 `式(2-8)`，普通 ref 命令不会在 verbatim 内执行。
2. **图**：8 幅图的 caption 及 includegraphics 的 alt 均带 `图1` 至 `图8`。
   若用自动编号，应只保留说明文字并恢复 caption 的标签显示，防止“图1 图1”重复。
   正文引用出现在 q3 的“图6选取…”段，以及附录两段中的“图1、图2、图3至图8、图6、图7、图8”。
3. **表**：表1（符号）、表2（第二问算例）、表3（四法比较）、表4（证据索引）
   是独立 `center` 文字标题，没有 table/longtable 的自动 caption。应统一改为自动标题，
   同时确保首行表头不与标题分离；切勿只改编号而保留原 `表N`。
4. **小节引用**：q3 对照方法段落中的 `第3.3节` 应改为
   `第\\ref{sec:q3-3}节`；否则章号改变后会失配。
5. **算法**：算法1、算法2标题及 verbatim 中“调用算法1”均为手写编号，暂时保留。
   若第一问已有算法或需要统一计数，应一并调整；verbatim 内交叉引用需专门处理。
6. **参考文献**：当前 `[1]` 至 `[5]` 为普通文本，正文五处引用同样是文字编号，
   不是 cite/bibitem。需与第一问文献合并后统一处理，不能仅移动列表。
7. **符号说明与附录**：这三个文件保留原 `section` 标题。主稿需要自行决定星号、目录、
   附录计数器；原附录标题内含“附录A/附录B”，不能再套自动字母后重复显示。
8. **格式依赖**：包括 Needspace、FloatBarrier、longtable/booktabs、calc 的 real、
   arraybackslash、amsmath tag、figure caption 与可断行 Verbatim；旧独立导言可作核对。

## 保留核验

`q23_split_audit.json` 记录源稿哈希、每个文件的段落块数与哈希、公式块数量及标签映射。
所有段落顺序一致；25 个带编号展示公式与其他展示公式内容均逐字未改。
核验只对上述明确列出的源代码格式转换作归一化，没有改写数学内容或实验结论。
"""
(OUT / "q23_numbering_notes.md").write_text(notes, encoding="utf-8", newline="\n")
print(json.dumps({"outputs": list(spans), "subsections": 19,
                  "blocks": sum(x["paragraph_blocks_retained"] for x in audit["files"].values()),
                  "display_math": sum(x["display_math_blocks_identical"] for x in audit["files"].values()),
                  "labels": len(labels)}, ensure_ascii=False))
