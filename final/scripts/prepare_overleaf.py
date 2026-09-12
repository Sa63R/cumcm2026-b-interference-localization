"""Make a portable Overleaf copy without altering the original paper."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "第二问与第三问论文稿.tex"
OUT = ROOT / "overleaf-import"
OUT.mkdir(exist_ok=True)
(OUT / "q23_figures").mkdir(exist_ok=True)
original = SOURCE.read_text(encoding="utf-8")
portable = original.replace("fontset=windows", "fontset=fandol")
portable = portable.replace(r"\setmainfont{Times New Roman}", r"\setmainfont{TeX Gyre Termes}")
portable = portable.replace(r"\setmonofont{Consolas}", r"\setsansfont{TeX Gyre Heros}" + "\n" + r"\setmonofont{DejaVuSansMono.ttf}[BoldFont=DejaVuSansMono-Bold.ttf,ItalicFont=DejaVuSansMono-Oblique.ttf,BoldItalicFont=DejaVuSansMono-BoldOblique.ttf]")
portable = re.sub(r"^\\setCJK(?:main|sans|mono)font[^\n]*\n", "", portable, flags=re.M)
portable = portable.replace("{figures/", "{q23_figures/")
assert "fontset=windows" not in portable
assert "Times New Roman" not in portable
assert "{figures/" not in portable
(OUT / "q23_paper.tex").write_text(portable, encoding="utf-8", newline="\n")
before, body = portable.split(r"\begin{document}", 1)
body, tail = body.rsplit(r"\end{document}", 1)
assert not tail.strip()
(OUT / "q23_body.tex").write_text(
    "% Full body of the Q2/Q3 paper; intended for deliberate later integration.\n"
    "% Includes its own title, symbols, references, and appendices.\n"
    + body.strip() + "\n", encoding="utf-8", newline="\n")
(OUT / "q23_preamble_reference.tex").write_text(
    "% REFERENCE ONLY: reconcile this preamble with the existing project before use.\n"
    "% Does not include a documentclass or begin/end document.\n"
    + "\n".join(before.splitlines()[1:]).strip() + "\n", encoding="utf-8", newline="\n")
figures = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{q23_figures/([^}]+)\}", portable)))
assert len(figures) == 8, figures
for name in figures:
    shutil.copy2(ROOT / "figures" / name, OUT / "q23_figures" / name)

(OUT / "README.md").write_text("""# Overleaf 上传包

这是现有第二问、第三问论文的可移植副本；原始正文、公式与 8 幅图均保留。

## 编译

1. 将此目录内的文件与 `q23_figures` 文件夹上传至 Overleaf 项目根目录。
2. 将主文档（Main document）设置为 `q23_paper.tex`。
3. 将编译器（Compiler）设置为 **XeLaTeX**；TeX Live 使用 2025 或更新版。
4. 重新编译即可。全文参考文献已直接写入正文，无需 BibTeX/Biber。

文件采用独立的 `q23_` 前缀，可与现有 `main.tex`、`q1.tex`、`q2.tex` 并存。
上传不会自动把内容接到现有主文稿中；切换主文档只改变当前编译入口。

## 可移植处理

- 将 `ctexart` 的 `fontset=windows` 改为 `fontset=fandol`。
- 中文使用 TeX Live 提供的 Fandol 字体；正文西文使用 TeX Gyre Termes，
  西文无衬线使用 TeX Gyre Heros，等宽使用 TeX Live 自带 DejaVu Sans Mono，
  以完整显示伪代码中的希腊字母（Latin Modern Mono 缺少 θ）。
- 原来显式指定的宋体、黑体、微软雅黑、仿宋、Consolas 等 Windows 字体已移除。
- 图片路径从 `figures/` 改为 `q23_figures/`，仅复制正文实际使用的矢量 PDF 图。
- 字体变化可能改变换行与页数，正文与实验数据不变。

## 后续合稿辅助文件

`q23_body.tex` 是完整正文片段（含自身标题、符号表、参考文献与附录）。
`q23_preamble_reference.tex` 仅供核对宏包、命令与格式依赖。
合稿前需人工处理既有模板、节编号、符号表、参考文献与页眉设置；
不要直接将参考导言覆盖现有项目导言，也不要直接重复引入完整正文。

`upload_manifest.json` 记录本上传包文件的 SHA-256 与源稿哈希。
""", encoding="utf-8", newline="\n")

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

package_files = [OUT / "q23_paper.tex", OUT / "q23_body.tex", OUT / "q23_preamble_reference.tex", OUT / "README.md"]
package_files += [OUT / "q23_figures" / name for name in figures]
manifest = {
    "source": str(SOURCE), "source_sha256": sha(SOURCE),
    "main_document": "q23_paper.tex", "compiler": "XeLaTeX", "minimum_texlive": 2025,
    "figure_count": len(figures),
    "files": [{"path": str(p.relative_to(OUT)).replace("\\", "/"), "sha256": sha(p), "bytes": p.stat().st_size} for p in package_files],
}
(OUT / "upload_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
zip_path = ROOT / "overleaf-q23-upload.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for file in [*package_files, OUT / "upload_manifest.json"]:
        z.write(file, str(file.relative_to(OUT)).replace("\\", "/"))
print(json.dumps({"directory": str(OUT), "zip": str(zip_path), "files": len(package_files) + 1, "zip_bytes": zip_path.stat().st_size}, ensure_ascii=False))
