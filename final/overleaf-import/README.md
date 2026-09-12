# Overleaf 上传包

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
