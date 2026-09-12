# 一、二、三问合并工作稿

**日常只需修改 `sections/q1.tex`、`sections/q2.tex`、`sections/q3.tex`。**

## 文件结构

```text
main.tex                 主文件，只负责按顺序引入各问
preamble.tex             字体、页边距、编号、图表等公共设置
sections/
  symbols.tex            各问符号对照
  q1.tex                 第一问：直接在这里手改
  q2.tex                 第二问：直接在这里手改
  q3.tex                 第三问
  references.tex         参考文献
  appendix.tex           实验来源与补充轨迹
figures/
  q1_case.tex            第一问原有TikZ图
  fig01_...pdf 至 fig08_...pdf  第二、三问配图
archive/                 Overleaf中保留的旧稿，不参与当前编译
```

## 修改方法

- 选择 `main.tex` 为主文档，使用 **XeLaTeX / TeX Live 2026**。
- 一级标题自动显示“问题一、问题二、问题三”，小节自动编号为1.1、2.1、3.1等。
- 方程按问题编号，如(1-1)、(2-1)。新增公式可用`equation`环境和唯一`label`。
- 正文中用`\eqref{eq:q2-1}`引用公式、`\ref{fig:q23-1}`引用图片；图、表全篇连续编号。
- 算法按问题编号为1-1、2-1、3-1；步骤使用普通`enumerate`，可直接增删。
- 文献使用`\cite{held1962}`等标签，序号自动更新。
- 改完点击Recompile。交叉引用有变化时，Overleaf会自动执行必要的重复编译。

## 采用的版本

第一问取自项目现有`q1.tex`，含原反例图；第二、三问取自刚导入的`q23_paper.tex`。
此次整理保留各问数学内容与实验数字，统一了文件结构、标题层级、字体和引用。
第一问与第二、三问的局部记号在符号表中作对应说明，未强行改写整篇推导。
原来的主稿、旧第二问及独立二三问稿均保留在Overleaf的`archive/`中，可随时对照。

当前工作稿集中呈现已有一、二、三问正文；旧主稿中的摘要占位、问题重述和假设等仍在备份中。
后续若要恢复全篇国赛模板，可从备份取用这些部分，再统一摘要和第四问。

本地副本位于`final/overleaf-merged/`，上传包为`final/overleaf-merged-upload.zip`。
`scripts/build_merged.py`只编译和检查现有文件，不会重新生成或覆盖手改后的各问正文。
