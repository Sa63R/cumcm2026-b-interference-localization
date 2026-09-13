# 分章节 Overleaf 项目

主文件为 `main.tex`，编译器为 **XeLaTeX**。在 Overleaf 中始终编译 main.tex；
章节文件只有正文，不含 documentclass 或 document 环境，不单独编译。

| 修改内容 | 文件 |
| --- | --- |
| 宏包、字体、页边距、页眉、编号格式 | config/preamble.tex |
| 标题、摘要、关键词 | sections/abstract.tex |
| 问题重述 | sections/01_problem.tex |
| 问题分析 | sections/02_analysis.tex |
| 模型假设 | sections/03_assumptions.tex |
| 符号说明 | sections/04_symbols.tex |
| 各问排列顺序与“模型的建立与求解”标题 | sections/05_models.tex |
| 第一问、第二问 | sections/q1.tex、sections/q2.tex |
| 第三问标题、引言与各部分顺序 | sections/q3.tex |
| 第三问模型与算法 | sections/q3/01_model_and_algorithm.tex |
| 第三问对照方法与实验设置 | sections/q3/02_baselines_and_setup.tex |
| 第三问结果与消融 | sections/q3/03_results_and_ablations.tex |
| 第三问训练与官方演练 | sections/q3/04_training_and_practice.tex |
| 第三问轨迹与适用条件讨论 | sections/q3/05_trajectories_and_discussion.tex |
| 参考文献 | references/references.tex |
| 实验来源附录、轨迹附录 | appendices/a_evidence.tex、appendices/b_trajectories.tex |
| 图片及第一问 TikZ 算例 | figures/ |

使用 `\input` 保留原稿的连续排版、分页命令、公式编号、标签和引用。
所有图片和 input 路径均相对于项目根目录书写。文献仍使用原有 thebibliography，
添加条目时沿用 bibitem 和 cite，不需要引入 BibTeX/Biber。

本次只整理文件结构。原稿仅有前三问模型，第四问仍只出现于重述和分析，未补写内容。
根目录外的旧版本未用作同步来源；本目录来自同步时的 Overleaf 最新稿。
Overleaf 内已有的参考论文 PDF 保持原位置，本地源码包仅包含编译本文所需文件。
