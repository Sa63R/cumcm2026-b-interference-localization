# 论文工作稿与第二、三问研究材料

当前手改入口已按用户要求转到本目录同级的 `../final2/main.tex`。其中第一节为新增“问题重述”，后续为符号说明和一、二、三问模型；全部正文仍在同一文件。`../final2/main.pdf`为22页编译稿。版式参考用户指定的2025 B060展示论文，第四问模型尚未合并。

`overleaf-single/`为新增问题重述前的本地快照；Overleaf停留在此前上传阶段。本次后续修改只在 `final2` 本地完成，不再遥控浏览器。

本目录同时保留第二、三问的独立研究稿与辅助材料，包含单源主动定位、发现覆盖、状态压缩A*、清除及终止证明、PPO对照、实验比较、消融、行为诊断与真实轨迹分析。

## 阅读与编辑

- `overleaf-single/`：当前单文件工作稿和阅读PDF；只需修改 `main.tex`，`figures/`存放配图。
- `overleaf-single-upload.zip`：当前单文件稿的完整可编译上传包。
- `backups/overleaf-before-single-20260912.zip`：清理前完整在线项目，36个文件，含原来全部旧稿；完整性及来源核验见 `backup_manifest.json`。
- `overleaf-merged/`与`overleaf-merged-upload.zip`：保留的上一版分文件工作稿。
- `第二问与第三问论文稿.pdf`：排版阅读版。
- `第二问与第三问论文稿.tex`：可直接编辑、XeLaTeX编译的源稿。
- `第二问与第三问论文稿.md`：便于移植到其他论文格式的正文。
- `sources/`：分节草稿、代码定位及13项结构化实验证据。
- `figures/`：8幅图，每幅有PDF矢量版、SVG版、400 dpi PNG及导出清单。
- `data/`：图表所需的费用汇总及三例四方法轨迹数据，附来源哈希。
- `references/`：两篇国赛官方展示范文、算法原论文、BibTeX和写作借鉴笔记。
- `qa/`：内部编译日志与逐页渲染，不纳入Git。
- `VALIDATION.md`：数据、文档结构和逐页视觉核验记录。
- `OVERLEAF.md`：已导入的Overleaf项目链接与编译入口；`overleaf-import/`和`overleaf-q23-upload.zip`为可移植副本。

正文方法以首版状态搜索为主干，后续相对静默与16源上限精化单列。首版四方法主表、后期RL配对、后续消融和官方50局分别报告，不跨场景、下界或模型版本拼接排名。首版主表中的前瞻rollout是较强基线，不是最初朴素方法。

## 本稿的实质边界

第二问已给出自足的几何候选域、可接收证明、有限选点策略与真实双测算例；算例原始动作、版本匹配与重建数值见`sources/q2_example.json`。现有材料主要验证第三问完整策略，尚无第二检测点单独的受控性能对比，因此正文没有声称该点选择具有连续全局最优性，也没有用第三问总耗时直接证明第二问策略最优。

四方法的程序实际计算时间与机器人计费时间分开。后期48局未能判定RL与状态搜索的平均差异方向；48/48先扫20频道仅支持有限候选空间中的行为诊断。50局官方演练作为真实接口与执行补充，不充当四方法同场景排名。

本轮只整理、核对并绘制已有数据，没有启动正式测试、演练采集、训练或新的性能实验，没有连接远程计算主机。

## 可复现构建

已验证环境：Windows，Python 3.12，Matplotlib 3.11.2，NumPy 2.5.3，pypandoc_binary 1.17（Pandoc 3.9），TeX Live 2026，XeLaTeX，字体SimSun/SimHei、Times New Roman、Microsoft YaHei、Consolas、FangSong。

当前单文件稿：运行 `python scripts/build_single.py`，或在 `overleaf-single/` 中运行两次 `xelatex main.tex`。此脚本只编译、核对引用并打包，不重新生成或覆盖正文，适合手改后的重编译。

旧二三问独立稿：在本目录运行两次 `xelatex 第二问与第三问论文稿.tex`。`figures/`中的PDF与源稿一同保留。编译只涉及已有正文与图形，不运行实验。

自动构建脚本：

```text
python scripts/plot_figures.py
python scripts/build_paper.py
```

`plot_figures.py`读取本目录已保存数据并使用安装的scientific-visualization skill。`build_paper.py`由分节源稿生成MD、TeX和PDF；它会覆盖三份派生稿，因此直接手改TeX后请勿再运行该脚本，或将对应修改同步回源稿/构建脚本。

`prepare_data.py`仅用于从原工作区重新抽取图表数据，需原始q3分支/档案仍位于本目录的上一级；不需要它即可用交付的data重绘。所有脚本均不调用策略或模拟器。

数据复核运行`python scripts/validate_data.py`（需保留原始研究目录）；编译后结构检查运行`python scripts/validate_document.py`。验证结果保存在`sources/validation_data.json`与`sources/validation_document.json`，分别记录432项数据检查、72项文档及图形检查。

本机任务使用Codex捆绑Python，补充绘图库置于工作区`tmp/paper-deps`，未改动各实验分支的Python环境。运行时将该目录加入PYTHONPATH即可。

## 绘图skill与参考资料

按用户要求从GitHub搜索、安装并实际使用了[Scientific Visualization](https://github.com/K-Dense-AI/scientific-agent-skills/tree/515742b7ed522c78f19bb9b61f346a0435785302/skills/scientific-visualization)，版本固定为`515742b7ed522c78f19bb9b61f346a0435785302`。调用其`style_context`、`export_figure`，并使用图像元数据与配色审核脚本；图注保留区间含义、样本数、案例选择规则。相似灰度由面板、文字及不同标记补充识别，不宣称仅凭配色通过无障碍认证。

软件出处：Kassis T, Agarwal V, He Y, Patel D, Brueckner A M. *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065, 2026. [原文记录](https://arxiv.org/abs/2609.00065)。它用于说明实际使用的软件工作流，不作为本题算法有效性的证据。

参考资料已核验官方展示来源、论文作者及年份。阅读PDF保留在本地，GitHub提交来源链接、书目信息、笔记与文件哈希，避免将第三方整篇资料作为本项目原创发布。Held–Karp全文未能下载，已保留SIAM原刊书目信息；其余下载结果见`references/download_manifest.json`。
