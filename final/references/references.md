# 第二问、第三问参考资料索引

核验日期：2026-09-12。论文内容以原论文为准；以下区分方法出处、写作范例及软件出处。PDF 为公开可获取的本地阅读副本，未改写原文。目录内 `.gitignore` 将 PDF、原网页与原文渲染图片保留在本地；GitHub 记录来源链接、书目信息和写作笔记。

## 1. 国赛官方公开范文

两篇均从教育部中国大学生在线的全国大学生数学建模竞赛组委会论文展示页取得。**作者姓名未在匿名展示稿中披露，以年份及论文编号识别，不推测学校和队员姓名。**

奖项证据链：全国组委会 2018-11-10《获奖名单（初稿）的说明》第三项明确说明，中国大学生在线展示当年部分全国一等奖论文。该通知的[湖南赛区官方转载](https://hunan.shumo.com/home/?p=1385)仍可访问，已保存为 `cumcm2018_award_notice_hunan.html`；原全国组委会旧链接目前返回 404。两篇论文现仍保留在组委会官方 2018 展示专题中。这里核验的是**官方一等奖展示来源及编号**，未将匿名编号和最终实名获奖名单作一一对应；不声称确认了具体参赛队。2023 年最终获奖名单另存作来源记录，与下面两篇 2018 范文没有身份对应关系。

| 编号 | 原文信息 | 官方入口及本地副本 | 适合借鉴的范围 |
|---|---|---|---|
| C18-B217 | 匿名参赛队，2018，《基于多原则比较和蒙特卡洛模拟的 RGV 动态调度模型》，45 页 | [组委会展示页](https://dxs.moe.gov.cn/zx/a/hd_sxjm_sxjmlw_2018qgdxssxjmjslwzs_2018btlw/240206/1699832.shtml)；[官方 PDF](https://dxs.moe.gov.cn/zx/2018/1101/1541052361277.pdf)；本地 `cumcm2018_B217_rgv.pdf` | 先形式化系统，再比较规则与学习；流程图、方法对照表和调度轨迹互相解释。 |
| C18-A440 | 匿名参赛队，2018，《高温作业服设计》，31 页 | [组委会展示页](https://dxs.moe.gov.cn/zx/a/hd_sxjm_sxjmlw_2018qgdxssxjmjslwzs_2018atlw/240206/1699834.shtml)；[官方 PDF](https://dxs.moe.gov.cn/zx/2018/1101/1541052741358.pdf)；本地 `cumcm2018_A440_thermal_design.pdf` | 同一基础模型逐问复用，新增决策变量和约束；在最优结果之后用参数曲线解释机制。 |

展示网页显示的 2024-02-06 是页面迁移/现有发布日期字段；原文属于 **2018 年竞赛**，参考文献年份使用 2018。两篇仅作结构和表达参考，不将其结果作为本题算法证据。

## 2. 算法与实验方法原论文

| 引用键 | 准确书目信息 | 核验来源、PDF及用途 |
|---|---|---|
| `held1962sequencing` | Michael Held, Richard M. Karp. A Dynamic Programming Approach to Sequencing Problems. *Journal of the Society for Industrial and Applied Mathematics*, 10(1):196–210, 1962. DOI: 10.1137/0110015. | [SIAM 原刊记录](https://epubs.siam.org/doi/pdfplus/10.1137/0110015)。用于子集动态规划/排序问题出处。出版社 PDF 返回访问验证页，未下载到正文，不用假 PDF 替代。IBM 另有 1961 ACM 会议记录，不将两版本的年份或刊名混写。 |
| `hart1968astar` | Peter E. Hart, Nils J. Nilsson, Bertram Raphael. A Formal Basis for the Heuristic Determination of Minimum Cost Paths. *IEEE Transactions on Systems Science and Cybernetics*, 4(2):100–107, 1968. DOI: 10.1109/TSSC.1968.300136. | [作者 Stanford 出版目录](https://ai.stanford.edu/~nilsson/publications.html)及其[原文 PDF](https://ai.stanford.edu/~nilsson/OnlinePubs-Nils/PublishedPapers/astar.pdf)；本地 `hart1968_astar.pdf`。用于启发式图搜索概念，只有代码满足相应条件时才援引最优性保证。 |
| `schulman2017ppo` | John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov. Proximal Policy Optimization Algorithms. arXiv:1707.06347, 2017. | [作者预印本记录](https://arxiv.org/abs/1707.06347)；本地 `schulman2017_ppo.pdf`。用于 PPO 裁剪目标、采样与多轮小批量更新出处。**该文是预印本，不标成 NeurIPS/ICLR 论文。** |
| `schulman2016gae` | John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel. High-Dimensional Continuous Control Using Generalized Advantage Estimation. *International Conference on Learning Representations*, 2016. | [ICLR 2016 官方录用列表](https://www.iclr.cc/archive/www/2016.html)，[作者预印本](https://arxiv.org/abs/1506.02438)；本地 `schulman2016_gae.pdf`。首次预印本为 2015，会议年份为 2016；当前下载为 arXiv v6（2018 更新）。 |
| `agarwal2021precipice` | Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, Marc G. Bellemare. Deep Reinforcement Learning at the Edge of the Statistical Precipice. *Advances in Neural Information Processing Systems*, 34:29304–29320, 2021. | [NeurIPS 官方记录](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)，[原文 PDF](https://papers.nips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf)；本地 `agarwal2021_statistical_precipice.pdf`。用于比较协议一致性、区间和性能分布。官方记录注明获 Outstanding Paper Award。 |
| `kool2019routing` | Wouter Kool, Herke van Hoof, Max Welling. Attention, Learn to Solve Routing Problems! *International Conference on Learning Representations*, 2019. | [会议原文记录](https://openreview.net/forum?id=ByxBFsRqYm)，[作者机构保存的正式版信息](https://dare.uva.nl/id/dbd5bafb-277e-40e0-887d-338d4898a090)，[作者预印本](https://arxiv.org/abs/1803.08475)；本地 `kool2019_attention_routing.pdf` 为 arXiv v3。用于学习方法与传统启发式的公平比较表达，**不表示本项目实现了注意力模型**。 |

## 3. 实际使用的软件/技能出处

- 实际绘图技能：K-Dense-AI/scientific-agent-skills，`scientific-visualization`，本次安装版本 `515742b7ed522c78f19bb9b61f346a0435785302`。[仓库](https://github.com/K-Dense-AI/scientific-agent-skills)。安装与使用事实由主任务记录。
- 文献已核验：Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, Aubrey M. Brueckner. *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065, 2026。[作者预印本记录](https://arxiv.org/abs/2609.00065)。当前 v2 发布于 2026-09-02，本地 `kassis2026_scientific_agent_skills.pdf`。
- 该文仅作为实际使用的科学绘图工作流出处；作者没有报告任务级评估，不能将它作为本题算法效果或图形质量的实验证据。宜写入软件环境或辅助工具说明，而非建模理论链条。

## 4. 文件完整性与阅读记录

`download_manifest.json` 记录下载 URL、页数、SHA-256。阅读中重点检查 B217 的第 1–3、9、17–19、22 页，A440 的第 1–4、16–18、21、23 页，PPO 的第 2–6 页，GAE 的第 1、3、4、7 页，NeurIPS 2021 的第 2–7 页，Kool 2019 的第 5–8 页。三张 `read_*.png` 为原论文相关页的只读渲染，不是本论文配图。
