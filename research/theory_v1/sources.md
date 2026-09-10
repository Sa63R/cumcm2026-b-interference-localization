# 已核查的原始资料及如何用于本任务

此表是研究依据，不是泛化算法综述。读取日期 2026-09-11。只将原论文、作者页面或官方出版页面作为技术依据；检索到的百科/问答只用于发现线索。

| 来源 | 原始链接 | 本任务采用的内容与限制 |
|---|---|---|
| 2026 国赛 B 题原题，第 1--4 页 | `../../../CUMCM2026Problems/B题/B题.pdf`（该相对路径需按最终目录调整；原绝对路径为 `D:/jwt/2026数模国赛/CUMCM2026Problems/B题/B题.pdf`） | 完整核对源数、独立频道、半径、反馈、移动与动作成本，实际PDF第3页另有渲染证据。不根据题面正式测试要求自动操作正式测试。 |
| Held & Karp, *A Dynamic Programming Approach to Sequencing Problems*, 1961 ACM / 1962 SIAM | https://doi.org/10.1145/800029.808532 ; https://doi.org/10.1137/0110015 | 官方 ACM 页面核对动态规划分解与旅行商子问题。源顺序子集 DP 可精确解指定边权图，但不能消除未知场景的观测分支。1962 版 DOI 的书目信息已交叉核对，未把付费全文假装读完。 |
| Silver & Veness, *Monte-Carlo Planning in Large POMDPs*, NeurIPS 2010 | https://proceedings.neurips.cc/paper/2010/hash/edfbe1afcf9246bb0d40eb4d8027d90f-Abstract.html | 作者原论文官方页面：后验粒子与树搜索、黑盒生成模型。用于区分“抽样搜索一个后验策略”与“证明全状态枚举最优”；有限粒子/浅搜索没有原问题最优性证书。 |
| Sunberg & Kochenderfer, *Online Algorithms for POMDPs with Continuous State, Action, and Observation Spaces*, ICAPS 2018 | https://ojs.aaai.org/index.php/ICAPS/article/view/13882 | 官方出版页面核对连续空间POMDP算法及 progressive widening 方向。说明连续观测使普通离散树扩展不能直接照搬，本记录不宣称已有代码等同完整 POMCPOW。 |
| Saldi, Yüksel & Linder, *Finite Model Approximations for Partially Observed Markov Decision Processes with Discounted Cost*, 2017 | https://arxiv.org/abs/1710.07009 | 原作者论文摘要明确是折扣成本、后验 MDP 的量化、弱连续性条件。用于说明连续转离散的近优结论需要条件，本题阈值和无折扣停止成本尚未满足这一套证明。 |
| Zahn, *Black Box Maximization of Circular Coverage*, 1962 | https://nvlpubs.nist.gov/nistpubs/Legacy/RPT/nbsreport7386.pdf ; https://nvlpubs.nist.gov/nistpubs/jres/66B/jresv66Bn4p181_A1b.pdf | NIST 原报告可检索，报告说明使用计算机实验研究圆覆盖。期刊PDF工具报文件过大，未完整读取；因此不能以该数字当六圆全局最优定理。此路线暂不进入严格界。 |
| Erich Friedman, *Circles Covering Circles* | https://erich-friedman.github.io/packing/circovcir/ | 构造收集者的原页面已打开，只作为几何布局候选线索。数值布局提供可行构造上界；没有在本任务中证明六圆不可能性。 |

新空间界采用折线胶囊增量的直接证明，不需要把适用于凸集/正 reach 集合的 Steiner 等式直接套给任意自交轨迹。小型双假设问题由脚本 Bellman 递推与独立连续解析分类双重验证。
