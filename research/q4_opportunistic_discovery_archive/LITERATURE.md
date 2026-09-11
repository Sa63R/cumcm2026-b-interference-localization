# R15 的参考价值与适用边界

1. Tan、Ghuge、Nagarajan，2024，[Informative Path Planning with Limited Adaptivity](https://proceedings.mlr.press/v238/tan24a.html)：在未知环境中，信息收集路径可按有限轮反馈重新规划。这里借鉴有限测量批次和反馈后重新调度；其子模信息目标、假设和近似保证没有被证明适用于本题，因此不移植其理论比值。
2. Golovin、Krause，[Adaptive Submodularity](https://arxiv.org/abs/1003.3967)：在满足自适应子模性等条件时，贪心可有竞争保证。R15未建立可信概率分布或验证该性质，只做预算受限的机会测量；既不称信息贪心最优，也不把未知源密度当作已知概率。
3. [Adaptive Informative Path Planning with Multimodal Sensing](https://ojs.aaai.org/index.php/ICAPS/article/view/6645)：说明将感知选择和路径选择放在同一序贯决策框架的研究背景。R15仍沿用题目的测向与光学操作，没有实现该论文算法。

具体可检验假设：清除后已到达的位置可能提供先前覆盖路径没有提供的观测；零额外入程不意味着零费用，每个真实测量及换频都计入整局。可能收益来自更早发现、后续可见性证书利用新增负反馈、以及覆盖与清除顺序变化。对不足16源或当前点接收不到任何剩余源的场景，补扫可能成为纯损失，所以必须做新旧同案例的完整闭环验证，不能用代理信息增益代替实际总用时。
