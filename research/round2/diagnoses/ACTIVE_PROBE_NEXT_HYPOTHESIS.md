# 主动定位移动成本：只读诊断与一个有限的新假设

本诊断只使用此前已经打开的 32 份基准历史（feedback/pilot 的 200101–200116，以及 negative_hull/pilot 的 203101–203116）和冻结基准 8c624d0 的公开决策源码。未读 SQLite、未开启模拟器、未运行新策略或新案例。以下候选比较只是对同一历史前缀重算无副作用的有限几何评分；没有向任何候选提供新观测或隐藏真值。

## 1. 40% 的账单不等于 40% 的绕路

旧 32 局均完整清除、失败清除 0 次。平均整局虚拟时间 3164.974209 s，旧口径物理先知下界均值 1788.654831 s；逐局 T/LB 的均值 1.782543319，sum(T)/sum(LB)=1.769471759。这里的 LB 知道源位置，是原来的物理先知下界，不是在线可达到的时间承诺。

| 主动定位阶段 | 实际测量数 | 移动均摊到整局（s） |
|---|---:|---:|
| 每段第一次主动测量 | 336 | 787.443853 |
| 同一源后续主动测量 | 213 | 476.569192 |
| 合计 | 549 | 1264.013044 |

主动定位移动占整局 39.94%，但 `_resolve(channel)` 是原子地定位并清除此源：第一次主动测量常同时承担从上一任务赶往目标邻域的路程。按真实前缀划分的 336 个完整定位段分别有 170/120/45/1 段需要 1/2/3/4 次主动测量。

在每段记录的起点、主动测量点和最终实际清除点之间计算折线长度：主动测量与最后清除合计移动均值 1387.948750 s/局（其中最后清除段 123.935706 s/局）。直接把该段起点连接到**事后已知的实际清除终点**，均值为 1316.406948 s/局，二者差仅 71.541802 s/局，即该折线移动的约 5.15%。这表明多数主动移动在前往最后清除位置；把全部 1264 s 称为探测浪费会严重高估空间。

这个 71.54 s 只是“固定实际端点”的事后折线差：在线还不知道该终点，直接去那里未必有清除证书；改变测量会改变后续端点和任务顺序。它既不是能兑现的收益，也不是整局更强下界或所有策略的优化上限。

从几何上，源仍在一个较长可行区域内时，一次方位测量往往不能获得半径 20 m 的清除证书，需要改变视角。原算法还要求候选对整个可行多边形保证接收，因此当前点过远时必须移动。重复在相同位置测同一频道不能获得独立误差样本，不能把重复测量当作免费提高精度的方法。

## 2. 当前选择器实际优化什么

`RefinedStateSearch._next_probe` 先在原 9 个点中选择，再保留全部原点并加入 14 个 axis-quantile 点，仍使用 `choose_radius_probe` 的同一个有限代理评分。原点是中心以及从当前位置到中心的半程/全程锚点，分别横向偏移 ±50/±150 m；附加点沿可行区域长轴四个分位点及横向偏移展开。

候选按待测频道的六位小数位置键去重，且对所有可行区域顶点验证距离不大于 1000 m。用中心及少数内部几何点作名义源，模拟零误差方位约束后的包围圆。代理为到测量点的移动，加到名义源的距离、仍未能清除的 6 s 惩罚和剩余半径惩罚；分支定界只加速这个固定评分。它不是实际后验期望，也没有证明决定后续测量数。相同本频道的首测服务及切换费用对这些位置是常数，评分省略常数不表示这次动作免费。

已检查归档中的 clear_lens、feedback、negative_hull、terminal_cover、ACTIVE_POINT_SHARING、有限 probe tree 及几何候选实现。本次不再建议修改清除落点、源之间抢占/恢复、负点非凸区域、尾段覆盖或额外跨频道共享测量。

还检查过一个接收范围方向：利用此前真实正观测点 p 证明 q 对所有可行源都比 p 更近，从而为原来被 1000 m 条件拒绝的候选提供接收证明。在这 549 个前缀的现有候选家族内，满足这种新增证明的被拒候选为 0。故没有依据为此另开实验；这只是现有候选中的零激活结论，不是一般不可能性定理。

## 3. 唯一建议：主频道的当前位置候选

原 23 个构造点没有显式包含当前位置。若上一任务把机器人送到了对下一源有价值的位置，尚未在此测过该频道，却可能被现有构造强迫先走到另一个探测点。建议只把当前位置加入**当前已选主频道的第一次主动测量**候选，沿用原评分、原保守接收条件和同分规则；如果不合法或不胜出，走原选择。它是替换同一测量动作的位置，不是增加其他频道测量，也不改变任务调度。

对 549 个既有前缀，先按旧动作顺序重建已观测频道、可行区域、原推断约束和六位小数真实测量键，核对 probe 日志与真实测量点逐条对应。结果如下：

| 当前位置条件 | 前缀数 |
|---|---:|
| 当前点已测过该频道，禁止重复 | 412 |
| 未测但不能对整个可行域保证 1000 m 接收 | 55 |
| 未测且满足保守接收条件 | 82 |
| 加入后仍选原点 | 37 |
| 加入后选择当前位置 | 45 |

82 个合法前缀全部是该原子定位段的第一次主动测量。对这些前缀，原几何评分重算的原位置与分数均匹配历史记录（分数差小于 1e-6）。新增当前位置后，45 个选择变化分布在 23 局，名义评分下降总量均摊仅 2.538496 s/局。被替换首入段的旧移动时间均摊 67.662942 s/局，**不能称为预计节省**：在当前位置取得观测后，机器人仍要赶往源并完成清除，甚至可能多测一次。

不少胜出只是小半径区域中的不足 1 s 名义分数优势，说明这最多支撑一次便宜、隔离的小批配对检验，不支撑“大幅优化”的预测。此前失败的 ACTIVE_POINT_SHARING 会额外插入别的频道读数；本建议始终是已选主频道的原第一读数位置替换，额外无线开销与调度扰动的来源不同。在所检查的归档实现中没有发现该单机制的独立消融。

## 4. 可检验合同与停止标准

实现应保留原 9→23 两阶段过程；新增点只能加入第二阶段嵌套候选集，不能替换原点或改变原评分。仅主频道已发现未清除、首次主动测量、当前位置真实测量键新鲜、全部顶点在 1000 m 内时考虑；空/非有限/退化几何拒绝新机制。近源和已有清除证书仍由原 `_resolve` 处理；关闭开关必须完全恢复基准动作。日志需把公开动作前缀、原选择、接收/去重条件、是否真正改变选择及之后实际测量绑定，便于独立审核。

先用人工几何和既有旧案例检查合法性，再由冻结协议调度新的小批配对案例；不能拿上述 45 个前缀筛选新的性能案例。评估完整清除、失败清除数、整局 T/LB、平均与尾部虚拟用时、真实 CPU，并检查是否以多测或后续长路抵消首段免移动。不能把局部分数下降或首段零距离直接当作优化成功。如果小批显示没有可靠整局收益，归档该方向并停止，不围绕旧前缀反复调阈值。

目前仅找到这一个有具体代码缺口和可观测前缀证据的有限假设；没有证据声称接近理论极限，也没有证据把主动定位的全部 40% 成本消去。

## 5. 输入证据与复核范围

以下 SHA-256 绑定本次查看的输入。32 个原始 gzip 均再次核验与既有诊断 JSON 的 `input_sha256` 一致；只读取 `summary.action_history`、公开策略日志和原诊断费用/LB，未读取隐藏场景或答案作为候选评分。

- BASELINE_OBSERVATION_COST.json: `3f513037d4e5539cf92d1c5228ccca11025abfa109b230b62288ac7d777513f4`

| 输入 | SHA-256 |
|---|---|
| results/round2/feedback/pilot/records/baseline-200101.json.gz | `acc4bea1b1e31b8606ff6592b3c7b733736c18247ef61e2dbbe6b0818b361d8e` |
| results/round2/feedback/pilot/records/baseline-200102.json.gz | `e60b2623f48de11a5d833ddd6924a8db4a6d1d0c10b149377c4c8b7484ed3624` |
| results/round2/feedback/pilot/records/baseline-200103.json.gz | `707b2fca3f37176a8962706e647093fa661c80a1f45f905cac12ec0b7f648433` |
| results/round2/feedback/pilot/records/baseline-200104.json.gz | `c0f850963c0c1e0369934c1c5aa46443b18218ebd7fa84157a0fe99dadad812d` |
| results/round2/feedback/pilot/records/baseline-200105.json.gz | `e793deb94fb351989833928b03329ad12956c3133ce520969c6a880ce893f33d` |
| results/round2/feedback/pilot/records/baseline-200106.json.gz | `755f42ce0a44793bb53dc7b95f226bea3bf453101ca7dd87538bb02f913a1527` |
| results/round2/feedback/pilot/records/baseline-200107.json.gz | `1c1914f7e72c7fb583d80ac8aa996ea07fa19dce46307cba483b44ddc5e671af` |
| results/round2/feedback/pilot/records/baseline-200108.json.gz | `9c5f8e960d3843f9988f7289a5dff5a11bfe5bdc25adddff95888515d860ac5b` |
| results/round2/feedback/pilot/records/baseline-200109.json.gz | `a86f8287581d59d5c2fb7d59db35cc160a5ce2d0b3e59b18aa3ae73fd2cd9687` |
| results/round2/feedback/pilot/records/baseline-200110.json.gz | `5e6f6104b708d9bfb003f7932bd3921be9fd3b3ac2e6d03933dc394dfc11af05` |
| results/round2/feedback/pilot/records/baseline-200111.json.gz | `ad89ad8986a7422c7ba2ce26aa897f764eaf722911a2f6f08b004224c24a9d1b` |
| results/round2/feedback/pilot/records/baseline-200112.json.gz | `c5bc13d02c73915b66529467ca4f51f42114dcc5243b7b13ea44f9d2ff28e0b4` |
| results/round2/feedback/pilot/records/baseline-200113.json.gz | `97265f57ef3bbb7ba041d4ee7149cfd415ad48bbd93200280fef7c56cc4e7625` |
| results/round2/feedback/pilot/records/baseline-200114.json.gz | `478dce09450a19fe05f5973b4c97fca2a2f2b7e4b6073e63f76831f6cf9be5f6` |
| results/round2/feedback/pilot/records/baseline-200115.json.gz | `5bebb6faa151ee7ba82c6cb2be64dde66da832b9673afcf9d9c62bbadf3e1f19` |
| results/round2/feedback/pilot/records/baseline-200116.json.gz | `396b93ed1f145b6fd798c8d75948ab907b8982b5a7fb455edc3f16e81eac1c44` |
| results/round2/negative_hull/pilot/records/baseline-203101.json.gz | `119fe208db2402ecab790944779946ad3d4ba36c375987d4998dfea8760b41d5` |
| results/round2/negative_hull/pilot/records/baseline-203102.json.gz | `c0d1b937c7e67198e011593577931dee70dce4620464d7406259205278bd3839` |
| results/round2/negative_hull/pilot/records/baseline-203103.json.gz | `bc2188a721d532755d5970a360ecc21065f8ad77fc5e1bed66e6f7c1a901cb27` |
| results/round2/negative_hull/pilot/records/baseline-203104.json.gz | `3a75daa9b2694127ef3512721f9185061a819e69a232c93dbe9f06c50a3ff2c5` |
| results/round2/negative_hull/pilot/records/baseline-203105.json.gz | `989063c8ffc31987a20b7fa770a5ac17cce6f63bc54557d3aef430530bd048a8` |
| results/round2/negative_hull/pilot/records/baseline-203106.json.gz | `b9be8bb61dabaffc66505a4d11615adea3948c2ecfc01806e3687adc8bd8ef66` |
| results/round2/negative_hull/pilot/records/baseline-203107.json.gz | `45b44fc3a704f8307786b67531c98a0f33f092c08390791fb1f9a5420af4be1c` |
| results/round2/negative_hull/pilot/records/baseline-203108.json.gz | `aeab8132ac4c0e4d8a79673f0ec167046ad0fd2b2f8d8838d672954157d933de` |
| results/round2/negative_hull/pilot/records/baseline-203109.json.gz | `d47e9734c4a330ed2cedccd5066327be35642c4aef944a0a076aff66ee974c98` |
| results/round2/negative_hull/pilot/records/baseline-203110.json.gz | `d47f21a49a3729b26a430e5bcaae63361ef3ee15cb796594a7b2b3db0185e3e1` |
| results/round2/negative_hull/pilot/records/baseline-203111.json.gz | `31d827b28443b587d474460bc4ca590db4089eb2b2475e9cb05724b81400a036` |
| results/round2/negative_hull/pilot/records/baseline-203112.json.gz | `6fbccb9e42b6f2361824e703634bb833eff3be4a391b3f669b25bf2f9144c734` |
| results/round2/negative_hull/pilot/records/baseline-203113.json.gz | `d1a0c1f0c7321c9fe296f8047ef217927d2791af318c002c3a1c1c1de4f78fa5` |
| results/round2/negative_hull/pilot/records/baseline-203114.json.gz | `f0bc865b1876b1ad70979022943802c04b81cc0845dd8572e279ed354dd9d561` |
| results/round2/negative_hull/pilot/records/baseline-203115.json.gz | `675dc6c3eef92967791f5fcb1daf61b9ebb585f0ff115fe9cf228e1ff211c406` |
| results/round2/negative_hull/pilot/records/baseline-203116.json.gz | `161baf7a259f252c14f4a09c3ff0e6b2bd80495a815e60e7808550d95f617cb8` |

| 冻结基准源码 | SHA-256 |
|---|---|
| q3-state-search/src/planning/radius_probe.py | `e76e927f27e48d13311b8b19c0ff54e8f6e3f7a50788b105e948da8edcb812d4` |
| q3-state-search/src/planning/probe_candidates.py | `2b79f97f5dd25025c18cc0e78a740135b4fcfb8ab19df6a35305b80ec8516f71` |
| q3-state-search/src/strategies/refined_state_search.py | `d2dd58f6d6ee325692737626ec1d26aa93bf6ae02dd752f7b53c41aa66cda683` |
| q3-state-search/src/strategies/efficient.py | `2349996a10685d62c441f2ebf8ff176d18f37ced660eb260e4daf38cf6b1f37a` |
| q3-state-search/src/strategies/relocating_state_search.py | `1454a7286d0180d97ad6c3e2f218160ad628d152735005f1320335bdfdac8d6d` |
| q3-state-search/src/localization/omni.py | `35a999a1533a7c557315f719e2dcea91ffd3e1ec695636b94d374d4aaddc1e3b` |
