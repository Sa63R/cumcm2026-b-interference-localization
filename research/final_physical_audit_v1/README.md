# 最终比较的独立物理、几何与下界深审入口

**后续最终交付：** 全部 1136 条冻结最终记录的审计已完成，结果、条件下界差距与复算入口见 [FINAL_RESULTS.md](FINAL_RESULTS.md)。下文“尚未执行最终命令”的表述保留为工具准备阶段的历史说明。

这是审计工具准备与已打开开发记录验证，不是最终测试结果，也不产生新仿真。源码：[`experiments/research_v1_physical_audit.py`](../../experiments/research_v1_physical_audit.py)。与 state 分支 `experiments/research_v1_selection.py` 的身份/归档审查相接；后者负责冻结配置、完整种子、归档 SHA、同场景真值身份与选择规则，本工具负责逐物理动作、观测前缀证书和下界。它不改策略、不训练、不导入模拟器或场景生成器。

## 审计的事实与证明层次

1. **物理重算**：从 `/enter` 到 `/exit` 的真实 `history` 重算每次移动距离/5、测量 5 秒、测量换频 1 秒、光学尝试 3 秒、成功去除另 2 秒；逐动作虚拟时间和各费用项均需一致。仅此阶段读取明确标注为策略终止后的真值，重核 case SHA、Q3 源数/半径/位置规则、每次真实反馈和清除范围。复用已审核并固定 SHA256 的 `audit_eval_bounds.audit_record`，没有调用仿真器来给自己出相同答案。
2. **清除前置证书**：另一段仅用物理响应及其之前历史，重建 128 边外接多边形、1.005° 保守角域、1500 米正反馈接收界和同一 R 的正负距离半平面。若当前清除点到所有候选顶点距离不超过 `20−1e−5` 米，或到已观察 near 点的距离加 5 米不超过该阈值，即得到充分证书。真实成功不自动成为“预先保证成功”；未证书的光学尝试单独计数，实际失败次数仍由物理账本决定。若轨迹阶段明确声称 `certified_clear` / `near_clear` 而证明不能重建，报审计失败。
3. **静默推断与 RL 扫描**：state 每条 `inferred_no_signal` 必须指向真实动作前缀，且在该前缀的保守区域上独立验证 `d(query,C)>1500+1e−5`，随后才能作为逻辑约束更新区域。它不增加物理动作、费用或未知频道覆盖点。RL 与几何策略的扫描不靠策略内部任务标签识别，只按每频道真实测量响应记账，因此扫描可分散到多个动作、不同顺序或共享位置。
4. **终止/absence 证书**：每个尚未实际成功清除的频道，要用它自己的真实 no-signal 测点的 1000 米排除圆，以及真实失败 clear 的 20 米排除圆，覆盖完整 1800 米场地圆盘；不能把其它频道测过的位置借给它。已发现但未清除的频道不能通过终止证明。唯一免除其余频道覆盖的分支是 **16 次不同频道的真实成功清除**，不是 16 次检测。
5. **下界**：source-edge 子集 DP 求的是减去首尾清除半径后的有限边图最短路；它是连续清除圆盘访问路线的下界，不是连续路线精确解。原自洽信息认证界保持原公式与 N=16 特例，移动/认证长度取已有组合最大值，避免重复计费。按真实物理动作数扣 `0.5 微秒 × 动作数 + 1 微秒`，保留逐例比值。未成功或无法独立认证的记录不产生成功比值，实际失败记录仍保留 360000 秒罚后时间。

观察到本轮全部成功或零失败清除，均不等于已经证明策略对所有合法场景成立。当前原下界与可选增强仍保留“对全部合法场景认证”的条件，不宣称达到最优或再无改进空间，也不逐局叠加理想随机观测模型的期望信息下界。

### 不等半径圆盘并的有限证书

相同的 1000 米排除圆直接复用已审查 `planning.disk_cover.disk_cover_radius`，验证最远最近站距离小于 `1000−1e−5`。若还含失败清除的 20 米圆，使用

$$F(x)=\min_i\{\lVert x-p_i\rVert^2-r_i^2\}.$$

圆盘并覆盖场地等价于场地上 $F$ 的最大值不大于 0。两个候选二次式之差为仿射函数，划分成凸 power cells；每格内是凸二次式，最大值可取在被场地裁剪的格点或场地圆弧极值。枚举三条等功率线的交点、两圆根轴与场地圆周交点、每个圆心相对原点的圆周反向点，以及基准圆周点和原点，形成这些位置的超集，再对各候选取真正的最小功率值即可。先将每个排除半径缩小 `1e−5` 米，混合分支还要求最大功率值不超过 `−1e−5 m²`。

这是带显式余量的浮点几何证书，不是有理数或区间算术的形式化证明。若充分证书无法数值确认，则报告未证；不会把“算法没证出来”直接推断为存在真实未清除源。单独测试覆盖内部孔洞，而不只检查外圆周。

## 已打开开发数据的验证

- 9 项测试全部通过：混合半径/重合圆心/内部孔洞、与原等半径 Voronoi 值一致、前置 clear 真假证书、16 检测与 16 清除区分、逐频道实际覆盖缺一拒绝、inferred 前缀校验且零覆盖信用、封存入口默认拒绝、真实失败短轨迹保留 360000 秒且无成功比值、空批拒绝/缺档保留错误。
- 开发 6000–6047：rollout、state inferred、RL PPO384、geometric single 共 192 条，全部逐步费用/前置清除/终止证书通过；2444 次成功 clear 全有可重建前置证书，state 共 100 条静默推断合法。另对已打开 state relocation 与 geometric relocation 共 96 条检查实际移动覆盖点，全部通过。
- 全部 288 条复用原同平台 48 场景缓存，**288 命中、0 次新 DP**。源码冻结后的实际耗时记录在两个 `*_audit.json`，初轮约 12.4 + 5.9 秒。旧缓存 48 次 DP 合计 8.77 秒、单例最大 1.28 秒，仅供计算量级参考，不是未来机器/场景的耗时保证。
- 输入清单、每个 gzip SHA、几何依赖源码 SHA、固定理论模块 SHA 和逐例审计结果均已保存。这里的六组是兼容性测试，不重新比较/选择候选。

## 调用方法

请从 `q3-geometric` 工作目录的全新 Python 进程调用，避免跨工作树已经导入的几何模块混用。开发清单是显式 `{label,path,sha256}` 列表；默认仅允许已打开的 6000–6047 和授权训练压力 113001–113112，不递归扫描资料目录。

```powershell
python -m experiments.research_v1_physical_audit --inputs research/final_physical_audit_v1/development_inputs.json --output YOUR_NEW_DEVELOPMENT_AUDIT.json --cache YOUR_SHARED_GEOMETRY_CACHE.json --seed-cache research/final_physical_audit_v1/geometry_cache.json --cited-six-disk
```

将来冻结选择及全部最终运行完成后，先运行 state 的 `audit-final` 生成 `kind=final_identity_archive_audit` 的身份审计 JSON，再把它直接传给本工具：

```powershell
python -m experiments.research_v1_physical_audit --inputs YOUR_FINAL_IDENTITY_AUDIT.json --output YOUR_FINAL_PHYSICAL_AUDIT.json --cache YOUR_SHARED_FINAL_GEOMETRY_CACHE.json --seed-cache research/final_physical_audit_v1/geometry_cache.json --allow-heldout --cited-six-disk
```

**上面的最终命令只是交付入口，本轮没有执行。** `--allow-heldout` 只接受通过完整 SHA 校验的独立最终身份审计，不能用于任意文件列表；从其 `partitions.*.evaluations.*.evidence.archives_sha256` 取得明确档案，并再次核对压缩文件字节。全套四方法的 256 随机 + 28 压力共 1136 条可一次处理，按同一几何键缓存，新的 284 个场景至多各计算一次 DP；若压力几何重复还会进一步复用。缓存每次新增后落盘，重复执行可接续；审计输出采用独占新建，避免覆盖既有结论。可去掉 `--cited-six-disk` 只保留原始自洽界。

理论帮助模块默认位于同级 `q3-state-search/research/theory_v1`；在 Linux 工具克隆中可用 `--theory-dir PATH_TO_FROZEN_STATE/research/theory_v1` 显式指定。四个模块的 SHA256 必须与源码中的固定值一致，不默默复制或接受修改后的依赖。保存的开发输入清单含本机绝对路径；迁移时另写指向同 SHA 档案的新清单，保留原清单不改。九项测试中的两个批量包装测试也使用该外部帮助目录，其中早停构造来自本机已经打开的 baseline case 6000；换机器运行时需提供相同开发档案路径或只运行其余纯合成单元测试。

可选六圆增强依赖外部已发表定理：N<16 时在旧空频道动作项增加 `3×(20−N)` 秒，N=16 不变，原舍入修正保留。其来源、证据等级及为何不是通用 35/36 秒界，见 state 分支 `research/theory_v1/SIX_DISK_COVER_THEOREM_AUDIT.md`、`CITED_SIX_DISK_AUDIT.md` 与 `audit_cited_six_disk.py`；本工具不重新声称已亲阅该定理的完整原始证明。

## 尚不由本工具证明的事项

完整冻结名单、跨方法同 case identity 和选择过程由上一层审计负责；均值/P95、罚后比较、置信区间和最终通过标准由共同报告负责。费用重放不是对墙钟、网络、训练数据独立性或官方未知分布的证明。未预先证书的光学尝试可以合法发生，审计会记录它，并由最终接受规则分别决定 baseline 或候选是否容许真实失败清除。真实反馈与费用完整不等于几何代理是精确后验，更不等于原 Q3 全局最优。
