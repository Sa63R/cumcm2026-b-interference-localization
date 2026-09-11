# Round 3：当前强化学习对照只读盘点

结论：把 **CPU v2 trial-1，SHA256 `3d21ba7931143f281d2e7554261b001867c65fec9c2bd7465a79f91e95b7cef4`** 冻结为本轮 RL 对照。它是此前明确选择、已有跨策略同世界评估、目前可完整复现的模型；后续分支截至本次本地记录快照没有完成足以替换它的升级证据。这个结论是“当前证据最充分的已选对照”，不是证明它优于所有未完成或未评估的检查点。

盘点日期：2026-09-12。只读工作区中的选择记录、完整评估摘要、源码、模型哈希及对象存储回传清单；没有运行策略、训练、官方接口、新场景或读取验证 SQLite。没有修改其他线程的 RL 源码及未提交研究记录，也没有主动刷新远端训练状态。

## 1. 精确模型、源码与调用

本地模型已存在并重新核验：

```text
<workspace>/q3-deep-rl-cpu-v2\handoff\remote-status-20260911T125320Z\run01\trial-1\latest.pt
bytes: 810358
SHA256: 3d21ba7931143f281d2e7554261b001867c65fec9c2bd7465a79f91e95b7cef4
```

虽然文件名叫 `latest.pt`，本轮应以这个固定哈希为准，不能随着另一个训练目录更新而自动换模型。它是 v3、MLP96、base 动作集、flat 分布、60 维候选特征和 12 维 context 的 PPO 模型。不要改成 range/anypoint/recurrent 模式去加载同一文件，或悄悄用初始化后的新检查点替代。

推理源码使用 `q3-deep-rl-cpu-v2` 的原冻结版本：

- 历史 48 对评估的源码提交：`4e2ba11b9cebdaf8adb92ec76f0ddfb36ed43851`。
- 当前分支 HEAD：`35b2dd3758a0e2575718e22bd657aeac61a5804d`，后续是配对证据归档；本次重新比较 manifest 中 38 个源码/评估文件，全部仍与冻结哈希一致。
- 完整源码包：`q3-deep-rl-cpu-v2/results/paired_rl_state_20260911/rl_trial1-source.zip`。
- 源码包 SHA256：`fc3592fb1f8012130c684eef8ca87f8ac7c0c37bb3e11c5804374231873372de`，本次核验一致。
- 冻结 spec 位于同目录 `manifest.json → policies.rl_trial1.spec`，不是一个必须另找的独立 spec 文件。

入口是 `research_rl:run_rl_search`。以下完整显式 spec 保持原推理默认语义，checkpoint 路径可在本轮独立归档中改为相同哈希文件的实际路径：

```json
{
  "name": "rl_cpu_v2_trial1_frozen",
  "entrypoint": "research_rl:run_rl_search",
  "kwargs": {
    "checkpoint": "<workspace>/q3-deep-rl-cpu-v2/handoff/remote-status-20260911T125320Z/run01/trial-1/latest.pt",
    "device": "cpu",
    "num_threads": 1,
    "deterministic": true,
    "max_decisions": 256,
    "max_active_probes": 6
  }
}
```

在独立进程中，将该源码树的 `src` 放在导入路径首位，由统一运行器提供 `problem=3,max_actions=10000` 和只读观测客户端，然后调用该入口。它返回原 `SearchReport`，可 `as_dict()`；有 `action_history`、计费分项、清除/发现/退出信息及 `learning` 日志。不要传状态搜索专属 `config`。入口自身默认 `max_actions=20000`，所以必须由统一运行器显式覆盖为共同的 10000；不能遗漏而使 RL 获得不同动作预算。

本地可用运行时已只读确认：

```text
<workspace>/q3-deep-rl\.venv-win\Scripts\python.exe
PyTorch 2.9.1+cpu; NumPy 2.2.6; torch.version.cuda is None
```

这次仅导入包查看版本，没有运行网络或环境。状态搜索常用的 `cumcm2026-b-interference-localization/.venv-win` 有 NumPy，但没有 PyTorch，不能直接据其缺包宣称模型不可用。公平实验可让各策略均使用上述 CPU 环境、同一台机器、各自独立进程；单策略 Torch 线程设为 1，整体并行度由总任务统一限制。

## 2. 为什么选这个模型，而不是最新训练端点

`q3-rl-autonomy/handoff/goal-start-remote/run01/summary.json` 记录原三个 CPU 训练种子全部完成、评估全部完成。`research/autonomy/protocol.json` 在新实验开始前明确选择原开发集表现最好的 trial-1 作为共同初始化，并锁定同一个 `3d21...` 哈希。选择发生在后续状态/RL对比以前，不把这些开发结果误称为独立最终检验。

后续分支的现有证据如下。下表同一组 64 个合成开发场景使用共同旧物理 LB 均值 **1702.288478 秒**，所有列出的端点均为 64/64 全清、0 次失败清除。比值采用 `sum(T)/sum(LB)`，不是逐局比值均值。

|64 局开发记录|平均 T / 秒|T/LB 总量比|目前选择意义|
|---|---:|---:|---|
|冻结 derived-silence 状态对照|3075.648804|1.806773|本次 RL 研究使用的已确认状态参考|
|原 trial-1，未追加本轮训练|3097.524141|1.819624|继续保留的 RL 父模型|
|base-r1 继续 PPO 最终端点|3145.353549|1.847721|未升级；比父模型平均慢 47.829408 秒|
|memory-r1 GRU 最终端点|3119.389138|1.832468|未升级；比父模型平均慢 21.864998 秒|
|cost-r1 第一块完整端点|3796.097844|2.229997|明显退步；比父模型平均慢 698.573703 秒|

原 trial-1 相对该状态参考平均慢 21.875337 秒，配对“节省”95%区间 `[-71.031050,27.715813]`，跨零；不能据此宣称一方平均可靠胜出。RL 的该批 P95/最大值更低，这也不能自动弥补均值证据不足。

cost-r1 本地记录在盘点期间更新：第一块 64 局全部比父模型慢，第二块开始后被所属线程停止并归档部分进程；不算完成一小时的等预算对照。`cost_r1_failed_endpoint.json` 与 `COST_TO_GO.md` 明确不替换原模型，并说明后续转向独立 range 试验。它们当时仍属该线程未提交文件，不能把分支 HEAD 或目录的更新时间当作新模型已验证升级。

各研究树本次观察到的 HEAD 与状态：

|工作树|HEAD|本次读到的状态|
|---|---|---|
|q3-deep-rl-cpu-v2|35b2dd3758a0e2575718e22bd657aeac61a5804d|保留已选 trial-1；完整 48 对配对证据|
|q3-rl-range-probes|9e89e15d9b468891f2b8903d536b93db2b746cb5|新增半程横向动作的接入测试与负向小烟测；后续独立训练无已读升级结果|
|q3-rl-anypoint-scan|0dc4387f2930655af2f018e3ecd5d533e4dfc2d0|新任意位置发现动作、机制审计；没有已读独立性能胜出记录|
|q3-rl-recurrent|6603c7c6d4fbc0a786890b0f15b02db07b1f6b21|GRU 机制实现；实际 memory-r1 端点归档见 autonomy，未升级|
|q3-rl-autonomy|5d3e8d3f0a91747f1109a03c43fdb32e100fa990|base/memory 的完成端点及继续保留父模型的决定|
|q3-rl-cost-to-go|e50ba07762c141691b0f8773a889acf2319b8221|完整续跑费用学习；当前失败端点与停止记录另有未提交更新|

这里只认已读的完整评估与明确选择，不断言其他线程以后不会产生更好模型。新的已验证升级应通过明确交接重新冻结，不能悄悄更换本轮对照。

## 3. 已有可复核的同平台跨策略比较

在本地 Windows 使用共同引擎的 48 对旧开发比较，场景为 `2100001..2100048`，旧 LB 均值 **1746.245216 秒**：

|冻结方法|平均 T / 秒|sum(T)/sum(LB)|全清 / 失败清除|
|---|---:|---:|---|
|当时已确认 certified-tail 状态版|3101.981733|1.776372|48/48，0|
|CPU trial-1|3110.201814|1.781080|48/48，0|

RL 平均慢 8.220081 秒，配对节省区间约 `[-58.8718,41.2131]`，证据不足以证明谁平均更好。该状态对照早于 derived-silence 的最终交付；新轮应使用当前已确认的 `760af823...` 及其冻结 combined spec，同时可保留 v1 辅助参考，不能用不同批次均值拼接比较。

原远端评估使用 `4e2ba11+local-cpu-acceleration`，其中几何代码有加速改动。训练来源必须保留这项说明，不能声称 checkpoint 的全部训练代码就是未经修改的 Git commit。48 对本地评估则明确使用原共同几何实现重新推理；历史记录显示跨平台逐局会漂移，平均差很小也不意味着逐动作相同。上述本地源码包是本轮复用的**推理**身份，不伪装成完整原训练环境。

## 4. 与受保护状态引擎的兼容性及必要适配

本次直接计算哈希，`q3-state-search`、`q3-deep-rl-cpu-v2`、`q3-rl-autonomy`、`q3-rl-cost-to-go` 的以下核心文件目前逐字节相同：

|文件|SHA256|
|---|---|
|src/simulation/engine.py|3ef36f508f773564baed47569e014309cfb1cbcebb1ba1ad268a4b0f3e125622|
|src/simulation/cases.py|4d5588d9c11ccda5f251a819b292d9d69f1d5f5f9580be20534aa3bb9b6eec39|
|src/simulator_client/client.py|441230d1f2bb231a7a64beaa119306d89b4535ffaf1f4393c8be9c4d08a81577|
|src/simulator_client/state.py|7a64a2a871532f76613000994bd850e86148a28f72f415d017fa56553815bb42|
|experiments/research_v1_eval.py|b1c93f0ffcb23970b7ea8c5981e36d62e680514b14d020fe06b32e05d18d796a|

**因此存在无需修改 RL 策略的同场景配对路径。** 必要工作在独立评估接入层，而非把 RL 改写成状态调度：

1. 统一生成或装载同一个场景对象，冻结其完整 case SHA。各方法在独立进程重新创建同一个物理世界；不仅比较 seed。浮点坐标末位进入固定误差哈希，不能把跨平台“同 seed”当作逐字节相同世界。
2. 使用共同 CPU 环境、共同几何/引擎/协议、相同虚拟预算、实际时间预算和动作上限。历史配对为 `10000 actions / 360000 virtual seconds / 300 real seconds`；本轮若采用更长统一 real budget，应全部策略一致并写入新 manifest，不冒称原复跑条件完全相同。
3. 入口只接观察包装客户端；隐藏真值只在完成退出后交给审核器和旧 LB 计算。原验证 SQLite 不用于推断网络、拟合分布或构造反事实反馈。
4. 归档模型内容和 SHA，不只归档代码；运行前后检查 checkpoint 与源码身份。每种策略单独 `src` 导入路径，避免共享进程的同名 `geometry/strategies/research_rl` 模块污染。
5. 保留 RL 的真实单频道覆盖动作及逐点逐频道账本。RL v3 没有强制整站扫描；不能按状态算法的“完成一个 scan 宏”来归类其发现费用或终止证明。原公共前缀审核按真实负测量圆盘和成功清除重建，应从已打开旧记录做接入核验，而非放宽规则使 RL 通过。
6. 统一行协议可沿用 `row/summary/history/evaluation/evaluation_phase`，报告 `learning` 和完整 action_history，计入256个神经决策用尽后的既定完成策略费用。不能只给网络控制阶段计时，把后续兜底移到计费之外。
7. 冻结两个方法后再分配双方未用于训练/选型的新评估案例，逐局报告旧 LB、T/LB、成功、失败清除、检测/换频/移动/清除费用和 CPU；保留 `mean(T/LB)` 与 `sum(T)/sum(LB)` 两种聚合。

已有 RL 分区需避让：CPU v2 训练 `1000001..1599999`（三段）、测速 `1900001..1900384`、开发 `2100001..2100048`、预留测试 `2200001..2200256`；autonomy 训练三段合计 `3100001..3999999`，测速 `3000001..3000384`，开发 `5300001..5300064`，确认 `5400001..5400128`，最终 `5500001..5500256`，压力 `5600001..5600056`。这只是读到的分区登记，不能仅凭本表认定某个预留区还未打开；启动前仍应由主任务与 RL 线程统一登记实际占用。

## 5. 本地与对象存储可用性

本地 checkpoint 和完整原推理源码包均已核验，可直接在本机用于独立接入，不需要等待远端训练结束。

对象存储历史交换前缀有明确记录：

```text
jiangsu10:bucket-c20250204-pool01/lianghao/bwc/shumo/q3-cpu-v2-20260911-r1
```

已回传并验证的 manifest 中，该模型对象键为：

```text
live/objects/3d21ba7931143f281d2e7554261b001867c65fec9c2bd7465a79f91e95b7cef4
```

清单将其绑定为 `run01/trial-1/latest.pt`、810358 bytes、同 SHA。清单记录模型采集于 `20260911T122512.749015Z`；所读 manifest 为 `20260911T125300.041102Z`，它的 `final_sync=false` 只说明这是逐文件阶段快照，不能用该标记说三个试验尚未完成；另一个更晚的 `goal-start-remote/run01/summary.json` 已明确记录三个全部完成。

这是**对象曾发布、已下载并核验**的可用性证据，本次没有发 live HEAD 请求，不声称此刻远端对象仍在线。若要恢复远端副本，按上述内容地址走对象存储并核验同 SHA；禁止通过 SSH/SCP 发送文件。无需把任何凭据加入本轮 spec、报告或 Git。

## 6. 本次主要证据路径

- `q3-deep-rl-cpu-v2/results/paired_rl_state_20260911/{manifest.json,README.md,comparison.json,rl_trial1-source.zip}`。
- `q3-deep-rl-cpu-v2/handoff/remote-status-20260911T125320Z/{manifest.json,LATEST.json,run01/trial-1/config.json}` 和上面的实际模型。
- `q3-rl-autonomy/handoff/goal-start-remote/run01/summary.json`，SHA `a9eb05b27f80d524df04f4ccf6f954794a85df6d00f9b32067b2e5140c0fa246`。
- `q3-rl-autonomy/research/autonomy/{protocol.json,EXECUTION.md,base_r1_final_snapshot.json,memory_r1_final_snapshot.json}`。
- `q3-rl-cost-to-go/handoff/autonomy-status-20260911T175422.828772Z-a4290aa77f04/cost-r1/summary.json`。
- `q3-rl-cost-to-go/research/autonomy/{COST_TO_GO.md,cost_r1_failed_endpoint.json}`：本次观察到所属线程正在归档，按快照解释，不据未提交状态进行任何修改。

本轮可直接采用的决策是：**固定 trial-1 作为 RL 参考臂，固定当前独立验证状态版作为主要状态参考，评估器统一物理环境和分母；不等待、也不自动追随另一线程的最新训练文件。**
