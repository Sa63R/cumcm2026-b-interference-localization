# 首版最后两项有界RL对照与选定模型交接

仅归档`milestone-0755.tar.gz`和`milestone-0818.tar.gz`内已完成的训练与既有48局开发验证，不读取扩展筛选或最终评测记录，也不改变已注册名单。

两组从best002共同起点出发，seed9112037、场景180001..196384、各512×32=16384条新增轨迹，无BC，学习率1e-4、entropy0.005。强attention仅改为既有一层四头关系模块；GAE组保留v3/base/flat/MLP96，只把lambda从1改为0.95。ea1/de7源码差异为attention构造时的通用Torch状态隔离，实际MLP更新路径未改；此前关于首轮shuffle的过强因果推断已[勘误](attention_strong_initialization.md)。

| 最终模型 | 平均虚拟秒 | 比共同parent节省秒 [95% CI] | 比matched lambda1末端节省秒 [95% CI] | 采样轨迹 | 末更新墙钟秒 |
|---|---:|---:|---:|---:|---:|
| 强attention u512 | 3188.226 | -25.981 [-71.425,19.108] | -11.808 [-58.145,36.264] | 16384 | 2286.021 |
| GAE0.95 u512 | 3114.041 | 48.204 [-4.726,100.606] | 62.377 [19.880,104.957] | 16384 | 1476.829 |

共同parent均值3162.245秒，matched lambda1控制末端3176.419秒。以上配对区间使用原协议seed20260911、10000次Python random.choices。它们是同一训练种子上的开发场景波动，不是跨训练种子不确定性；数据已经用于开发和候选选择，不能作为独立最终测试保证。进程墙钟含并发负载，不代表GPU独占时长。

强attention的u147/u291/u421/u512均值都高于parent，未提供额外收益证据，本轮止于既定预算。GAE0.95对同预算lambda1有正向证据，但相对原parent的区间仍跨0。对rollout，GAE最终节省259.212秒（7.684%），95%区间[206.779,313.335]，45胜3负，全清48/48、零失败清除，P95比值0.947615，最坏局慢215.211秒；不能称每一局都改善。

GAE没有改变gamma=1或真实总时间回报，但改变优势估计和critic的lambda-return目标。较小lambda引入对critic的依赖，是偏差/方差取舍；没有无偏、单调优化或普遍成功保证，也没有仅凭验证均值就证明方差确实降低。[算法公式与边界](rl_algorithm_basis.md)明确区分了这些概念。

完整审计现覆盖18个有完整日志的训练试验、53个固定验证端点，共359168条策略采样及11968条额外paired baseline，继承成本、重复场景和BC分别记录。强attention的负结果和所有中间端点均保留在[训练审计](rl_training_audit/README.md)。

## 实际GAE权重Windows交付检查

实际文件为`../q3-v1-artifacts/milestone-0818/rl-de7d65b-git/results/rl/cold-gae095-001/ppo_000512.pt`，SHA256：

`8985b7fc8d1709c5a54bb1aaf8f21df66f56782119eb901163f723a03281734f`

[检查脚本](audit_gae095_windows.py)已在独立PyTorch2.9.1+cpu环境实际加载此文件。训练场景100121..100124全部清除、零失败、持证正常退出；单线程平均0.32251秒/局，首次模型载入0.08063秒，不含Torch导入。仅4局兼容性检查，不作模型排序或跨平台收益比较。

真实Q3 practice入口的dry-run在禁止HTTP客户端构造、socket连接和urllib发送的监控下通过，触发次数0。权重前后字节未变，未复制为新的final模型、未启动官方模拟器。文件及张量摘要、模式、4局记录和dry-run证据见[windows_gae095.json](windows_gae095.json)。

## 已选模型卡：rl-gae095-u512

这是已注册名单中最终选定的RL候选；本节交接算法和复现条件，不新增训练、不重选模型，也不提前填写尚待统一审计的最终统计。前面的48局数值仍只属于开发验证。

| 项目 | 固定值 |
|---|---|
| 算法源码提交 | `de7d65b60ec05e4d1098f7b5a2c9c7a99c392adb` |
| 冻结时间 | `2026-09-11T00:48:28.838027+00:00` |
| 算法/特征 | `q3-joint-scan-ppo-v3` / 60维候选特征、12维全局上下文 |
| 网络 | 62,882个参数，hidden96，MLP候选编码与mean/max集合汇聚，actor与critic |
| 动作/概率 | `base` / `flat`，部署使用CPU、deterministic argmax |
| 相对模型路径 | `results/rl/v1-candidates/rl-gae095-u512.pt` |
| 相对spec路径 | `research/v1-candidate-rl-gae095-u512.json` |
| 模型文件SHA256 | `8985b7fc8d1709c5a54bb1aaf8f21df66f56782119eb901163f723a03281734f` |
| 模型张量摘要 | `870283a657ad19febd7772d88bc344daaa07f10c4bc48064ae92c7aff6e9fb20` |
| 原始spec文件SHA256 | `93d1f5a90c3a9dda140a6a13811de230bd7ed10f7c1a123145fdc0653e2ee6a4` |
| 冻结记录文件SHA256 | `93dc2def82a83aea8d87ecf6a0112ef0177fe17d7503cc00de75735efd6132ae` |

网络每次在合法候选中选择“测点、频道、扫描是否中断、定位哪一个源、采用哪个探点、何时清除已获证源”。未知源扫描候选是固定七站×尚欠频道；已发现源的探点来自原基线探点、可行外包区域最小包围圆中心、相对首测方位的两侧点、当前位置与中心中点、当前位置，去除重复。原点20频道全扫不是硬规则。所选模型没有attention、group概率校正、axis扩展或v4覆盖负债特征，也不调用状态压缩规划器替actor选择。

安全与完备性机制保持几何规则：方位观测更新保守外包区域，网络直接选择的清除候选只能来自已确认近距离点或半径≤19.9米的外包圆安全位置；每源主动探测上限6次，超过后执行原有有界光学搜索，其逐点光学尝试不等于每次均有先验清除证书，所有成功或失败尝试均计费；决策上限256后用确定性基线完成剩余工作。完整七站频道账本或成功清除16个源提供完成证书，不使用网络置信度代替证书。网络不任意输出连续坐标，不学习光学搜索内部轨迹，固定兜底的实际时间全部计入回报。合法观测候选输入不含真值或场景seed；teacher标签可计算作诊断，但不进入forward，所选训练链辅助BC系数均为0。保证的适用范围和有限候选相对连续全局最优的差别见[算法依据](rl_algorithm_basis.md)。

## 57,344局训练继承链

下表同时由真实训练日志和选中checkpoint逐层`state.initialization.source_state`核对。`episodes`是本阶段参与更新的完整策略采样轨迹数，不是optimizer步数；最终checkpoint表面的16,384局不含继承成本。

| 阶段 | 源码 | 从何处初始化 | 更新×32 | 场景段（闭区间） | 训练seed | λ / lr / entropy |
|---|---|---|---:|---|---:|---|
| joint-cold-mlp-001 | d33074b | 随机初始化 | 512×32=16384 | 160001..176384 | 9112032 | 1 / 3e-4 / .01 |
| cold-finetune-ppo-001 | 20bb10d | 上行u512 | 384×32=12288 | 140001..152288 | 9112034 | 1 / 1e-4 / .005 |
| cold-finetune-ppo-002，即best002 | f912dc8 | 上行u384 | 384×32=12288 | 120001..132288 | 9112035 | 1 / 1e-4 / .005 |
| cold-gae095-001，即选中模型 | de7d65b | best002 u384 | 512×32=16384 | 180001..196384 | 9112037 | .95 / 1e-4 / .005 |

合计 **57,344条**，本链这四段不重叠，但其他独立研究试验使用过部分相同场景，不能称全研究57,344个首次使用世界。四阶段实际BC均为0，也没有paired REINFORCE继承：纯随机起点PPO→PPO→PPO→GAE0.95 PPO。第二阶段config保留默认`bc_episodes=128`，但`--initialize-from`将BC标记完成，日志没有BC阶段；不能把默认参数误算为128条专家示范。每次初始化只迁移权重，重新建optimizer和计数；它不是连续一个optimizer跑1792轮。

全部研究的“18个完整日志试验、53个固定验证端点”与这四阶段链不是同一计数。前者保存正负路线，后者解释选中权重成本。最终阶段原始父文件SHA为`53f5fe9b659a82659343a84c227aa49e85761f3ef32557ea003a21df34966ee9`；移植成便携Path字段的best002副本文件SHA可能不同，应同时核验张量一致性，不把便携导出当新训练。

训练以`r_t=-Δ虚拟秒/1000`及失败罚项优化真实总虚拟时间，γ固定为1。最后一阶段λ=.95同时改变GAE优势和critic的lambda-return目标，未改变真实回报、输入、动作集或部署argmax。它是依赖critic的偏差/方差对照，不能只称“降低方差且无偏”，更不保证每次更新改善。公式和比较区间见上文及算法依据。

## 离线恢复与本地研究评估

交接需要**自包含**`source-history.bundle`、原始选中模型、原始相对spec及冻结记录。开发时的`rl-de7d65b-git.bundle`只有增量，单独clone会缺ea1前置提交，不应当作完整交接包。已实际从包含de7全部祖先的临时自包含bundle离线恢复，保留LF，并核对恢复后的整个`identity`与注册记录逐值相同，包括源码、spec、模型和协议哈希。该临时检查证明恢复方法可用；最终全分支bundle的文件哈希由总交付清单给出，不冒充已经验证尚未生成的最终包字节。

以下PowerShell命令假设交付文件放在`source-history.bundle`和`selection-complete-0905/`，且当前`python`已是准备好的PyTorch CPU环境；若总包目录不同，只调整输入路径。目录`rl-selected-source`应尚不存在。

```powershell
git clone -c core.autocrlf=false --no-checkout source-history.bundle rl-selected-source
git -C rl-selected-source checkout --detach de7d65b60ec05e4d1098f7b5a2c9c7a99c392adb
New-Item -ItemType Directory -Path rl-selected-source/results/rl/v1-candidates -Force | Out-Null
Copy-Item -LiteralPath selection-complete-0905/rl-de7d65b-git/results/rl/v1-candidates/rl-gae095-u512.pt -Destination rl-selected-source/results/rl/v1-candidates/rl-gae095-u512.pt
Copy-Item -LiteralPath selection-complete-0905/rl-de7d65b-git/research/v1-candidate-rl-gae095-u512.json -Destination rl-selected-source/research/v1-candidate-rl-gae095-u512.json
Set-Location rl-selected-source
$env:PYTHONPATH='src;.'
python experiments/research_v1_eval.py --help
python experiments/run_q3_practice.py --spec research/v1-candidate-rl-gae095-u512.json --dry-run
```

研究评估入口`experiments/research_v1_eval.py`完全使用本地仿真；它的CLI只暴露协议分区，因此这里仅检查`--help`，不借恢复检查重开封存集。对一个**已经使用过的训练seed100121**，可在恢复源码根目录复制执行：

```powershell
@'
import torch
from experiments.research_v1_eval import read_json, run_case
from simulation import random_scenario
torch.set_num_threads(1)
r = run_case(random_scenario(3, 100121), read_json("research/v1-candidate-rl-gae095-u512.json"), read_json("research/v1_protocol.json"))
print(r["row"])
assert r["row"]["successful"] and r["row"]["failed_clear_count"] == 0
'@ | python -
```

更完整的交接核验可以在任意目录运行`python check_selected_rl_offline.py --source <恢复源码目录> --freeze <rl-gae095-u512.json冻结记录>`。此[独立小脚本](check_selected_rl_offline.py)不属于冻结算法，只核验原始identity、模型、继承链，调用同一`run_case`复跑100121，并在禁止客户端/socket/urllib的条件下验证practice dry-run。[实测记录](selected_rl_restore.json)：全部哈希相等，62,882参数，57,344局继承链，11/11源清除、零失败清除、持证正常退出、网络触发0次；模型原字节未变。它不提供新的算法收益或正式验证结论。

最低包声明是Python≥3.10、NumPy≥1.24、PyTorch≥2.3；本次实际训练环境是Ubuntu/Python3.11.14、NumPy2.4.3、PyTorch2.9.1+cu128、RTX4060 Laptop GPU，4个CPU采样worker、每worker1线程，GPU批量更新。实际Windows恢复环境是Python3.12.14、NumPy2.2.6、PyTorch2.9.1+cpu、单线程。离线运行须事先准备相应依赖/离线wheel，Git bundle不包含Python或PyTorch；无需为推理安装CUDA、matplotlib、SciPy或模拟器GUI。跨Python、NumPy、OS或CUDA内核的浮点/采样和进程调度差异可能影响训练轨迹、墙钟及边界greedy选择，因此同seed不是跨平台逐位训练保证；Windows四局证据和这次LF恢复一局证据不能替代Linux统一统计。

## 准确训练命令（记录，不在交接检查中执行）

以下Bash命令在**新建的离线恢复副本**中复现四阶段设置，不能在正在工作的分支切换提交。`python`须指向上述CUDA训练环境；输出必须是新目录。它们保留原截止时间`2026-09-11T06:00:00+00:00`（北京时间14:00），历史时间经过后会按原保护停止；未来重新研究需另记新的截止参数和试验身份，不能修改冻结协议或覆盖选中模型。各阶段预算达到即止，不保证重新训练得到同一文件SHA。

```bash
export PYTHONPATH=src:.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
training_dir="$(pwd)/results/reproduce-selected"
common=(--feature-version v3 --architecture mlp --device cuda --hidden 96
  --episodes-per-update 32 --workers 4 --num-threads 1 --epochs 4 --minibatch 128
  --clip .2 --value-coef .5 --aux-bc-coef 0 --target-kl .03 --max-grad-norm .5
  --max-decisions 256 --checkpoint-seconds 600 --deadline-utc 2026-09-11T06:00:00+00:00)

git checkout --detach d33074b
python -m research_rl.train "${common[@]}" --output "$training_dir/cold" --seed 9112032 --scenario-start 160001 --bc-episodes 0 --bc-epochs 20 --updates 512 --lr 3e-4 --entropy-coef .01 --gae-lambda 1 --max-wall-s 1800
git checkout --detach 20bb10d
python -m research_rl.train "${common[@]}" --output "$training_dir/fine1" --initialize-from "$training_dir/cold/ppo_000512.pt" --seed 9112034 --scenario-start 140001 --bc-episodes 128 --bc-epochs 20 --updates 384 --lr 1e-4 --entropy-coef .005 --gae-lambda 1 --max-wall-s 1800
git checkout --detach f912dc8
python -m research_rl.train "${common[@]}" --output "$training_dir/fine2" --initialize-from "$training_dir/fine1/ppo_000384.pt" --seed 9112035 --scenario-start 120001 --bc-episodes 0 --bc-epochs 0 --updates 384 --lr 1e-4 --entropy-coef .005 --gae-lambda 1 --max-wall-s 2400
git checkout --detach de7d65b60ec05e4d1098f7b5a2c9c7a99c392adb
python -m research_rl.train "${common[@]}" --output "$training_dir/gae095" --initialize-from "$training_dir/fine2/ppo_000384.pt" --seed 9112037 --scenario-start 180001 --bc-episodes 0 --bc-epochs 0 --updates 512 --lr 1e-4 --entropy-coef .005 --gae-lambda .95 --group-alpha 0 --probe-candidates base --max-wall-s 2400
```

只复现最后阶段时，保持de7源码，将`--initialize-from`指向原始best002 u384（其SHA见前文），执行最后一行即可；不应改用最终GAE权重当起点，也不应把`--initialize-from`换成`--resume`。命令依据四份归档config及真实日志，未在本次交接中重新训练。原始路径、配置、日志摘要和阶段终止证据均保留在[训练审计](rl_training_audit/audit.json)。
