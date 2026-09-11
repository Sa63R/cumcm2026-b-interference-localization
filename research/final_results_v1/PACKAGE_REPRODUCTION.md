# 复现与演练交接

## 1. 原始结果已经齐全

原始 480 条筛选案例和 1,136 条最终案例均保留在 evidence 中，不需要重跑才能阅读或重新计算统计。候选及最终名单已经冻结，不能用本轮 final 数据继续选模型。下一版预留 900000–900999 未打开。

本机原目录仍是：`cumcm2026-b-interference-localization`（main）、`q3-state-search`、`q3-deep-rl`、`q3-geometric`。SSH 主机别名为 `gpu4060`，远端记录根目录为 `/home/volleyball/q3-research-v1`。无需重新训练便可使用选定 RL 模型。

## 2. 完全离线恢复源码

在交付目录中执行，输出必须是一个不存在的新目录：

```powershell
python restore_sources.py --verify-only
python restore_sources.py --output ../q3-v1-restored
```

要求 Git 与 Python 3.10 以上；恢复本身只用 Python 标准库。脚本从本地自包含 bundle 克隆完整对象，使用 LF 行尾，新建 detached worktree，恢复下面四个算法身份及两个原评估工具提交。它复制原配置和原模型字节，随后核对所有源码、配置、模型和协议 identity，与原冻结记录不一致即报错。不会覆盖现有目录、切换 main 或调用模拟器。

| 恢复目录 | 算法提交 | 原始 spec 路径 |
|---|---|---|
| baseline | `2f0a486` | `research/v1_baseline_rollout.json` |
| state | `8aa6206` | `experiments/state_search_candidate_relocating_cover_v1.json` |
| rl | `de7d65b` | `research/v1-candidate-rl-gae095-u512.json` |
| geo | `3c4f86f` | `research/v1_geometric_probe_relocation.json` |
| report-tools | `d5c082b` | 共同统计与冻结选择工具 |
| physical-tools | `ee10f6e` | 原物理／下界审计工具 |

当前研究分支上的后续提交包含报告、图、兼容入口和监督工具修正；不等于把实际最终算法身份改成这些新提交。完整 branch heads 另见 `source-heads.json`。恢复目录 `history` 保存这些分支的全部历史，可使用 `git show` 阅读提交时的资料。

选定 RL 模型为 `results/rl/v1-candidates/rl-gae095-u512.pt`，SHA256：

`8985b7fc8d1709c5a54bb1aaf8f21df66f56782119eb901163f723a03281734f`

## 3. 环境与离线兼容性

最终配对结果来自 Ubuntu、Python **3.11.14**、NumPy **2.4.3**、PyTorch **2.9.1+cu128**；完整安装包、CPU、GPU及线程信息在 `environment-linux.json`。GPU为 RTX 4060 Laptop，训练用 CUDA，最终推理及串行计时用 CPU。Windows兼容检查使用 Python 3.12.14、NumPy 2.2.6、PyTorch 2.9.1+cpu，不能混入 Linux 的逐场景配对结果。

基础策略无需第三方数值依赖；RL 需要 NumPy、PyTorch，绘图需要 Matplotlib。各源码树 `pyproject.toml` 保留 `rl`、`test`、`report` 可选依赖。已有环境可直接使用；新建环境若需复现最终数值，应优先匹配登记的 Linux 版本，而不是只安装最新依赖。

模型离线恢复检查脚本位于 RL 分支 `research/check_selected_rl_offline.py`，已从真正的离线 Git 克隆运行成功。交付包也复制到 `tools/`，恢复后可执行：

```powershell
python tools/check_selected_rl_offline.py --source ../q3-v1-restored/rl --freeze configs/rl/freeze.json
```

该检查只运行一个已经用过的训练种子 100121，并禁止 HTTP、socket 连接和官方客户端构造。它验证完整 identity、模型张量摘要、57,344 局继承链、清除与演练 dry-run，不是新一轮泛化测试。需要在安装了 PyTorch 与 NumPy 的 Python 中运行。

## 4. 从原始档案重算报告

将 `evidence/final-evaluations-complete-0930.tar.gz` 解压到一个新目录 `unpacked-final`。其内部为 `v1-selection/evaluations/final_random/` 和 `final_stress/`。报告程序只读已有档案，不重新运行策略：

```powershell
$cases = (Resolve-Path ./unpacked-final/v1-selection/evaluations/final_random).Path
python ../q3-v1-restored/report-tools/experiments/research_v1_report.py --baseline "$cases/baseline" --candidate "state=$cases/state-future-cover" --candidate "rl=$cases/rl-gae095-u512" --candidate "geo=$cases/geo-future-cover" --output ./recomputed-random --figures
$stressCases = (Resolve-Path ./unpacked-final/v1-selection/evaluations/final_stress).Path
python ../q3-v1-restored/report-tools/experiments/research_v1_report.py --baseline "$stressCases/baseline" --candidate "state=$stressCases/state-future-cover" --candidate "rl=$stressCases/rl-gae095-u512" --candidate "geo=$stressCases/geo-future-cover" --output ./recomputed-stress --figures
python ../q3-v1-restored/report-tools/experiments/research_v1_report.py --final-random-report ./recomputed-random/comparison.json --final-stress-report ./recomputed-stress/comparison.json --output ./recomputed-acceptance
```

`--figures` 需要 Matplotlib，也可去掉仅生成 JSON 和 Markdown。复制目录后记录中的路径、图形生成环境可能不同，不应要求重算输出文件所有字节等于原件；应比较完整精度指标、案例身份与统计规则。原始报告、图和文件 SHA 保留不改。

严格的最终 identity／physical 审计使用已登记的绝对路径与冻结源码，因此原执行命令、环境和监督记录保存在第三个 evidence 档案中。若迁移到另一目录重新运行全审计，应保留原输入另建立迁移映射，不能直接把原 registry 路径改掉后仍声称是原注册文件。

## 5. 之后接入官方演练

你需要在模拟器界面开启 **问题3演练测试**，并提供本次案例编号；队号已经保存为 `202627001104`。自动程序可以负责预检查、加载方案、发出操作、保存返回与分析结果。当前界面必须能够确认是演练，因为本地接口不能替代界面确认正式／演练类型。

当前三研究分支均已提供 `experiments/run_q3_practice.py`；该入口只支持 Q3，真实运行前要求演练确认、案例编号与队号。`--dry-run` 不发送请求。几何分支在冻结算法提交之后补入通用入口，入口文件和算法提交需要分别记录。

本轮协议的行动截止为北京时间 **2026-09-11 15:59:30**，并预留 30 秒退出。到期后不会继续动作。后续演练放在新一轮明确的执行窗口中，保留本轮协议与记录；这里没有自动解除截止或消耗正式测试次数的命令。

第一版建议先运行状态搜索，再用独立演练验证 RL。实际服务器不会保证相同随机场景，因此不要用各一次不同案例的时间差直接证明某方法更优。需要核对的是完成数量、失败清除、实际收费、反馈与研究模拟的一致性。

## 6. 论文材料在哪里

比较表、配对区间、时间组成图、真实路线图、下界审计、训练曲线和负结果均已完成。正文模板没有修改。各分支的 `research/state_search_v1/state_search_algorithm_basis.md`、`research/rl_algorithm_basis.md`、`research/geometric_algorithm_basis.md` 与 `research/theory_v1/` 保存详细依据，完整包含在 bundle 中。论文应明确区分本地研究模拟、官方演练、条件下界和数值离散模型，保留未达到门槛及坏局说明。
