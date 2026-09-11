# 第四问全定向来源的独立补充验证协议

目的：补充当前本地生成器只生成混合类型场景的覆盖缺口。公开 Q4 允许
全定向源；本轮不把既有混合类型性能外推为已验证的全定向性能。
这是本地假设分布上的补充鲁棒性验证，不是官方成绩，不拟合采集数据库。

只新增独立 `AllDirectionalScenario(Scenario)`，复用 `Source` 对位置、
频道、接收半径和朝向的合法性检查；以子类显式替代旧 `Scenario` 的混合
类型限制，不修改或 monkeypatch 冻结生成器、物理引擎、策略及既有审计。
所有源朝向均非 None；源数仍为 10–16、频道互异、位置在 1800 m 圆盘内。

## 事先确定的两个集合

* 随机 16 例：616001–616016。源数均匀抽取 10–16、独占频道随机抽取，
  位置按圆盘面积均匀，接收半径均匀 1000–1500 m，全部朝向均匀 0–360°。
* 压力 14 例：616031–616044。按 `minimum_radius`、`boundary_outward`、
  `cluster`、`positive_error`、`negative_error`、`alternating_error`、
  `narrow_strip` 的顺序，每族相邻两例分别 N=10、N=16。压力全部 R=1000 m；
  边界族半径 1799.9 m 且朝向外，聚类族在距原点约 1100 m 的小区域，
  窄条带族沿轴向 900+30i m、横向 ±0.2 m。正负极端误差族采用对应固定
  有界误差，聚类/窄条带/交替族采用原引擎的坐标散列正负极端误差。

每个场景在各臂使用同一个种子和位置误差场。相同点重复测量不产生新的
独立误差。全部算法收到 `ObservationOnlyClient`；源真值仅在策略结束、
执行退出清理和 `finish_for_evaluation` 后用于共同下界、类型和物理审计。

## 候选及冻结顺序

固定两臂为原 `compact_baseline` 和已验证 `compact_combo/onroute`，完整
配置在新脚本中给定。只有 615xxx 的事先登记决定产生一个合格新版本时，
才允许在**打开这两个集合之前**增加这一个固定候选；否则仅两臂。
新增候选的资格文件需明确 `passed=true`、`selected=<候选label>`、
`spec=<完全相同的entrypoint/kwargs>`，与冻结清单一并保存。

`freeze` 命令只保存源码 SHA、协议、两个完整种子清单及规格，不生成任何
场景。其后显式 `run` 必须先核冻结内容、当前源码与协议完全一致，才能
构造预留案例。源码身份包含当前全部 src、旧实验依赖、全部现有审计器
以及本脚本；每个结果目录另存完整源码 ZIP、manifest 与 freeze。
本提交阶段只做非 616 种子的构造/防错测试，不运行上述验证集合。

每次源代码变更后，必须重新冻结且只能在集合仍未打开时进行。该验证
用于报告固定版本的补充结果；不能依据这批数据重新调参后继续将其称为
独立验证，也不单凭小样本的一次排名改变已验证主线。

## 费用、失败与独立审计

使用真实完成虚拟时间、完整分项费用及同场景共同历史下界，逐例保存
T、LB、T/LB；失败/未完成/证书或退出错误保留并罚 360000 s，合法光学
失败只计真实成本，不自动判整局失败。统计沿用共同 paired report，明确
区分均时/均界与逐例比值分布。报告均值、配对区间、尾部和全部失败。

记录格式兼容 `experiments.run_q4_round2 --audit`。本工具的 `audit`
先调用该通用物理、几何、范围及调度前缀审计，再从保存的终止后真值
核对所有来源均为定向、压力 N 分层、场景/类型/配对及资格身份。
输出既有 `independent_audit.json` 和新增 `all_directional_audit.json`；
已有审计文件不覆盖。额外类型审计不是重新运行策略，也不替代通用审计。

工具命令仅作说明，尚未执行任何 616 生成：

```powershell
python -B -m experiments.run_q4_all_directional freeze --output <plan.json>
python -B -m experiments.run_q4_all_directional run --freeze <plan.json> --stage random --output <new-random-dir>
python -B -m experiments.run_q4_all_directional run --freeze <plan.json> --stage stress --output <new-stress-dir>
python -B -m experiments.run_q4_all_directional audit --input <new-random-dir>
python -B -m experiments.run_q4_all_directional audit --input <new-stress-dir>
```
