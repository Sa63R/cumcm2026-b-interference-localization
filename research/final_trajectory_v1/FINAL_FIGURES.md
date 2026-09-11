# 冻结最终案例的真实轨迹图

三幅图均使用提交 `4544ae3` 的原绘图脚本，源码 SHA256 `4aa0850b9dac74dda4157a8743a26e1075f8114087aa052ee2745a20da5a4658`。读取根代理已下载并核验的最终随机集；每次核对四方法完整 256 局的身份、归档和逐例 case SHA，只绘制所选场景，没有调用策略或生成场景。最终策略、权重、选择规则均未修改。

| 图与选例依据 | seed | 原 rollout / 秒 | 状态 / 秒 | RL / 秒 | 几何 / 秒 |
|---|---:|---:|---:|---:|---:|
| [基线中位案例 PNG](final_random_median.png) / [SVG](final_random_median.svg) | 800035 | 3331.226 | 3118.225 | 2943.015 | 3202.839 |
| [状态最差回归 PNG](final_random_state_worst.png) / [SVG](final_random_state_worst.svg) | 800015 | 2919.668 | 3044.750 | 2843.126 | 2935.522 |
| [RL 最差回归 PNG](final_random_rl_worst.png) / [SVG](final_random_rl_worst.svg) | 800081 | 3238.722 | 3059.929 | 3604.168 | 3170.368 |

中位图只根据 baseline 的实际虚拟耗时选例：256 局中位数为 **3331.1364115 秒**，选择离中位数最近者，十进制精确等距时取较小 seed。因此本例不是三个研究方法的“共同中位表现”，也不是挑出的有利展示例。另两图按最终共同 comparison 的相对 baseline 最大耗时退化事后选取，图标题明确标注此规则；它们用于展示边界，不能用其频率估计总体概率。

每图四子图采用相同尺度，源真值只在策略终止后用于展示。真值、测量、实际清除点和真实移动轨迹均保留；源编号是频道号，清除顺序另列。图下 JSON 包含全部档案 SHA、字体与绘图版本、每个点坐标和各项真实费用：[中位来源](final_random_median.json)、[状态负例来源](final_random_state_worst.json)、[RL 负例来源](final_random_rl_worst.json)。三幅 PNG 已逐幅目视，源编号、坐标、图例和选例说明可读；SVG 同源导出并验证 XML。

## 两个退化案例的费用分解

**状态 seed 800015：** 相对 baseline 慢 **125.081450 秒**，由移动 **+152.081450**、测量 **−25**、换频 **−2** 组成，成功清除费用相同。两者前八个清除频道同为 `8→17→19→5→1→11→13→10`；之后 baseline 为 `2→12→14→6`，状态为 `6→12→2→14`。图中南部路线及探测点不同；这证明此例测量减少未抵消移动增长，但不是对某个单独模块的因果消融证明。

**RL seed 800081：** 相对 baseline 慢 **365.446723 秒**，由移动 **+340.446723**、测量 **+20**、换频 **+5** 组成。baseline 顺序为 `4→5→17→19→12→13→18→6→8→11`，RL 为 `5→4→11→8→18→13→6→12→17→19`；RL 在北部出现跨区域行进，同时测量次数由 136 增至 140。真实账本支持“主要损失在移动”的描述；不能只凭轨迹认定网络内部作出这些决策的因果原因，更不能据最终负例回调模型。

所有 12 条所画轨迹均全清、零失败清除。整体全样本统计与下界差距见 [最终物理审计素材](../final_physical_audit_v1/FINAL_RESULTS.md)，不由这三张图代替。

## 复算命令

从工作树根目录执行；将输出前缀换成不存在的新文件名。Windows 使用已安装 Microsoft YaHei 的 Matplotlib 环境。

```powershell
$trajectoryBase = '../q3-v1-artifacts/final-evaluations-complete-0930/v1-selection/evaluations/final_random'
$trajectoryArgs = @('--baseline', "$trajectoryBase/baseline", '--state', "$trajectoryBase/state-future-cover", '--rl', "$trajectoryBase/rl-gae095-u512", '--geo', "$trajectoryBase/geo-future-cover", '--partition', 'final_random', '--allow-heldout')
python research/final_trajectory_v1/plot_trajectories.py @trajectoryArgs --output-prefix NEW_FINAL_MEDIAN
python research/final_trajectory_v1/plot_trajectories.py @trajectoryArgs --seed 800015 --selection-reason '事后选取：状态搜索随机集最大退化（+125.081 s）' --output-prefix NEW_STATE_WORST
python research/final_trajectory_v1/plot_trajectories.py @trajectoryArgs --seed 800081 --selection-reason '事后选取：强化学习随机集最大退化（+365.447 s）' --output-prefix NEW_RL_WORST
```
