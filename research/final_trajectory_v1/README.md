# 同一真实案例的四方法轨迹素材

[`plot_trajectories.py`](plot_trajectories.py) 只读取明确给定的四个完整评估目录，不调用策略、模拟器或场景生成器。沿用现有科研图的圆域、颜色与实际动作轨迹表达，不重复绘制算法构造图。

默认以 **baseline 的全部案例实际虚拟耗时中位数** 为中心，选择最接近的案例，等距按 seed 较小者。时间以原始十进制值比较，避免浮点末位误差打破偶数样本中央两例的平局。不按 state/RL/geo 的表现选例，也不悄悄排除失败案例。显式 `--seed` 必须同时给 `--selection-reason`，原因会写进图标题和 JSON，供之后展示预先指定的最坏回归案例。

四目录必须具有相同分区、完整种子清单、冻结协议 hash，且每个案例的 `case_sha256` 都一致；逐档核对 rows、spec 和策略终止后的真值 SHA。只在选定案例画图。每幅图使用相同坐标尺度，展示真实行进及方向箭头、测量点、红色源频道号、实际成功清除点、起终点和清除顺序；注明实际总时、移动时间、全部测量次数。显示时合并同坐标测量点，但不减少计费和测量次数；成功清除点与源真值可能非常接近。源坐标仅用于事后展示，与规划输入无关。

Windows 使用 Microsoft YaHei；SVG 把文字转为字形路径，阅读时不依赖目标机器安装中文字体。旁侧 JSON 保存源码/字体 SHA、Matplotlib 版本、全部输入归档 SHA、选例规则、实际清除点与费用，可追溯复算。图只表达该例，不能代替完整样本统计或最终物理深审。

本次唯一预检使用已打开开发 6000–6047：原 rollout、state inferred、RL PPO384、geometric relocation。中位数 3359.1435185 秒，选 seed **6019**；与 seed 6029 等距，按小 seed 取 6019。PNG 已视觉检查，SVG 同源导出并做 XML 检查；3 项防错测试通过。这些方法只是绘图兼容性预检，尚未在此确定最终候选。

- [开发例 PNG](development_median.png)
- [开发例 SVG](development_median.svg)
- [来源与数据](development_median.json)

从 `q3-geometric` 运行本次复算（换一个新输出前缀，工具保护已有文件）：

```powershell
python research/final_trajectory_v1/plot_trajectories.py --baseline ../q3-v1-artifacts/baseline-validation/validation-rollout --state ../q3-v1-artifacts/milestone-0550/state-ac57ffe/results/research_v1/validation-state-inferred --rl ../q3-v1-artifacts/milestone-0510/rl-f912dc8/results/research_v1/validation-cold-finetune-ppo-002-ppo_000384 --geo ../q3-v1-artifacts/geo-relocation-0720/geo-3c4f86f-git/results/research_v1/validation-geometric-relocation --output-prefix research/final_trajectory_v1/NEW_DEVELOPMENT_FIGURE
```

最终准确目录由根代理提供后，仍使用上述四目录参数，另加 `--partition final_random --allow-heldout`，或 `final_stress`；默认入口拒绝封存分区。若画指定负例，再加 `--seed SEED --selection-reason "明确说明预先采用的最坏回归选择规则"`。本轮没有读取 extended/final，也没有生成最终图。
