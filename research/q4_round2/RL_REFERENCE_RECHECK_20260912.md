# Q4 RL 参照只读复核（2026-09-12 04:55 北京时间）

只读核查本机 Git worktree 和各 Q4 RL 树已同步的完整评估摘要；未连接远端、接管训练、使用 GPU 或更改已冻结参照。结论：尚未发现完整评估优于当前宏 PPO512 的新 RL 模型。未同步的远端结果不在本次结论范围。

最新完整面板是 `q4-rl-micro-attention/results/q4_rl/server-bundle-v3-evaluation-001/` 的 summary/manifest/progress，12 方法 × 512 场景，6144/6144 完成。512 场景来自 32 个 seed 簇、8 个家族、2 种源类型模式，不能称为 512 个独立重复。所有方法全清，档案独审见该树 `research/q4_rl/BUNDLE_V3_EVALUATION_REVIEW.md` 和 `results/q4_rl/bundle-v3-independent-review-001.json`。最新文档提交 `afa77738`，03:22。

| 方法 | 平均 T/s | 平均旧 LB/s | 均 T/均 LB |
|---|---:|---:|---:|
| 当前 macro_v2_ppo512 | 7998.673062 | 1790.822234 | 4.466481 |
| 新 attention_v3_ppo512 | 8717.620257 | 1790.822234 | 4.867943 |
| 新 mlp_v3_ppo512 | 9218.714256 | 1790.822234 | 5.147755 |

本次从 summary 逐行复核这三臂 512 个 case 身份、case SHA、共同 LB 及 successful/audit_passed 均一致。已有独审报告中，attention 对 macro 的节省率为 −8.99%，32 seed 簇配对区间 [−10.38%, −7.64%]，因此无替换依据。这个开发面板不是新模型的独立晋级确认。

身份：

- 当前模型 SHA256（新面板 manifest 的 macro alias 再次一致）：`3e830b070d5247e3573b3e59b7dfc32d54da101c5aab05cd4effb0dec2b354e8`。
- R9 冻结源包：`86580efae901756eccaeeb48edb6045ca2a9bb8ac9b436ca60cf17fa9179229d`；身份文件 `q4-r9-joint-visibility/research/q4_joint_visibility/rl_reference/reference.json`。
- 新 12 臂评估 summary SHA256：`2848138b614b571be706f5f3ebe18e28031610886c4c45c0ebce40c59adc2026`；其完整评估源包 SHA256：`c5f12f70d49d6a9a43b9cc73905081ab40b992ad3ad4113b83f1401966269797`。后者含新微动作模块，并非当前固定模型的部署源包。

更新的 memory-v4 本地只有 BC-fit 与 C 面板恢复材料：`q4-rl-negative-memory/research/q4_rl/recovery_sidecar/README.md`、`results/q4_rl/memory-v4-bc-fit-001/SUMMARY.json`；04:33 提交记录评估恢复启动。BC 分类精度不是策略运行时间。`q4-rl-gae` 04:46 最新记录涉及训练/存储实现，没有新的完整策略比较摘要。未来若完整新面板同步，应另行审核模型与同场景对照，不能悄改正在运行的 R16 参照。
