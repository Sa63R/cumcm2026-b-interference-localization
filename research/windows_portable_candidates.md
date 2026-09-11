# 0810候选权重的Windows CPU交付检查

结论：三个实际候选文件均可由本地独立Torch环境载入。每个模型在指定训练场景`100121..100124`上运行4局，共12局全部清除、零失败清除、具备完成证书并正常退出。三份实际文件还分别通过了现有Q3演练入口的`--dry-run`；未使用官方模拟器，没有发送HTTP请求。

本检查不选择最终模型，不代表正式验证或跨平台性能提升。没有复制、改写或重新导出任何权重。

| 候选文件 | 特征/网络/分布 | 动作模式 | 首次模型载入秒 | 平均CPU单局秒 |
|---|---|---|---:|---:|
| `rl-best002.pt` | v3 / MLP96 / flat | base | 0.07164 | 0.26019 |
| `rl-axis-initial.pt` | v3 / MLP96 / flat | axis_quantiles | 0.00423 | 0.32007 |
| `rl-base-u393.pt` | v3 / MLP96 / flat | base | 0.00610 | 0.27260 |

环境为Windows、PyTorch2.9.1+cpu、单线程。模型载入时间不含Torch导入，三次载入存在顺序和库预热差异；单局时间包括本地策略执行，不含世界构造。样本仅4局，不据此排序模型或与Linux耗时进行算法收益比较。

实际文件位于`../q3-v1-artifacts/rl-portable-candidates-0810/rl-de7d65b-git/results/rl/v1-candidates/`。检查前后都与0810归档清单逐字节SHA256一致：

- best002：`e602c96224bd6bcfae6e0ad8a217aedf4075809d4901af2a43d5f08df7739c37`
- axis-initial：`a536d70a8f021a1f222325e1d8fe72f4cd27ed3a86cd826a3a42ef20b5a8fd73`
- base-u393：`484010a7be9d942a2f4bff2ac9ed06e5b703dacaa19e8024b5cf998b7adb0eb2`

复现入口：

```powershell
$env:PYTHONPATH='src;.'
.\.venv-win\Scripts\python.exe research/audit_portable_candidates_windows.py
```

[审计JSON](windows_portable_candidates.json)保留实际文件路径、权重与模型张量摘要、原始spec摘要、模式元数据、12局记录及dry-run身份摘要。[脚本](audit_portable_candidates_windows.py)仅把原spec的相对checkpoint路径替换为当前实际文件的绝对路径，临时spec随后清理；dry-run调用真正的`run_q3_practice.main`解析入口，并将HTTP客户端构造、socket连接和urllib发送入口设为立即失败，实际触发次数为0。

远端0810清单另记录了三个模型各48局与其原开发评估完整动作历史一致；那是既有Linux验证，本次没有重跑它，也没有读取扩展或最终测试。最终名单仍由独立选择流程确定，正式演练仍需另行确认当前GUI处于演练模式。
