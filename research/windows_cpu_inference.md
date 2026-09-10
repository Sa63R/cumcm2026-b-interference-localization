# Windows CPU 推理准备与检查点便携性

本机已经准备好独立的 `.venv-win`：Python 3.12.14、PyTorch 2.9.1+cpu、NumPy 2.2.6。该环境在 Git 忽略目录中，不修改 main 的环境。训练仍在远程 RTX 4060 上进行；演练时网络可在 Windows CPU 推理，避免每一步通过 SSH 往返。

安装依据是 [PyTorch 官方旧版本安装页](https://docs.pytorch.org/get-started/previous-versions/)的 Windows/Linux CPU wheel 通道。项目不需要 torchvision 或 torchaudio。新机器的安装命令（在本分支工作目录运行）为：

```powershell
python -m venv .venv-win
.\.venv-win\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv-win\Scripts\python.exe -m pip install numpy==2.2.6
```

## 为什么旧模型要导出

训练检查点原先把 `argparse` 中的具体 `PosixPath` 对象一起保存。Linux 可以重新加载，Windows 会在 `torch.load` 报 `cannot instantiate 'PosixPath'`。这不是网络参数或 CUDA 设备不兼容。`23817e2` 起保存前递归将路径元数据转成文本，不改网络、优化器或随机状态。

此前保存的检查点需在 Linux 显式导出到新文件：

```bash
PYTHONPATH=src python -m research_rl.portable_checkpoint --input /path/to/original.pt --output /path/to/new-portable.pt
```

导出保留原文件和原训练 source manifest，并生成原文件 SHA256、输出 SHA256、网络张量 SHA256 和转换路径清单；拒绝覆盖已有目标。只修改元数据路径表示，不能把它计为新训练。没有全局替换 `pathlib`，也没有绕开动作/特征语义检查。最终冻结应针对实际部署的便携文件重新记录字节哈希。

## 已完成的实测范围

2026-09-11 05:10 前，使用真实训练的 `cold-finetune-ppo-001/ppo_000384.pt` 做了跨系统验证。该模型是当时的临时候选，最终选择仍以独立评估和冻结为准。

- Linux 原始检查点 SHA256：`c670824b1e8e4aff8dfc9ab1323d21f3a404630826bd8e9052c36b9019301944`。
- 便携文件 SHA256：`4c0ae80b599a4a3effda71f640201a22ae5bd3be21b01ec9adaad849e6aa31f5`。
- 网络张量摘要在 Linux 导出前后与 Windows 加载后完全相同：`ba3207954875492bdf4648451d254ef82588667f6e336e2bcf2c3daf6269c3a1`。
- Linux 导出前后同输入的 logits/value 逐值相等。Windows 对同一组40候选输入的最大 logits 差 `3.8147e-6`、value 差 `3.5763e-7`，在 `atol=rtol=1e-6` 检查内，greedy 动作相同；不宣称跨平台浮点结果逐位一致。
- 无请求演练 dry-run 成功；没有创建官方 HTTP 客户端或发送官方指令。
- Windows 训练分区场景 `100121/100122` 均全部清除、零失败清除，本地策略程序耗时约0.22/0.30秒。两局只检查加载和执行能力，不作为性能提升证据，不与 Linux 场景逐局配对。
- 基础推理/训练/演练回归82项通过；便携导出及相关 PPO/paired 回归34项通过，两组有重叠，不能相加当作116个独立测试。

详细本地证据位于相邻 `q3-v1-artifacts/models-provisional/windows-cpu-verification.json`，验证脚本为 `q3-v1-artifacts/verify_windows_inference.py`；原始模型仍保存在 Linux 与 `milestone-0410.tar.gz` 中。文件路径和训练日志不是官方成绩。

用户回来并开启问题3演练后，使用 `experiments/run_q3_practice.py` 和届时选定的本地便携模型 spec。当前只完成准备与离线验证；正式测试未调用，官方演练也仍待用户就绪。第一版执行截止和演练身份检查仍按共同协议执行。
