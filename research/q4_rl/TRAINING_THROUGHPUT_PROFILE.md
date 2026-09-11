# 训练吞吐离线微基准

结论：在同一已下载的首批原始数据上，主要串行开销来自流式 `json.dump` 的大量 Python 分块写入，以及 `pack_observations` 的逐标量有限性检查。只降低 gzip 等级收益有限；整块 JSON 编码的收益明显。该实验没有启动新场景，没有修改或保存模型，不是策略效率或泛化提升。

测试输入是 16 局、2048 决策的 `server-training-pilot-001/batch-000000.json.gz`，SHA-256 `6320dbf1e5260db865e863deffb4ec27882a2365d6fa8c9ba8f998e262b8da81`，解压大小 `106554238` 字节。所有结果均在本地单个逻辑 CPU affinity、PyTorch/BLAS 线程 1、禁用 CUDA 下完成。每种写出管线只测一次，顺序固定，不能将本机耗时等同于远程吞吐；读入、对象校验与训练更新分开计时。

| 写出方式 | 总墙钟 s | CPU s | 压缩字节 |
|---|---:|---:|---:|
| 原 `json.dump` + gzip 1 | 13.756267 | 13.250000 | 4662833 |
| 原 `json.dump` + gzip 6 | 14.341423 | 13.953125 | 2989754 |
| 原 `json.dump` + gzip 9 | 16.930994 | 16.343750 | 2822994 |
| `json.dumps(...).encode()` + gzip 6 整块写入 | 2.595995 | 2.296875 | 2989740 |
| `json.dumps(...).encode()` + gzip 9 整块写入 | 3.878276 | 3.375000 | 2822981 |

整块编码本身为 `1.887469 s` 墙钟 / `1.593750 s` CPU。保持 gzip 9 时，总写出时间减少 `13.052717 s`，约为原来的 `22.91%`；这是单次离线管线微基准，不能承诺整批端到端训练提升同一倍数。两种整块写出逐字节等于原始解压内容；五份输出解压后 SHA-256 都为 `69a3f53de1a7113e050e74d12be1d0049b76c6067a42bf04df2e0dd570cfb095`，并逐一确认 `json.loads(...)` 后对象完全相等。压缩后很小的字节差异不涉及数据内容。代价是额外保留约 106.6 MB UTF-8 字节缓冲以及完整 JSON 字符串；Python 字符串内存取决于字符集，峰值不能直接当成只增加一个字节缓冲，需要在真实训练进程中留出余量。

只在内存中的 checkpoint 副本上执行了一次原 `imitation_update(epochs=1, minibatch_size=128)`，共 16 个更新，输入 checkpoint hash 在更新前后相同，没有写出模型。cProfile 总墙钟 `10.580269 s`、进程 CPU `10.187500 s`；函数累计时间如下，嵌套行不能相加：

| 函数 | 累计 s | 说明 |
|---|---:|---|
| `imitation_update` | 10.559215 | 含 packing、前向、反向、优化器 |
| `pack_observations` | 6.789401 | profile 总时间约 64.3% |
| `builtins.all` | 5.536849 | 属于上行，Python 特征遍历和有限性检查 |
| `math.isfinite` | 1.071107 | 上层累计时间的一部分，10428336 次调用 |
| `torch.tensor` | 0.797496 | 4128 次调用 |
| 网络 `forward` | 1.724001 | 包括线性层、池化和激活 |
| Tensor `backward` | 1.957432 | 含本地反向引擎 |
| `Adam.step` | 0.020832 | 小于主要数据整理开销 |

cProfile 对大量 Python 调用有测量扰动，这些份额用于定位，不是未经 profile 的精确加速上限。保留原输入和更新设置，可用修复后的实现复测管线吞吐，但不能把多一次训练更新作为新策略实验结果。

建议先做两项不改变合法输入策略语义的修复：整块 JSON 编码；保持 shape、dtype、mask 和原 tensor 完全相同，将有限性检查移动到 tensor 转换后批量执行，并明确拒绝 NaN/Inf 和 float32 转换溢出。后续如有证据再考虑缓存一批已验证的 tensor，不能直接省去检查。主线程已实现这两项修复，实际选择 gzip 6：本批压缩体积约比 gzip 9 大 5.9%，进一步节省 CPU，解压内容仍须完全相同。此审阅任务没有修改 `train.py` 或 `network.py`。

归档：

- `results/q4_rl/training-profile-001/`：流式三个压缩包、原始计时 `report.json`、当时完整基准脚本 `source.py`、修复前策略源码 `code_identity.json`。cProfile 函数计时保存为经过路径脱敏的 Top80 JSON，没有保存含本机个人路径的二进制 `.prof`。不能据此声称保留了全部低耗时调用图。
- `results/q4_rl/training-profile-bulk-001/`：两份整块压缩包、原始编码/写出计时、当时基准脚本和修复前源码 hash。校验耗时未混入写出速度。

主线程完成矢量有限性检查后，按授权仅再用同一批次、同一 checkpoint 和同一恢复 RNG 执行一次 `epoch1/B128` 离线 cProfile，不重测压缩、不运行场景、不保存模型。结果在 `results/q4_rl/training-profile-pack-fixed-001/`：总墙钟 `4.667781 s`、CPU `3.984375 s`；`pack_observations` 累计从 `6.789401 s` 降为 `1.149010 s`，在本次 profile 中占比约 `24.6%`，不再是最大的单项。网络 forward `1.465782 s`，向量 `torch.isfinite` 合计 `0.163102 s`。16 次更新的平均 imitation loss 前后均精确为 `5.717947006225586`，输入 checkpoint 前后 hash 不变。该次 profile 保留全部 231 项脱敏函数计时、脚本快照和实际修复后源码 hash；精确输入 tensor/非法值/压缩字节等价性另由主线程单测验证。以上仍是一次前/后离线 profile 对照，有 profiler 扰动及运行顺序影响，不是多次端到端吞吐置信区间。

复现当前实现的离线测量：

```powershell
python scripts/q4_profile_training_io.py `
  --batch results/q4_rl/server-training-pilot-001/batch-000000.json.gz `
  --checkpoint results/q4_rl/server-training-pilot-001/checkpoint-000001.pt `
  --mode all --profile-imitation `
  --output-dir results/q4_rl/training-profile-new
```

只测编码可使用 `--mode bulk` 并省略 checkpoint 和 `--profile-imitation`；只做一次更新 profile 可使用 `--mode profile --profile-imitation`。脚本只读取明确提供的本地数据；输出必须是新的独立目录。每次实际执行会保存脚本快照与 `train.py`/`network.py` hash。修复前历史基准需要对应 `code_identity.json` 的源码，不能拿当前不同实现冒充原基准。
