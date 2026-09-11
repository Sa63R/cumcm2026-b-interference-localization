# 问题3强化学习：纯CPU远程运行说明

本包仅运行本地研究仿真。全部采样、网络优化和评估必须使用CPU；不要调用任何官方模拟器或正式测试。包内不含对象存储密钥。不要将私钥、S3密钥、系统环境变量转储混入返回结果。

## 需要的机器

Linux x86_64，Python 3.10～3.13，可以联网安装依赖，建议为本次运行提供独占或明确限额的CPU和内存。脚本读取进程CPU亲和性和cgroup v2额度；在96个可用CPU槽时可以测试16/32/64/96个采样进程。内存不足会自动降低上限。这里CPU槽可能是逻辑核/vCPU，不宣称就是96个物理核心。

依赖固定为PyTorch 2.9.1 CPU版本和NumPy 2.2.6，安装在包内独立的`.venv-cpu`。安装需要能访问PyTorch CPU wheel源和Python包源；若系统缺少`python3-venv`或版本不符，请由机器管理员准备Python后重试。不要把现有CUDA环境直接用于本包。

## 开始运行

把收到的训练包与同名`.sha256`文件放到同一空目录，逐行执行：

```bash
sha256sum -c q3-cpu-handoff.tar.gz.sha256
tar -xzf q3-cpu-handoff.tar.gz
cd q3-cpu-handoff
bash START_CPU.sh
```

实际训练包可能带版本号；前两条把文件名替换为实际文件名即可。默认前台运行；若需要关闭SSH窗口，使用已有的`tmux`/`screen`会话运行，或把最后一条替换为：

```bash
nohup bash START_CPU.sh > operator-console.log 2>&1 &
```

不要重复启动同一目录。正常顺序：

1. 校验代码与父模型SHA256、记录CPU和内存配置；确认CPU版Torch。
2. 短时间比较几个采样进程数。基准模型权重固定，学习率为零，第一批用于预热，不拿测速结果声称算法变好。
3. 在固定的48个新开发场景上评估原模型。
4. 从同一已选模型分别初始化三个训练种子，保持PPO/v3/base/MLP96/GAE0.95及原总时间奖励。每个种子最多60分钟、每10分钟保留阶段模型，`latest.pt`每次更新保存。
5. 每个训练端点在相同48个开发场景上评估。自动打包完整日志、全部阶段模型、端点评估轨迹和摘要。

默认总耗时为三段各最多60分钟，另加测速、安装和评估时间；不是承诺必定达到某个采样局数。每种子最多尝试199936局，达到墙钟预算可能远少于此。默认停止时间为北京时间2026-09-12 10:00，以便回传分析；截至此时尚未完成的训练和评估会明确保留为不完整。结果压缩和上传仍需要额外时间。

如机器只允许占用32个CPU槽，可以运行：

```bash
bash START_CPU.sh --max-workers 32
```

如要先做功能检查：

```bash
bash START_CPU.sh --smoke --output cpu_runs/smoke
```

功能检查只跑极少轨迹，不能作为训练收益或96核性能证据。正式长训练继续使用默认`cpu_runs/run01`，不要复用smoke目录。

## 中断和结果返回

测速完成后如训练中断，可在同一包中用**相同参数**重新执行`bash START_CPU.sh`。它从`latest.pt`恢复优化器、随机状态和样本游标，累计训练时间不会因为重启而归零；已经正常结束的种子跳过。若中断发生在测速阶段，请保留原目录和日志，使用新的输出目录重新运行，例如`--output cpu_runs/run02`。

运行结束后会打印`RESULT_ARCHIVE`，文件类似：

```text
cpu_runs/run01-results-20260911T140000Z.tar.gz
cpu_runs/run01-results-20260911T140000Z.tar.gz.sha256
```

只需把这两个文件传回约定对象存储的`results/`子目录。如果失败，也请返回已生成的结果包；安装阶段失败尚无结果包时，返回`cpu_runs/startup.log`、`cpu_runs/preflight.json`和`operator-console.log`（若存在）。不要只返回截图或平均分，分析需要模型和原始日志。

已经配置好截图所示`jiangsu10`的rclone时，交换方式为：

```bash
# 下载：<交换目录>由发包者提供，位于本次指定的lianghao/bwc/shumo目录之下。
rclone copy 'jiangsu10:bucket-c20250204-pool01/lianghao/bwc/shumo/<交换目录>/input' . --s3-no-check-bucket
# 返回：替换为实际生成的文件名，使用copyto单文件复制，勿用sync。
rclone copyto 'cpu_runs/<结果文件>.tar.gz' 'jiangsu10:bucket-c20250204-pool01/lianghao/bwc/shumo/<交换目录>/results/<结果文件>.tar.gz' --s3-no-check-bucket
rclone copyto 'cpu_runs/<结果文件>.tar.gz.sha256' 'jiangsu10:bucket-c20250204-pool01/lianghao/bwc/shumo/<交换目录>/results/<结果文件>.tar.gz.sha256' --s3-no-check-bucket
```

若未安装rclone，仍可用已有的S3客户端交换文件；包内`scripts/object_exchange.py`也支持外置JSON凭据（需另装boto3）。不要把凭据写进训练代码或Git仓库。

## 训练期间自动回传（推荐）

服务器已有配置好的`rclone`时，在解压后的包根目录，将启动命令替换为以下命令，`<交换目录>`替换为发包者提供的任务目录名：

```bash
nohup bash RUN_WITH_SYNC.sh 'jiangsu10:bucket-c20250204-pool01/lianghao/bwc/shumo/<交换目录>' > operator-console.log 2>&1 &
```

其他训练参数可以接在远程目录之后，例如`--max-workers 64`。不要再同时启动`START_CPU.sh`。每约60秒回传日志/摘要，每约600秒回传已经保存的模型；实际间隔会加上上传耗时。正常结束后还会同步全部模型和完整结果包到`results/`。不需要操作者持续手动上传。

同步器先读取稳定快照，以SHA256命名内容对象；全部上传后发布`live/manifests/`清单，最后更新`live/LATEST.json`指针。清单含每份文件的采集时间，日志与模型可能来自不同更新时间，不将“最后一次同步”当作“训练完成”。各文件SHA256可在接收端复核。

网络失败不会中断训练，下一轮重试；可查看`cpu_runs/.sync-console.log`了解上传状态。断电、强制杀进程或持续断网仍可能使最后阶段只留在服务器。若结束时同步失败，恢复网络后在包根重试：

```bash
python3 scripts/sync_cpu_results.py --remote 'jiangsu10:bucket-c20250204-pool01/lianghao/bwc/shumo/<交换目录>' --once
```

工具只复制结果文件到本次专属目录，不执行远端删除。所有rclone写入均携带`--s3-no-check-bucket`；该参数避免查询或创建桶，保留对象上传校验。依据：[rclone S3参数说明](https://rclone.org/s3/#s3-no-check-bucket)、[copyto说明](https://rclone.org/commands/rclone_copyto/)。

## 研究口径

这是扩大采样与纯CPU训练的后续开发试验，不保证比原模型更快。三个训练种子都从同一个父模型初始化，属于相同起点下的独立后续训练，不是三个独立从零训练。批量由旧32局扩大为128局，因此本次变化同时包括采样量与PPO更新批量；不能把任何收益单独归因于“96核”。

训练采用互不重叠的新场景段，旧最终数据不用于训练。48局仅为开发端点评估；未来256局保留集不由本包自动打开。所有源数、失败和完整性结果由退出后的评估器统计。最终选型和与状态压缩版的配对比较由接收方用返回模型继续完成。

用户另行采集的`practice_training.sqlite3`整库只用于验证，即使库中旧字段带有`train`/`trainable`也不得用于训练或拟合分布。该库不随本包发给训练器。它保存已执行轨迹和保守位置估计，缺少完整场景真值；可做轨迹审计和有限离线验证，不能精确计算新策略改走其他路线的反馈和总耗时。闭环性能先用本包独立研究场景评估，回传后再用官方演练确认。

包内`PACKAGE_MANIFEST.json`记录源文件、父模型和提交身份；`research/cpu_v2_protocol.json`记录训练与评估分区。不要在运行过程中编辑源码、替换模型或更换依赖版本；需要改动时保留这次结果，重新生成一个独立包。
