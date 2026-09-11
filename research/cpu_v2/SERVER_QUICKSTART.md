# 发给服务器操作者

请在能联网的Linux x86_64服务器运行，需Python 3.10～3.13、venv、util-linux（flock/setsid）以及配置好的`jiangsu10` rclone。整个任务只用CPU，禁止GPU。独立虚拟环境会自动安装CPU版PyTorch和NumPy。

逐行执行：

```bash
mkdir -p q3-cpu-v2-20260911-r1
cd q3-cpu-v2-20260911-r1
rclone copy 'jiangsu10:bucket-c20250204-pool01/lianghao/bwc/shumo/q3-cpu-v2-20260911-r1/input' . --s3-no-check-bucket
sha256sum -c q3-cpu-handoff.tar.gz.sha256
tar -xzf q3-cpu-handoff.tar.gz
cd q3-cpu-handoff
nohup bash RUN_WITH_SYNC.sh 'jiangsu10:bucket-c20250204-pool01/lianghao/bwc/shumo/q3-cpu-v2-20260911-r1' > operator-console.log 2>&1 &
```

启动后可检查：

```bash
tail -n 30 operator-console.log
tail -n 5 cpu_runs/.sync-console.log
```

默认先测速，再从同一父模型分别进行三个随机种子的训练，每个最多60分钟；还有安装、测速和评估时间，最迟北京时间2026-09-12 10:00停止计算，压缩和上传另需时间。CPU采样进程最高96，自动按当前CPU/内存限额下调并选取已测配置。共享服务器请在启动命令末尾明确加`--max-workers N`限制进程数。

进度约每60秒上传、保存的模型约每10分钟上传，结束后完整结果包自动到同一任务目录的`results/`。`live/LATEST.json`指向最新一轮已发布的文件清单。同步失败不会打断训练；本地文件会保留，下轮重试。停止后若上传失败，恢复网络执行：

```bash
python3 scripts/sync_cpu_results.py --remote 'jiangsu10:bucket-c20250204-pool01/lianghao/bwc/shumo/q3-cpu-v2-20260911-r1' --once
```

不要在同一解压目录重复启动，也不要修改源文件或替换模型。中断恢复、手动回传及资源选项见包内`README_OPERATOR.md`。本地已做功能测试；服务器96核的实际训练尚需本次启动。
