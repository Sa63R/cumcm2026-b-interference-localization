# 第三问原点扫描版与有限提前光学版

本目录打包两个已经本地验证并接入官方演练的既有策略：

- `v3_origin20`：起点扫描20个频道，随后联合搜索、定位和清除。
- `optical`：普通v3上增加有限提前光学尝试；半径上限80米、面积覆盖阈值40%、每源最多两次。默认不做原点全扫描。

完整原理、结果区别与第四问方案见 [统一方法说明](../../docs/Q3_Q4_METHODS.md)。本次整理保留原策略定义；没有把两个第三问开关合并成新方案。

## 运行

Python 3.12，从仓库根目录执行：

```sh
python -m pip install -r experiments/q3_selected/requirements.txt
python experiments/q3_selected/run_practice.py --method v3_origin20 --robot-id YOUR_TEAM_ID --practice-confirmed
# 下一场演练可改用：
python experiments/q3_selected/run_practice.py --method optical --robot-id YOUR_TEAM_ID --practice-confirmed
```

运行前在官方模拟器界面选择问题3演练，等待准备完成。HTTP无法识别演练/正式模式，该标志仅表示人工确认。每条命令只运行一个已经打开的会话，日志保存在本机 `results/q3/selected_practice/`。

仅需NumPy，Numba可选，pandas不需要。原始 `vendor/optical_experiment_original.py` 包含旧版独立评测脚本及历史注释，仅作来源保留；应通过本页的新入口运行。`q3_optical.py`保留相同算法定义；`q3_runtime.py`负责正式附件的响应映射、角度舍入余量及成功计数核验。vendor内的ToySimulator是原始自建测试环境，运行器不把它当作官方环境。

## 检查与证据

```sh
python -m unittest discover -s tests -p test_q3_selected_release.py -v
python experiments/q3_selected/vendor/test_optimized.py
python experiments/q3_selected/vendor/test_v3.py
```

- [原源码与提取说明](SOURCE_MANIFEST.json)
- [本次发布检查](validation/release_checks.json)
- [历史本地400组配对结果](validation/local_summary.json) / [逐例CSV](validation/local_paired400.csv)
- [历史官方182/900轮统计快照](validation/official_snapshot_182.json)

本地数据是自建场景；官方数据是不同地图的非配对样本。快照不用于调参，也不随后台演练刷新。有限样本全清不构成任意场景全清或全局最快的证明。
