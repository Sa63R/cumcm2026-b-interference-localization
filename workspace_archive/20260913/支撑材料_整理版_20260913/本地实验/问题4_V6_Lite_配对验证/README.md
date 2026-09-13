# Q4 V6 Lite：消融后简化版

本版删除学习路线重排和后验共享补测评分，恢复 V4 路线顺序及共享补测启发式。保留局部前瞻、沿途学习检测、前四站保护和全部几何／覆盖／真实清除反馈条件。局部前瞻所用的后验推断仍然存在。

可部署代码在 `source_v6_lite/`：18 个 Python 模块、1 个沿途检测模型。它不包含 `q4_route_critic.py`、路线模型权重、`q4_rb_sharing.py` 或 V5 的组合入口。`reference/` 另外保存原版 V6 与复原模拟器，用于复现对照实验。

## 使用

Python 3.10+，求解器仅需标准库。将 `source_v6_lite/` 加入模块路径后：

```python
from q4_v6_lite import solve_v6_lite, default_config

report = solve_v6_lite(device, default_config())
```

`device` 提供 `position`、`channel`、`move(point)`、`detect(channel)`、`clear(channel)`；观测对象沿用 `q4_baseline.Observation`。调用前应建立新的任务会话。API 与原版一致，函数名改为 `solve_v6_lite`。

在本包的复原模拟器上演示，不访问官方服务：

```bash
python3 run_lite_local.py --seed my-new-demo --out LOCAL_ONLY_lite_demo.json
```

此命令使用冻结对照中的 1.005 度保守观测界和完整任务计时。已有输出会被保护，重新运行时更换文件名。官方协议适配和端到端一致性尚未验证，不能把这些会话当作官方演练或正式测试记录。

## 对照实验

在上一轮消融之后、生成本轮新地图之前，固定唯一候选：同时关闭路线学习与后验共享评分。没有根据本轮结果再调整参数、删选地图或重训权重。

- 1,000 个新随机场景是主测试；主指标是每场完整虚拟任务时间除以该场源数，再对场景等权平均。
- 全定向、边界朝外最小半径各 70 场，单独汇总，不混入主均值。
- 每场配对运行原版 V4、原版 V6、可部署的 Lite，共 3,420 次。
- 另选冻结随机组前 50 场单进程复测计算时间，共 150 次；轮换运行顺序。它们不是新的独立地图。
- 主比较只有一个：Lite 减原版 V6。50,000 次同场景配对 bootstrap 给出 95% 区间；压力组、源数分组及尾部统计作为探索和描述。

结果见 `V6_Lite_简化版实测报告.md`。`plan.json`、`source_hashes.json` 固定配置、地图与代码；`main/sessions/` 保存完整会话，`paired_records.csv` 保存逐场结果。

macOS/Linux 复现批量实验时，在副本中先改名保存原 `main/`、`serial/`，然后运行：

```bash
python3 test_lite.py
python3 run_validation.py --phase main --workers 4
python3 run_validation.py --phase serial
python3 verify_validation.py replay
python3 verify_validation.py serial
python3 analyze_validation.py
python3 make_report.py
```

批量入口使用 POSIX 超时机制。分析需 NumPy；图形脚本另需 Matplotlib。复跑这些已公布地图属于复现，不是新的独立验证。
