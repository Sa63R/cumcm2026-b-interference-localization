# Q4 V6 Lite

简化版删除学习路线重排、后验共享评分，保留局部前瞻、沿途学习检测和前四站保护。具体收益与退步见同目录的实测报告；本地结果不等于官方演练或正式测试。

Python 3.10+，只需标准库。在本目录运行：

```bash
python3 run_lite_local.py --seed my-new-demo --out LOCAL_ONLY_lite_demo.json
```

已有输出会被保护，复跑请更换输出名。本命令在附带的复原模拟器生成新场景，不访问官方服务。

接入自己的设备时，把 `source_v6_lite/` 加入 Python 模块路径：

```python
from q4_v6_lite import solve_v6_lite, default_config
report = solve_v6_lite(device, default_config())
```

设备接口沿用原 V6 的 position、channel、move、detect、clear。运行目录为 18 个 Python 模块和 1 个沿途模型；局部前瞻仍使用后验推断。官方协议适配仍需单独验证。

完整对照代码、冻结地图、逐场会话和校验记录在另附的“第四问_V6_Lite_简化版_代码与完整实测数据.zip”。
