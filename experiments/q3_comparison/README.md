# 第三问本地配对实验

已下载的第三问代码来自同级 `gpt_download/q3`，不是 `q4`。现有版本直接调用原包；另外实现了多目标场景预测、未来搜索/清除位置联合设计及两者组合。

- `本地速度对比.md`：从本次结果生成的中文报告。
- `新方法设计说明.md`、`new_methods.py`：新实验方法、参数、近似与退路。
- `run_comparison.py`：九种方法共用的自建模拟器、附件规则修正和批测入口。
- `test_local.py`：附件计费、测向量化、源位置包含性、关闭新功能与 v3 的等价性、假想观测不污染真实证据。
- `results/annex_all400.csv`：400 个共同案例 × 九种方法的逐例数据。
- `results/runtime50.csv`：50 个共同案例 × 两轮 × 九种方法的串行计算计时。
- `results/stress350.csv`：350 个压力案例 × 九种方法。
- `results/original400.csv`：六种旧方法按下载包原规则运行的 400 案例结果。
- `vendor/`、`source_manifest.json`：保留的原包内容与源压缩包 SHA-256；`results/vendor_integrity.json` 记录 94 个包内文件字节一致性检查。

使用 Python 3.12；本目录 `.venv` 已配置好，依赖版本见 `requirements.txt`。完整重跑命令见中文报告。

所有动作是本地自建仿真，不发起 HTTP 请求、不连接官方模拟器，也未生成或替代任何正式测试日志。题目虚拟动作耗时与电脑计算耗时分别统计。
