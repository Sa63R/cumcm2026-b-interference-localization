# 第四问 V4 与等价计算加速

本目录提供第四问 V4 的完整演练入口。默认 `v4_fast` 保留 V4 的动作决策，仅加速路线、包围圆、裁剪和连续覆盖证明；使用 `--method v4` 可运行未启用这两层计算优化的对照版。

策略运行仅依赖 Python 标准库，发布整理版已在 Python 3.13 验证。原始 V4 位于 `../q4_comparison/vendor_v4/`，这些文件保持下载版原样。在线状态机、官方客户端桥接和两个计算缓存分开保存。

## 演练运行

先启动官方模拟器、登录，在界面选择**问题4演练测试**，等待准备完成并显示等待机器狗进入。然后从仓库根目录运行：

```sh
python experiments/q4_official_practice/run_speedup.py --method v4_fast --robot-id YOUR_TEAM_ID --practice-confirmed
```

将 YOUR_TEAM_ID 换为界面上的队号。不要把登录密码写进命令。Windows 可将 python 替换为已安装的 python.exe 完整路径。

每次命令只处理一场已经打开的演练；新场次需重新确认界面。HTTP接口不能查询或切换演练/正式模式，`--practice-confirmed` 是操作员确认，并非服务器校验。此入口没有启动正式测试的功能。

输出默认保存到 `results/q4/practice/<唯一运行目录>/`，包含原始请求、结果、版本、几何常量、代码哈希、CPU与墙钟用时。原始记录可能包含队号，已通过忽略规则保留在本机。

## 正确性与失败处理

- 接收测向物理误差按±1°建模，反馈舍入0.01°，定位采用±1.005°保守界。
- 只有接口返回清除成功才记录完成；无信号不会直接丢弃已发现目标。
- 达到16源上界，或完成实际扫描点的连续覆盖证明，才结束任务。
- 请求结果未知时保留原请求，不发替代动作；正常和异常退出都恢复计算函数及释放缓存。
- 缓存上下文用于单线程单场运行，不应重叠；并行测试请使用独立进程。

## 验证

本次发布检查：客户端及适配37项通过（另含10个子测试），原V4测试7项通过。详见 [发布验证记录](validation/release_checks.json)。

发布目录的离线完整对照覆盖真实全向、真实全定向和边界朝外场景，比较 V4 与 V4 Fast 的全部动作、反馈、虚拟时间及非计时报告；还验证缓存异常恢复和缺少演练确认时禁止连接。

```sh
python -m unittest discover -s tests -p test_q4_v4_release.py -v
python experiments/q4_official_practice/test_adapter.py
python experiments/q4_comparison/vendor_v4/test_v4.py
```

安装仓库已有测试依赖后，也可运行：

```sh
python -m pip install -r requirements-test.txt
python -m pytest -q tests/test_simulator_client.py tests/test_q4_v4_release.py
```

之前的两个专项配对测量摘要在 [主计算验证](validation/compute100.json) 和 [覆盖证明验证](validation/coverage_compute100.json)：
主计算100场、34,237条动作完全一致，CPU减少32.36%；覆盖证明100种输入除计时外的判定与记录一致，CPU减少51.76%。

这两项来自非独占机器上的不同测量对象，百分比不能相加，也不能当成官方HTTP整场提速。动作相同意味着机器人虚拟任务时间相同。新任务策略在后续复验中未获得稳定收益，因此此次发布只提供 V4 及其计算加速；此处不声明整理后的发布入口有新增官方演练成绩。

源码来源与原样保留的文件哈希见 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)。旧的第三问和第四问入口保持原样，V4需通过本页的新命令显式使用。
