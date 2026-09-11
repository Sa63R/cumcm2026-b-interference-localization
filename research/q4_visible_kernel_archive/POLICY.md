# R11 单候选接入骨架

配置固定为 `after_silence_once`，入口 `strategies.q4_visible_kernel:run_q4_visible_kernel`。继承已合格 R9 的 `probe` 配置，R9 九个新增生产、审计与测试文件从 `e3c6cda00547085a9b065ae5e44bb2c2ca978b4f` 按原字节导入，清单见 `r9-import.json`；原同名不同字节文件不会覆盖。

仅当本次 resolver 的首探之后，上一个接受动作是同频道 `active_localization=no_signal`，仍有主动探测预算，canonical 与 R9 aux 都没有现成清除证书时，考虑内核救援。候选只从 canonical 与真实正点生成，至多24个；评分器沿旧至多5名义支持。原首探、原 R8 中心尝试、全局覆盖/调度/删测/光学不变。现成 canonical 或 aux 清除证书优先。

每次 resolver 最多实际执行一次救援：helper 空核或没有 fresh 候选时回到原 R9，不消费这一次实际机会；预算或请求拒绝没有产生接受动作时也不能伪装已执行。若 rescue 点恰好满足原 R8 谓词，仍允许真实 clear 替代 measure。判断 R8 成功只使用实际成功回包。

`_next_probe` 在真正选中 rescue 时直接返回自己的点，不能先调用 R9 五点 hook 再偷换结果。新日志明确拥有该探测索引；父 R9 的 `_perform` 仍负责 canonical 和 aux 的真实方位更新，新层仅补执行前缀与次数账。所有父 `finally` 保留，服务片/虚拟时间/真实时间/动作预算沿原路径执行。

`strategy_parameters.visible_kernel_rescue_log` 保存 resolver ID、探测 index、触发的真实无信号动作索引、决策前后接受动作数量、完整内核证书和候选分值、选点、R8 插入标志与实际测量/消费标志。`fallback` 事件是一次几何尝试，不拥有真实探测索引；选中事件与 R9 五点事件共同构成连续索引序列。原 R9 auditor 单独使用会正确拒绝这类新归属，因此冻结前必须完成 R11 专用独立前缀审计，不能通过删日志或伪造 R9 五点候选绕过它。

19项新控制测试与冻结 R9/R8 回归合计 **71 passed**：首探/单探预算逐动作等价、真实失败后救援、实际次数限制、空核回退、预算/拒绝不假执行、同频道和本 resolver 前缀边界、父 aux 真方位更新、aux 清除优先以及全局方法继承。

```text
python -m pytest tests/test_q4_visible_kernel.py tests/test_q4_joint_visibility_strategy.py tests/test_q4_clear_before_probe.py -q
```

`development-specs-draft.json` 仅为接口草稿，不代表已冻结或已运行场景。主比较对象应为新合格 R9，其他臂仅作背景。不得把旧诊断的名义评分下降当作性能收益；需先冻结专用审计与协议，再打开620开发集。本提交没有运行620或任何新完整场景、官方模拟器、数据库或SSH。
