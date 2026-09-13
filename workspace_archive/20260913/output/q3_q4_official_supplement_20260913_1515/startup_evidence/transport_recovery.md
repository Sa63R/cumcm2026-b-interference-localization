# 启动期间的后台传输恢复

2026-09-13 15:16 北京时间，utmctl 报错：

```
failed to get scripting definition from /Applications/UTM.app
NSInvalidArgumentException: -[SBApplication virtualMachines]: unrecognized selector
```

只读检查确认 UTM 和 QEMU 进程仍在，直接 AppleScript 能列出虚拟机。根据本机 UTM.sdef 的 open file、read/pull/write、execute 接口实现独立传输助手后，成功启动补充批处理。没有重启虚拟机、没有改变算法，没有调用 Mac 鼠标。

截图保留补跑前官方演练入口与客户端显示的 17:30 截止。批处理实际采用用户指定的更早截止：15:29:30 起不再开始新轮次，15:30 前收尾。

归档等待器在 15:26:21 读取正在更新的 status.json 时遇到文件共享锁，停止了等待器；演练批处理继续正常运行。15:29:34 确认 52 轮全部完成后，单独启动归档并按最终台账导出全部数据。此事件不涉及任何算法重跑。
