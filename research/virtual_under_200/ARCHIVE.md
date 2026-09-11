# 未采用试验的恢复方式

试验提交：`81e298c84d4e8915d8581bd7da01752bc2c09873`。
原本轮新分支：`experiment/q3-enroute-scan`，未采用，不是推荐部署入口。
依赖基准：`36859f4d8d61eaae820005f4da48c02fcf36e309`。

`enroute-trial.bundle`包含该试验提交新增的源码、测试、完整配置、筛查记录、协议和复现工具。
Git已验证bundle完整且前置基准存在；先推送本归档，再清理本轮试验分支/工作树。
原有所有核心分支及基准策略保留。本次不存在新整局性能试验结果，勿把旧前缀筛查当成对照成绩。

在本仓库根目录恢复至新名称，不覆盖已有分支或工作：

```text
git bundle verify research/virtual_under_200/enroute-trial.bundle
git fetch research/virtual_under_200/enroute-trial.bundle refs/heads/experiment/q3-enroute-scan:refs/heads/archive-replay/q3-enroute-scan
git worktree add ../q3-enroute-replay archive-replay/q3-enroute-scan
```

随后在恢复出的工作树中运行`diagnostics/ENROUTE_PREFIX_SCREEN.md`给出的只读复现命令。
它需要相邻保留的q3-round3旧基准记录；脚本会验证已登记的16个文件SHA，拒绝换一批数据冒充复现。
基准成本诊断使用相邻保留的q3-runtime-equivalent与主项目历史演练记录，参见BASELINE_DIAGNOSIS.md。

最终源码仅做了新控制器CRLF/LF统一，之后重新通过92项针对性测试。
较早独立控制器评审里的文件SHA对应当时版本，最终源码身份以validation.json和本试验提交为准。
