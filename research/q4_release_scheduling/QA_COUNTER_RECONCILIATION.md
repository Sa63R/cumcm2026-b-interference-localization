# 旧QA触发计数修正

294项构造与工具测试先通过，但旧621003、621013各一次实际QA暴露了一个计数边界遗漏：独审已通过物理、实际宏、R12/R8/range/scheduling检查后，把覆盖已结束、release全为0时的ready/nonready混合服务也计入forecast_scheduled_macros；release工具按协议只计仍有未来覆盖任务的混合规划，因此QA最后的一致性断言失败。

两份首次失败审计、原始轨迹、原source.zip、原summary/console与294项测试输出全部保留。只修独审的触发计数，加“covers非空”条件及对应回归；生产策略、参数、release工具、原轨迹不变，案例不重跑。新文件reconciled-audit保存旧/新计数、首次审计SHA、原始轨迹SHA及完整重审结果。最后冻结明确first_audit_passed=false、final_audit_passed=true，不能报告旧QA首次全通过。

该修正发生在637开发场景打开前。之后的开发/独立批次仍必须首次完整审计通过；不会拿已运行的调参案例当作独立验证证据。
