# 本地实验归档（2026-09-13）

本次归档把此前只保存在本机的第三问、第四问代码和实验记录纳入仓库。它们包含开发预试、固定种子配对评测、逐例记录、重放材料、官方演练接口与阶段性证据。虚拟环境、Python 缓存和系统临时文件未归档。

## 第三问

- [`experiments/q3_comparison/`](../experiments/q3_comparison/)：九种方法的本地自建仿真对照、400 例与压力案例结果、运行时测量以及保留的上游源码。
- [`experiments/q3_local_top5/`](../experiments/q3_local_top5/)：暂列前五方法在本地复原模拟器上的 2000 个同场景配对案例、逐轮压缩记录、重放审计和串行计时。
- [`experiments/q3_official_practice/`](../experiments/q3_official_practice/)：附件 2 HTTP 接口、UTM/Windows 自动化辅助脚本、九方法官方演练记录及 900 例批量任务的阶段性材料。批量目录是阶段性归档，不能解释为 900 例已经全部完成。

## 第四问

- [`experiments/q4_comparison/`](../experiments/q4_comparison/)：五组方法的本地 60 案例配对实验、开发预试、完整结果和审计脚本。
- [`experiments/q4_speedup/`](../experiments/q4_speedup/)：第二轮提速研究的候选实现、开发记录、计算等价性验证与筛选材料。
- [`experiments/q4_official_practice/`](../experiments/q4_official_practice/)：在已发布 V4/V4 Fast 入口之外补齐本机保留的运行、收集与汇总工具；已有发布文件以远端最新版为准。

## 独立研究分支

`research/q3-state-search` 分支同步保存 clear-region/history-silence 研究代码、地图分布审计、开发/留出/压力实验及其逐例证据。该分支保留研究过程，不并入 `main` 的稳定发布入口。

## 证据边界

- 标注 `LOCAL_ONLY`、本地模拟器或自建仿真的结果不是官方成绩。
- 官方演练的不同场次不是同场景配对，不能直接据此比较方法优劣。
- 原始请求记录可能含队号，但不含登录密码、访问令牌或其他认证凭据。
- 各目录内的 README、清单和审计文件是相应结果的优先解释依据。
