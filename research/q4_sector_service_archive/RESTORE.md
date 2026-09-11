# 恢复R26分组覆盖访问研究

完整HEAD `1fb1769c3056b42aea0a32d41265b590dd000655`，唯一前置提交为可靠R12 `81aa6e1a1231a2358cf098fbba3fdd0557b540ef`。完整增量包为 `../q4_round2/archives/sector-service-rejected.bundle`，SHA256 `5928e874f78ebab520c31649094ba8a0478e9cbc53ac5bbfd46ead85e910923d`，大小83344901字节。当前仅准备归档，状态 **archived_pending_cleanup**；工作树、本地与远程分支均保留，等待root核对并提交推送归档后再处理清理。

已用本轮专用全新bare仓库，仅depth-1取得R12（可达提交数1），确认目标HEAD不存在，再从bundle恢复。HEAD/tree一致；315个新增或修改blob、69个冻结文件、368个原始文件的Git对象和SHA256逐字节核验通过。R12全部49个src文件均不变，明确含44个.py及5个.gitkeep。结果目录全部270个文件、旧烟测全部7个文件均核对，不只比较JSON语义。

在含R12提交的q4-round2仓库中，先确认恢复分支与目录不存在，再执行：

```text
git bundle verify research/q4_round2/archives/sector-service-rejected.bundle
git fetch research/q4_round2/archives/sector-service-rejected.bundle refs/heads/experiment/q4-r26-sector-service:refs/heads/restored/q4-r26-sector-service
git -c core.autocrlf=false worktree add ../q4-r26-sector-service-restored restored/q4-r26-sector-service
git -C ../q4-r26-sector-service-restored rev-parse HEAD
```

这些命令仅恢复本地Git文件，不运行仿真。完整bundle含两候选238局原始记录、四份首次审计、四批source.zip/manifest/freeze、两spec和全部八份计划，以及方法、图、分析脚本、两次旧621001烟测。238次策略运行对应119个不同开发场景，不能当238个独立场景。6322001/6324001独立场景未打开。可读副本不重复原始records；分析脚本应放回完整恢复仓库原research路径运行。

sector_1随机/压力mean(T/N)为541.461/547.422秒，T/LB为3.357436/3.954891；sector_3为532.096/541.134秒，T/LB为3.301067/3.907169。两候选均全清全审，但都未通过500开发继续门槛，460目标未达成。完整均时、下界、逐局分量及口径见原样RESULTS与metrics，不能与不同630案例伪作逐局因果比较。MANIFEST、RESTORATION_CHECK、RESTORED_BLOBS、ORIGINAL_EVIDENCE及BASE_SRC_BYTE_CHECK保存恢复证据。
