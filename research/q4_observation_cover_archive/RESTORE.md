# 恢复R29观测覆盖研究

本次用已推送的固定Git标签保存完整源码和238次开发记录，避免再往主研究分支重复加入大型bundle。标签为 `archive/q4-r29-observation-cover-rejected-20260912`，精确提交 `32587ff2f16ac9494b4dc99bb9be446e4814dc35`。不能删除或改指向这个归档标签。工作树/分支清理状态见MANIFEST及archives/observation-cover-rejected.json。

归档已随 core 提交 `a15a40e6937bbe00473c54e72305b29a268770fe` 推送。本轮工作树及本地、远程实验分支已清理，状态为 `archived_and_removed`；归档标签、47份读者副本及独立恢复 bare 均保留。删除前后引用清单与精确 lease 命令见 `CLEANUP.json`：只有本轮分支引用被删除，10个保护分支及其他引用未变。

已从全新空bare仓库通过depth-1标签fetch恢复，先确认目标不存在；目标完整tree、所有变更blob、69运行身份、49基础src和全部本轮原始文件逐字节核验。远端标签单独核对指向同一提交。完整记录由标签保留；本目录复制方法、结果、计划、图及几何证据，原样复制文件SHA在MANIFEST，全部原始文件SHA在ORIGINAL_EVIDENCE。

在原仓库确认新恢复目录尚不存在后：

```text
git fetch origin refs/tags/archive/q4-r29-observation-cover-rejected-20260912:refs/tags/archive/q4-r29-observation-cover-rejected-20260912
git -c core.autocrlf=false worktree add --detach ../q4-r29-observation-cover-restored archive/q4-r29-observation-cover-rejected-20260912
git -C ../q4-r29-observation-cover-restored rev-parse HEAD
```

恢复是本地文件操作，不运行模拟器。统计/绘图脚本应在恢复仓库原research路径执行；不是从这个归档副本直接运行。两候选四批全部清除/审计通过，但主均值均高于500，选择器拒绝晋级，未打开633独立场景。详见RESULTS，不能用公开双接收几何性质冒充耗时改善。
