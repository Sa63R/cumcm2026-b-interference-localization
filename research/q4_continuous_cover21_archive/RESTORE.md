# 恢复 R38 完整研究

永久标签 `archive/q4-r38-continuous-cover21-rejected-20260912` 指向 `cc204db85877dccdf6f4bfccbe93034c653657ea`，已推送，禁止删除或改指向。本目录仅为读者副本；完整源码、研究/结果的 Git 版本与诊断在标签中；唯一原工作区换行字节例外保存在本目录 working-bytes 中。公开摘要采用workspace相对路径，不改原证据字节。

已从新空bare直接从远端depth-1 fetch该标签：目标对象事前不存在；完整tree身份、49个变更Git blob、49个基础源码均逐字节核验。47个本轮原研究/结果文件也已核验；若MANIFEST的working_byte_overrides非空，所列原工作区字节另存本归档，恢复时须覆盖对应文件，其余文件与标签字节相同。恢复仓库 `q3-v1-artifacts/archive-checks-20260912/continuous-cover21-tag-r38.git` 保留。冻结运行文件数量为无新增策略运行文件（仅公开几何工具）；详细raw/QA/proof计数和SHA见MANIFEST、ORIGINAL_EVIDENCE、RESTORED_BLOBS。

本恢复副本为完整浅层tree，未使用本地共享对象；包含目标树全部blob。

在确认新恢复目录不存在后：

```text
git fetch origin refs/tags/archive/q4-r38-continuous-cover21-rejected-20260912:refs/tags/archive/q4-r38-continuous-cover21-rejected-20260912
git -c core.autocrlf=false worktree add --detach ../q4-r38-continuous-cover21-restored archive/q4-r38-continuous-cover21-rejected-20260912
git -C ../q4-r38-continuous-cover21-restored rev-parse HEAD
```

上述命令不运行模拟器。原路径分析脚本应从恢复树执行。当前状态为 archived_and_removed。core 归档 `e642e3dd1e3db07ce09a3c023b17cecd132c26b3` 推送完成后，已用不带 force 的 git worktree remove 移除原树，并按精确 HEAD lease 删除该实验分支；永久标签及独立恢复仓库保留。删除窗口其他 refs 零变化，10 个核心保护分支不变；证据见 MANIFEST 与 CLEANUP。未通过开发/几何门槛的研究不能写成独立验证失败或全体方法不可能，见RESULTS。

原控制台日志的精确字节恢复（从 core `q4-round2` 目录执行，先完成上面的新恢复树创建）：

```powershell
Copy-Item -LiteralPath 'research/q4_continuous_cover21_archive/working-bytes/research/q4_continuous_cover21/optimization-console.txt' -Destination '../q4-r38-continuous-cover21-restored/research/q4_continuous_cover21/optimization-console.txt'
```

该文件原工作区 SHA256 为 `604faa7972af6961af33d35e99426f9282bfe8b156b1e11ea5279252f8aaf7dd`；标签内 LF blob 为 `eed980c7b091ee9936134d43cc6a90f632ddc9e345e1eef9b10e1d5248112c91`。唯一差别为 12 处 CRLF/LF；所有 proof、working-set、源码字节不受影响。没有修改原标签或原证据。
