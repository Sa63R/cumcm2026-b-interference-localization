# 在本地使用其他 agent 修改论文

工作目录：`/Users/zephyrr/竞赛/26国赛/数模/Overleaf论文`

在编辑器或其他 agent 中直接打开这个文件夹即可。请先读 `AGENTS.md`，
它列出了各章节的位置和校验方法。所有源码、图片、参考论文 PDF 和 Git 历史均已拉取。

可直接给 agent 这段指令：

> 请在 `/Users/zephyrr/竞赛/26国赛/数模/Overleaf论文` 中工作，先阅读 AGENTS.md。
> 按我的后续要求修改对应章节，保留分文件结构，修改完成后运行 python3 build_local.py 并检查 PDF。

## 编译与预览

```sh
cd '/Users/zephyrr/竞赛/26国赛/数模/Overleaf论文'
python3 build_local.py
open .build/main.pdf
```

此电脑已有 MacTeX。脚本优先使用 PATH 中的 TeX 工具，也会查找 `/Library/TeX/texbin/`。
它在临时编译副本中把字体名称映射为同一字体的 TeX Live 文件名，以适配本机字体查找；
提交用源码和 Overleaf 字体设置保持原样。脚本使用 latexmk 自动完成交叉引用所需的编译轮次。

## 查看修改与同步

```sh
git status --short
git diff --stat
git diff --check
```

`origin` 连接原 Overleaf 项目，保留完整 Git 历史。本地修改不会自动上传。
之后要同步时，让 agent 检查本地差异、整合 Overleaf 的最新变化，再提交并推送即可。
Git 认证沿用本机已有的 macOS 钥匙串配置，本目录没有存放访问令牌。

本地入口以此文件夹为准。`output/overleaf/分章节项目` 是先前的交付快照，
连接器在 `.codex/mcp/` 下的目录是内部缓存，都不是当前本地编辑目录。
