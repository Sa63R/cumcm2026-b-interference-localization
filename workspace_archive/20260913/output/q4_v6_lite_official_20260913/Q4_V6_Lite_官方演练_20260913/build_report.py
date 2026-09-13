"""Write the final official-practice report from independently audited results."""
import json,statistics,hashlib,shutil,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'audited_results.json').read_text())
assert len(rows)==11 and all(x['all_cleared'] for x in rows)
values=[x['seconds_per_source'] for x in rows]
total=sum(x['total'] for x in rows); virtual=sum(x['virtual_s'] for x in rows)
summary=dict(cases=len(rows),all_cleared_cases=len(rows),sources=total,virtual_total_s=virtual,
    case_mean_s_per_source=statistics.mean(values),pooled_s_per_source=virtual/total,
    case_median_s_per_source=statistics.median(values),case_sample_sd_s_per_source=statistics.stdev(values),
    case_min_s_per_source=min(values),case_max_s_per_source=max(values),
    wall_mean_s=statistics.mean(x['wall_s'] for x in rows),wall_min_s=min(x['wall_s'] for x in rows),
    wall_max_s=max(x['wall_s'] for x in rows),cpu_mean_s=statistics.mean(x['cpu_s'] for x in rows),
    failed_clear_attempts=sum(x['failed_clear_attempts'] for x in rows))
(ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
lines=['# V6 Lite 官方第四问演练：11 轮实测','',
    f'2026 年 9 月 13 日（北京时间），在 UTM 的 Windows 11 ARM 虚拟机内完成首轮及追加 10 轮官方第四问演练，共 {total}/{total} 个源全部清除，11/11 轮正常发送 `/exit`。官方模拟器版本为 v1.1。', '',
    '固定运行 GitHub 已发布的 V6 Lite，提交 `ed795304a37a5cce8541c046a0ec0a0e4e88ba6d`。没有按中间成绩更换算法或参数；接入层沿用此前本地验证的 1.005° 示向度误差界。后续控制使用虚拟机内的后台助手，通过 `utmctl` 传递命令和文件，不发送 Mac 鼠标键盘事件。', '',
    '## 速度结果','',
    '| 指标 | 实测 |','|---|---:|',
    f'| 各轮“总虚拟时间 / 源数”的算术平均 | {summary["case_mean_s_per_source"]:.3f} 秒/源 |',
    f'| 全部虚拟时间 / 全部源数 | {summary["pooled_s_per_source"]:.3f} 秒/源 |',
    f'| 各轮中位数 | {summary["case_median_s_per_source"]:.3f} 秒/源 |',
    f'| 各轮范围 | {min(values):.3f}–{max(values):.3f} 秒/源 |',
    f'| 各轮样本标准差 | {summary["case_sample_sd_s_per_source"]:.3f} 秒/源 |',
    f'| 程序实际运行时间均值 | {summary["wall_mean_s"]:.3f} 秒/轮 |',
    f'| 程序实际运行时间范围 | {summary["wall_min_s"]:.3f}–{summary["wall_max_s"]:.3f} 秒/轮 |',
    f'| 进程 CPU 时间均值 | {summary["cpu_mean_s"]:.3f} 秒/轮 |',
    f'| 未成功的清除尝试 | {summary["failed_clear_attempts"]} 次，已计入虚拟耗时 |','',
    '虚拟时间包含移动、切换频道、测量、光学定位及清除成本，并包含清除最后一个源之后为排除遗漏而继续覆盖搜索的时间。实际运行时间是 `/enter` 到 `/exit` 的程序耗时，不含登录、等待官方分配案例、人工核验或文件传输。', '',
    '案例耗时受布局、定向比例和源总数共同影响。清除达到 16 个源时可用题目源数上界停止；不足 16 个源时还须完成覆盖搜索来排除遗漏。因此不能把单例差异都解释为定位快慢。', '',
    '## 逐轮结果','',
    '| 轮次 | 官方案例编号 | 全向/定向 | 清除 | 总虚拟秒 | 秒/源 | 实际运行秒 |','|---:|---|---:|---:|---:|---:|---:|']
for n,r in enumerate(rows,1):
    lines.append(f'| {n} | {r["case"]} | {r["omni"]}/{r["directional"]} | {r["total"]}/{r["total"]} | {r["virtual_s"]:.3f} | {r["seconds_per_source"]:.3f} | {r["wall_s"]:.3f} |')
lines += ['', '## 核验与边界','',
    '- 每轮源总数同时核对官方完成界面、官方 `.result.json` 和成功清除响应；类型数量以官方 `.result.json` 为准，保留完成界面截图供复核。加密 `.jlog` 原样备份，并逐一验证其 SHA-256 等于官方结果文件记录的值，没有解密。',
    '- 独立审计逐条请求与响应，检查请求序号、接受状态、时间增量、成功清除频道、正常退出，以及不足 16 个源时剩余频道是否覆盖全部 21 个扫描站。',
    '- 首轮界面编号曾被人工读成 `EHKR-PC7N-PNWY-K6A9`；官方文件中的准确编号为 `EHKR-PC7N-PNVY-K6A9`。保留原始运行目录及原始记录，汇总统一使用官方文件编号。',
    '- 案例标签只是采集记录，不参与策略决策。就绪检查同时要求第四问演练、尚未进入、等待机器狗进入、有效剩余时间和非占位编号；编号识别不完整时先使用内部标签，结束后核对官方日志编号。',
    '- 追加批次两次创建案例时遇到“等待测试状态 / XXXX”，第一次恢复列表后官方界面显示无法连接服务器。两次均没有运行策略，未计入算法成绩。恢复网络连接并正常重启、重新登录后继续。过程中修复了 OCR 判定、旧版 PowerShell 读取已完成记录和文件锁兼容问题；已生成的案例均原地接续，没有重跑算法来挑选成绩，故障现场另行保存。',
    '- 接口适配提前通过 Windows 内四组本地端到端检查：同一 Windows 环境内直接调用与 HTTP 调用动作、响应和时间一致。与 Mac 相比，两组压力样例出现路线差异，因此不声称跨平台逐位相同。',
    '- 这是 11 个新生成案例的官方演练观测，不是正式测试。未在相同案例上运行 V4 或完整 V6，不能凭本批成绩认定 Lite 比两者更快。','',
    '官方程序 SHA-256：`2373b9e7af83735a04309e2983eb433ec46faf7e0b8494410ce7fded2a297c27`。', '',
    '## 文件','',
    '- `audited_results.json`：每轮独立审计结果、分项成本与日志校验值。',
    '- `summary.json`：机器可读的汇总指标。',
    '- `official_results/`、`additional_10/runs/`：算法原始请求响应和结果。',
    '- `official_logs/`、`additional_10/official_logs/`：官方加密日志和结果摘要。',
    '- `evidence/`、`additional_10/screenshots/`：官方界面的准备和结束证据。',
    '- `source_bundle.zip`：本次运行的代码、客户端与预检脚本。',
    '- `audit_results.py`：独立审计脚本。','']
(ROOT/'V6_Lite_官方演练11轮报告.md').write_text('\n'.join(lines))
print(json.dumps(summary,ensure_ascii=False,indent=2))
