"""Verify downloaded Q3 practice evidence and produce a reproducible report."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
METHODS = {
    'phased': '分阶段搜索与清除',
    'joint': '初版联合策略',
    'v2': 'v2 完整组合',
    'v3': 'v3 默认',
    'v3_origin20': 'v3 保留原点扫描',
    'optical': 'v3 有限提前光学尝试',
    'scenario': '新：多目标场景预测',
    'future_cover': '新：搜索与清除位置联合设计',
    'scenario_future': '新：两方向组合',
}


def main():
    archive = OUT / 'q3_evidence.zip'
    index = json.loads((OUT / 'q3_evidence_index.json').read_text())
    observed = {r['case']: r for r in json.loads((OUT / 'ui_observations.json').read_text())}
    assert archive.stat().st_size == index['archive_bytes']
    rows = []
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(set(z.namelist())) == len(index['files'])
        for f in index['files']:
            name = f['archive_path'].replace('\\', '/')
            data = z.read(name)
            assert len(data) == f['bytes']
            assert hashlib.sha256(data).hexdigest() == f['sha256'], name
            target = (OUT / 'evidence' / name).resolve()
            assert target.is_relative_to((OUT / 'evidence').resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        for item in index['results']:
            name = item['path'].replace('\\', '/')
            r = json.loads(z.read(name))
            case = r['case_label']
            u = observed[case]
            assert r['method'] == u['method']
            assert r['status'] == 'policy_completed_and_exited'
            assert r['mode'] == 'operator_confirmed_q3_practice'
            assert r['enter_response']['accepted'] and r['exit_response']['accepted']
            assert r['exit_response']['exit_reason'] == 'user_exit'
            assert r['client_state']['session'] == 'exited'
            assert r['pending_request'] is None and not r.get('exception')
            assert r['policy']['complete_channel_certificate']
            assert len(r['policy']['cleared']) + len(r['policy']['absent']) == 20
            if u.get('screenshot'):
                assert (OUT / u['screenshot']).is_file()
            events = [json.loads(s) for s in z.read(name.replace('result.json', 'requests.jsonl')).decode().splitlines()]
            requests = {e['payload']['request_id']: e for e in events if e['event'] == 'request'}
            responses = {e['request_id']: e for e in events if e['event'] == 'response'}
            assert set(requests) == set(responses)
            assert all(e['http_status'] == 200 and e['response']['accepted'] for e in responses.values())
            clear_results = Counter(e['response'].get('clear_result') for key, e in responses.items() if requests[key]['path'] == '/clear')
            total = u['official_total_sources']
            assert clear_results['success'] == total == r['client_state']['cleared_count'] == len(r['policy']['cleared'])
            failed = sum(clear_results.values()) - clear_results['success']
            assert failed == sum(s['failed_clear_count'] for s in r['client_state']['sources'].values())
            states = [e['state'] for e in events if e['event'] == 'state']
            cost_error = max(abs(s['estimated_total_virtual_time_s'] - s['virtual_time_s']) for s in states)
            assert cost_error <= 1e-5
            assert all(a['virtual_time_s'] <= b['virtual_time_s'] for a, b in zip(states, states[1:]))
            official = [n for n in z.namelist() if n.startswith('official_logs/') and n.endswith('-' + case + '.jlog')]
            assert len(official) == 1
            assert len(z.read(official[0])) > 0
            assert json.loads((OUT / 'live_cases' / (case + '_' + r['method'] + '.json')).read_text()) == r
            rows.append(dict(method=r['method'], name=METHODS[r['method']], case=case,
                             cleared=total, total=total, failed_clear_attempts=failed,
                             virtual_s=r['client_state']['virtual_time_s'],
                             seconds_per_source=r['seconds_per_accepted_clear'],
                             wall_s=r['wall_seconds'], policy_wall_s=r['policy']['policy_wall_s'],
                             request_wall_s=r['policy']['request_wall_s'], nonrequest_wall_s=r['policy']['nonrequest_wall_s'],
                             accepted_requests=len(responses), max_cost_error_s=cost_error,
                             result='evidence/' + name, official_log='evidence/' + official[0]))
        matched = {Path(r['official_log']).name for r in rows}
        excluded = [n for n in z.namelist() if n.startswith('official_logs/') and Path(n).name not in matched]
    assert len(rows) == 9 and {r['method'] for r in rows} == set(METHODS)
    rows.sort(key=lambda r: list(METHODS).index(r['method']))
    summary = dict(recorded_at=datetime.now().astimezone().isoformat(), status='verified',
                   mode='Q3 practice', runs=len(rows), paired=False, runs_per_method=1,
                   total_cleared=sum(r['cleared'] for r in rows),
                   failed_clear_attempts=sum(r['failed_clear_attempts'] for r in rows),
                   files_hash_verified=len(index['files']), archive_crc_verified=True,
                   archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                   excluded_unassigned_logs=excluded, rows=rows)
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    lines = [
        '# 第三问：UTM 内官方模拟器演练结果', '',
        '2026-09-12 在现有 Windows 11 ARM 虚拟机中实际运行。九种方法各完成一轮，共 113 个干扰源全部清除，全部正常发送 `/exit`，官方行为日志均已保存。本报告是演练结果。', '',
        '**这九轮使用九个不同的随机案例，每种方法仅一例，因此下表用于验证运行速度和接口可用性，不能据此认定策略优劣或可靠排名。** 本地 400 组相同案例的配对比较见 [本地速度对比](../q3_comparison/本地速度对比.md)。', '',
        '## 官方演练观测', '',
        '| 方法 | 清除/总数 | 虚拟总秒 | 虚拟秒/源 | 实际执行秒 | 非请求部分秒 | 清除失败尝试 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for r in rows:
        lines.append(f"| {r['name']} | {r['cleared']}/{r['total']} | {r['virtual_s']:.3f} | {r['seconds_per_source']:.3f} | {r['wall_s']:.3f} | {r['nonrequest_wall_s']:.3f} | {r['failed_clear_attempts']} |")
    lines += ['',
        '实际执行时间覆盖创建客户端、`/enter`、策略运行、`/exit` 和关闭客户端，包含接口通信；不含 Python 进程启动、模块导入、模拟器打开、登录、准备案例和界面操作。非请求部分为策略墙钟时间减去策略中的接口请求墙钟时间，包含规划、Python 调度及适配开销，并非纯 CPU 时间。计时使用单调时钟。', '',
        '虚拟时间来自官方已接受响应，包含移动、切频、检测和清除费用。optical 的一次失败光学尝试按规则计入 3 秒；最终仍清除了该例全部 15 个源。实际执行时间为 1.334–3.299 秒/例，证明这些实现可以在当前虚拟机中快速完成本批案例；它不是多次重复后的稳定延迟估计。', '',
        '## 结合本地配对比较的判断', '',
        '本地 400 组配对自建案例中，v3 默认为 242.796 虚拟秒/源，有限提前光学尝试为 241.736，平均改善 1.060 秒/源。新场景预测为 251.860，当前实现比 v3 多用 9.064 秒/源；联合位置设计为 242.757，改善仅 0.038 秒/源且配对区间包含零；组合版为 251.950。现有证据支持继续以 v3/有限光学尝试作为基线，当前两个新方向尚未显示明确收益。这些比较的分布和局限详见本地报告，不将自建结果混作官方成绩。', '',
        '## 环境与验证', '',
        '- UTM：现有 `Quartus9-Windows11-ARM`；Windows 11 ARM，Python 3.13.15 ARM64、NumPy 2.5.3，未启用 Numba。',
        '- 官方程序 SHA-256：`2373b9e7af83735a04309e2983eb433ec46faf7e0b8494410ce7fded2a297c27`，已与用户下载包内程序核对。',
        '- 接入前的适配测试验证九种策略与本地对应案例的耗时一致，并检查清除不切频及失败清除计费。',
        '- 本次逐轮核对官方界面总源数、成功清除响应、20 频道完成证书、正常退出响应、完整请求日志和案例对应的官方 `.jlog`。所有接口请求均获接受，未发生异常退出。',
        f"- 对 {summary['files_hash_verified']} 个归档文件逐个校验大小及 SHA-256，ZIP CRC 检查通过。客户端费用重算与官方虚拟时间的最大绝对差为 {max(r['max_cost_error_s'] for r in rows):.9f} 秒。",
        '- 官方 `.jlog` 保留原始加密文件，未解密。干扰源总数取自演练完成界面，清除数和时间取自公开接口响应。',
        '- 切换后，截图与点击由 Windows 登录会话内的临时脚本执行，经 UTM guest agent 传输文件，不发送 Mac 鼠标键盘事件。', '',
        '## 案例与文件', '',
        '| 方法 | 官方案例编号 | 结果 | 官方日志 |', '|---|---|---|---|',
    ]
    for r in rows:
        lines.append(f"| {r['method']} | `{r['case']}` | [JSON](results/{r['result']}) | [jlog](results/{r['official_log']}) |")
    lines += ['',
        '另有一条未运行策略、在进入接口前结束的演练记录 `9G4U-6EVR-ZN9J-CCDN`，没有对应方法结果，已保留原始日志并从九种方法统计中排除。', '',
        '完整归档：[q3_evidence.zip](results/q3_evidence.zip)；校验清单：[q3_evidence_index.json](results/q3_evidence_index.json)；机器可读汇总：[summary.json](results/summary.json)；界面核对记录：[ui_observations.json](results/ui_observations.json)。', '',
        f"归档 SHA-256：`{summary['archive_sha256']}`。", '',
        '本报告由 `python3 summarize.py` 从已下载的原始结果重建，并在生成前校验数据与日志。', '',
    ]
    (ROOT / '官方演练结果.md').write_text('\n'.join(lines))
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, ensure_ascii=False, indent=2))
    for r in rows:
        print(r['method'], r['cleared'], round(r['seconds_per_source'], 3), round(r['wall_s'], 3))


if __name__ == '__main__':
    main()
