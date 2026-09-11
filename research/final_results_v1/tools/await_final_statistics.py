"""Wait for the completed identity audit, then report immutable final records.

No simulator or scenario generator is called. This is a Linux delivery task,
not a policy, selection rule, or opportunity to change the selected identities.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path('/home/volleyball/q3-research-v1')
TOOLS = ROOT / 'evaluation-tools-0830-git'
RUN = ROOT / 'v1-selection'
METHODS = {'baseline': 'baseline', 'state': 'state-future-cover',
           'rl': 'rl-gae095-u512', 'geo': 'geo-future-cover'}
LABELS = {'baseline': '冻结 rollout 基线', 'state': '状态搜索＋未来覆盖移站',
          'rl': '强化学习 GAE=.95 / u512', 'geo': '几何法＋未来覆盖移站'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def require(value, message):
    if not value:
        raise ValueError(message)


def stop(signum, frame):
    raise TimeoutError(f'Statistics task interrupted by signal {signum}')


def execute(command, deadline, label, status):
    remaining = deadline-time.time()-15
    require(remaining > 10, 'Insufficient deadline budget for '+label)
    child = None
    log = RUN / ('statistics-'+label+'.log')
    before = time.time()
    item = dict(label=label, command=command, log=str(log), started_epoch=before)
    status['steps'].append(item)
    write(RUN/'statistics-status.json', status)
    try:
        with log.open('x', encoding='utf-8') as stream:
            child = subprocess.Popen(command, cwd=TOOLS,
                env=dict(os.environ, PYTHONPATH='src', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLBACKEND='Agg'),
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            item['pid'] = child.pid
            item['returncode'] = child.wait(timeout=remaining)
            require(item['returncode'] == 0, 'Report subprocess failed: '+label)
    except BaseException as exc:
        item['error'] = f'{type(exc).__name__}: {exc}'
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=2)
        raise
    finally:
        item['elapsed_wall_s'] = time.time()-before
        write(RUN/'statistics-status.json', status)


def family_statistics(protocol, audit):
    partition = protocol['partitions']['final_stress']
    families = protocol['final_stress_families']
    rows = {label: read(Path(audit['partitions']['final_stress']['evaluations'][key]['directory'])/'rows.json')
            for label, key in METHODS.items()}
    base = {row['seed']: row for row in rows['baseline']}
    result = dict(scope='Seven fixed stress families, four cases each; descriptive paired statistics, no IID pooling or family confidence claims.',
                  methods=METHODS, families={})
    for index, family in enumerate(families):
        seed_set = {seed for seed in base if (seed-partition['seed_start']) % len(families) == index}
        require(len(seed_set) == 4, 'Stress family does not have four declared cases')
        result['families'][family] = {}
        for label, records in rows.items():
            group = [row for row in records if row['seed'] in seed_set]
            require(len(group) == 4, 'Incomplete stress family')
            paired = [dict(seed=row['seed'], case_id=row['case_id'], baseline_s=base[row['seed']]['penalized_time_s'],
                           candidate_s=row['penalized_time_s'], saved_s=base[row['seed']]['penalized_time_s']-row['penalized_time_s'])
                      for row in group]
            mean = statistics.mean(row['penalized_time_s'] for row in group)
            mean_base = statistics.mean(base[seed]['penalized_time_s'] for seed in seed_set)
            result['families'][family][label] = dict(runs=4, successful=sum(row['successful'] for row in group),
                failed_clear_count=sum(row['failed_clear_count'] for row in group), mean_penalized_s=mean,
                mean_saved_s=mean_base-mean, mean_reduction_fraction=1-mean/mean_base,
                wins=sum(row['saved_s']>1e-6 for row in paired), losses=sum(row['saved_s'] < -1e-6 for row in paired),
                ties=sum(abs(row['saved_s'])<=1e-6 for row in paired),
                worst_paired_case=min(paired, key=lambda row: row['saved_s']), paired_cases=paired)
    return result


def interpretation(random, stress, acceptance, family):
    lines = ['# 首版独立最终统计结果', '',
             '候选在扩展集上选定后保持不变；本报告分别汇总 256 个随机案例和 28 个压力案例。这里只评价预先声明的统计门槛，物理、几何证书和下界深审另行提供。', '',
             '| 方法 | 随机全清成功 | 失败清除 | 随机平均秒 | 相对基线节省 | 配对节省 95% 区间（秒） | 随机 P95 比 | 压力 P95 比 | 统计门槛 |',
             '|---|---:|---:|---:|---:|---|---:|---:|---|']
    for label in METHODS:
        method = random['methods'][label]
        if label == 'baseline':
            lines.append(f"| {LABELS[label]} | {method['successful_runs']}/{method['runs']} | {method['failed_clear_count']} | {method['penalized_mean_total_time_s']:.3f} | — | — | 1 | 1 | 参照 |")
            continue
        contrast, stress_contrast = random['comparisons'][label], stress['comparisons'][label]
        ci = contrast['saving_ci95_s']
        passed = acceptance['acceptance'][label]['first_version_practical_target_met']
        lines.append(f"| {LABELS[label]} | {method['successful_runs']}/{method['runs']} | {method['failed_clear_count']} | {method['penalized_mean_total_time_s']:.3f} | {contrast['mean_reduction_fraction']:.3%} | [{ci[0]:.3f}, {ci[1]:.3f}] | {contrast['p95_time_ratio']:.5f} | {stress_contrast['p95_time_ratio']:.5f} | {'通过' if passed else '未通过'} |")
    lines += ['', '“通过”同时要求基线有效、两个完整最终分区均成功且零失败清除、随机均值改善至少 5%、随机配对节省区间下端严格大于 0，以及随机与压力 P95 比均不超过 1.05。P95 比是两个分位数之比，不是逐局比值的分位数。', '']
    for label in ('state', 'rl', 'geo'):
        decision = acceptance['acceptance'][label]
        failed = [key for key, value in decision.items() if value is False and key != 'first_version_practical_target_met']
        worst = random['comparisons'][label]['paired_cases'][0]
        worst_stress = stress['comparisons'][label]['paired_cases'][0]
        lines.append(f"- {LABELS[label]}：{'全部共同统计门槛满足' if not failed else '未满足：'+', '.join(failed)}。随机最不利配对为 {worst['case_id']}（节省 {worst['saved_s']:.3f} 秒）；压力最不利配对为 {worst_stress['case_id']}（节省 {worst_stress['saved_s']:.3f} 秒）。负节省表示退化。")
    lines += ['', '## 各压力家族（每族仅 4 局）', '',
              '| 家族 | 方法 | 成功 | 失败清除 | 平均秒 | 相对基线节省秒 | 赢/平/输 |',
              '|---|---|---:|---:|---:|---:|---:|']
    for name, methods in family['families'].items():
        for label, summary in methods.items():
            lines.append(f"| {name} | {LABELS[label]} | {summary['successful']}/4 | {summary['failed_clear_count']} | {summary['mean_penalized_s']:.3f} | {summary['mean_saved_s']:.3f} | {summary['wins']}/{summary['ties']}/{summary['losses']} |")
    lines += ['', '压力家族是固定构造集合，各族结果用于描述失效与尾部，不能由四局推断普遍可靠性，也不与随机集混合来改善均值或区间。共同报告的区间属于预先声明的逐方向对比，不是事后挑选最佳方向的同时置信保证。并发评估的 wall time 不用于算法速度排名；串行公开案例计时另报。', '']
    return '\n'.join(lines)


def main():
    require(not sys.flags.optimize, 'Do not disable Python assertions in audited reports')
    protocol = read(TOOLS/'research/v1_protocol.json')
    deadline = datetime.fromisoformat(protocol['hard_deadline']).timestamp()
    require(deadline-time.time()>30, 'No deadline budget remains')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, stop)
    signal.setitimer(signal.ITIMER_REAL, deadline-time.time()-10)
    state = dict(kind='final_statistics_delivery', status='waiting_for_identity_audit', steps=[],
                 started_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(__file__), methods=METHODS)
    status_path = RUN/'statistics-status.json'
    require(not status_path.exists(), 'Statistics task already has a status record')
    write(status_path, state)
    try:
        audit_path = RUN/'final-identity-audit.json'
        while True:
            require(time.time()<deadline-120, 'Identity audit did not arrive with sufficient reporting budget')
            if audit_path.exists():
                try:
                    read(audit_path)
                    break
                except json.JSONDecodeError:
                    pass  # Exclusive writer may still be completing the JSON.
            time.sleep(20)
        module_spec = importlib.util.spec_from_file_location('final_delivery_selection', TOOLS/'experiments/research_v1_selection.py')
        selection = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(selection)
        audit = selection.read_sealed(audit_path, 'final_identity_archive_audit')
        require(audit['identity_and_archive_checks_passed'], 'Final identity audit did not pass')
        require(audit['registry_file_sha256']==sha(RUN/'registry.json')
                and audit['selection_file_sha256']==sha(RUN/'selected.json'), 'Registry/selection evidence changed')
        for direction in ('state', 'rl', 'geo'):
            require(audit['selected'][direction]['selected']==METHODS[direction], 'Unexpected selected identity')
        for split in ('final_random', 'final_stress'):
            evidence = audit['partitions'][split]['evaluations']
            require(set(evidence)==set(METHODS.values()), 'Unexpected final method set')
            for key, item in evidence.items():
                path = Path(item['directory'])
                require(path==RUN/'evaluations'/split/key, 'Unexpected final input directory')
                require(sha(path/'manifest.json')==item['evidence']['manifest_sha256']
                        and sha(path/'rows.json')==item['evidence']['rows_sha256'], 'Rows/manifest changed after identity audit')
                for name, expected in item['evidence']['archives_sha256'].items():
                    require(sha(path/name)==expected, 'Case archive changed after identity audit')
        commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=TOOLS, text=True).strip()
        require(commit.startswith('d5c082b'), 'Unexpected common report tool commit')
        state.update(status='reporting', audit_sha256=sha(audit_path), report_tool_commit=commit,
                     report_tool_sha256=sha(TOOLS/'experiments/research_v1_report.py'))
        write(status_path, state)
        report = str(TOOLS/'experiments/research_v1_report.py')
        for split, folder in (('final_random', 'report-random'), ('final_stress', 'report-stress')):
            require(not (RUN/folder).exists(), 'Refuse replacing an existing report: '+folder)
            command = [sys.executable, report, '--baseline', str(RUN/'evaluations'/split/'baseline')]
            for label in ('state','rl','geo'):
                command += ['--candidate', label+'='+str(RUN/'evaluations'/split/METHODS[label])]
            command += ['--output',str(RUN/folder),'--figures']
            execute(command, deadline, folder, state)
        require(not (RUN/'final-acceptance').exists(), 'Refuse replacing final acceptance')
        execute([sys.executable, report, '--final-random-report', str(RUN/'report-random/comparison.json'),
                 '--final-stress-report', str(RUN/'report-stress/comparison.json'),
                 '--output', str(RUN/'final-acceptance')], deadline, 'final-acceptance', state)
        random = read(RUN/'report-random/comparison.json')
        stress = read(RUN/'report-stress/comparison.json')
        acceptance = read(RUN/'final-acceptance/final_acceptance.json')
        families = family_statistics(protocol, audit)
        write(RUN/'final-acceptance/stress-families.json', families)
        (RUN/'final-acceptance/statistical-interpretation.md').write_text(
            interpretation(random, stress, acceptance, families), encoding='utf-8')
        state.update(status='completed', finished_utc=datetime.now(timezone.utc).isoformat(),
                     statistical_acceptance=acceptance['acceptance'],
                     output_sha256={str(p.relative_to(RUN)):sha(p) for folder in ('report-random','report-stress','final-acceptance')
                                    for p in sorted((RUN/folder).iterdir()) if p.is_file()})
    except BaseException as exc:
        state.update(status='failed_or_incomplete', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        write(status_path, state)


if __name__=='__main__':
    main()
