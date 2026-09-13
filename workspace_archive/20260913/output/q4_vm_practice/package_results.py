from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json, hashlib, zipfile

p = Path(__file__).resolve().parent
data = json.loads((p/'official_summary.json').read_text())
assert len(data['runs']) == 15 and data['audit'] == 'passed'
f = p/'environment_manifest.json'
r = json.loads(f.read_text())
r.update(updated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
         official_practice_status='completed_and_audited',
         completed_practice_runs=15, completed_clear_count=181,
         audit_completed_practice_runs=15, audit_completed_clear_count=181,
         official_behavior_logs=15, formal_attempts_started_by_agent=0)
r['known_model_limitation'] = 'Frozen prior conditions on both types existing; official rollout-03 contains 12 directional and 0 omnidirectional sources.'
r['raw_evidence_archive_sha256'] = hashlib.sha256((p/'q4_completed_evidence.zip').read_bytes()).hexdigest()
f.write_text(json.dumps(r, ensure_ascii=False, indent=2))
files = {}
for name in ['official_summary.json','official_ui_observations.jsonl','environment_manifest.json','source_audit.json','v4_case_label_correction.json','第四问_官方演练15场实测报告.md']:
    files['results/'+name] = (p/name).read_bytes()
for f in (p/'evidence').rglob('*'):
    if f.is_file():
        files['results/'+str(f.relative_to(p))] = f.read_bytes()
for f in p.glob('observations_batch_*.json'):
    files['results/'+f.name] = f.read_bytes()
with zipfile.ZipFile(p/'q4_practice_bundle.zip') as z:
    for n in z.namelist():
        if n.startswith('code/') and n.endswith('.py'):
            files[n] = Path(n).read_bytes()
for f in Path('code/experiments/q4_official_practice').iterdir():
    if f.suffix in ('.py','.md'):
        files[str(f)] = f.read_bytes()
for f in Path('code/experiments/q4_comparison').rglob('*.md'):
    files[str(f)] = f.read_bytes()
files['第四问_官方演练15场实测报告.md'] = (p/'第四问_官方演练15场实测报告.md').read_bytes()
files['此前本地配对实测报告.md'] = Path('output/q4_local_comparison/第四问_本地实测报告.md').read_bytes()
files['README.md'] = '''# 第四问官方演练数据包

五种方法各 3 场，15 场、181 个源全部清除。先读“第四问_官方演练15场实测报告.md”。

- results/evidence：每场原始请求、结果与官方行为日志。
- results/official_summary.json：逐场与分方法描述性统计。
- results/official_ui_observations.jsonl：人工核对的官方结束反馈；字符更正有记录。
- code：运行代码、适配检查、日志收集与统计工具。
- 此前本地配对实测报告.md：另一次合成实验，不能与官方数据混为一谈。

只做第四问演练，每次运行前都必须在模拟器界面核对模式。Python 的 --practice-confirmed 是操作者断言，不是服务器查询结果。密码不在本包中。

在本目录重新审计：

    python3 code/experiments/q4_official_practice/summarize_results.py results
    python3 code/experiments/q4_official_practice/make_report.py results

独立 Windows Python 和官方模拟器保留在虚拟机 Downloads/Q4Practice；本包侧重代码和测试证据，不重复打包运行时。
'''.encode()
files['SHA256SUMS.txt'] = ''.join(hashlib.sha256(v).hexdigest()+'  '+k+'\n' for k,v in sorted(files.items())).encode()
archive = p/'第四问_官方演练15场_代码与数据.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for k,v in sorted(files.items()):
        z.writestr(k,v)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert sum(n.endswith('/result.json') for n in z.namelist()) == 15
    assert sum(n.endswith('.jlog') for n in z.namelist()) == 15
    for line in z.read('SHA256SUMS.txt').decode().splitlines():
        digest,name = line.split('  ',1)
        assert hashlib.sha256(z.read(name)).hexdigest() == digest
print(json.dumps({'archive':str(archive),'files':len(files),'bytes':archive.stat().st_size,
                  'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
                  'audit':'15 result files, 15 official logs; all hashes match'},ensure_ascii=False,indent=2))
