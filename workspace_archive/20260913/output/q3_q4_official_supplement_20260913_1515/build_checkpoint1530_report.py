"""Publish separately audited supplement and combine with the previous 100+100 batch."""
from pathlib import Path
import argparse,hashlib,json,statistics,shutil
from datetime import datetime,timezone,timedelta
p=argparse.ArgumentParser();p.add_argument('batch',type=Path);a=p.parse_args();b=a.batch
base=Path(__file__).resolve().parent
old=base.parent/'q3_q4_official_100each_20260913'/'practice_checkpoint_200'
rows=json.loads((b/'completed.json').read_text('utf-8-sig'));s=json.loads((b/'independent_audit_summary.json').read_text())
previous=json.loads((old/'completed.json').read_text('utf-8-sig'));olds=json.loads((old/'independent_audit_summary.json').read_text())
assert s['passed']==s['audited']==len(rows)
assert olds['passed']==olds['audited']==len(previous)==200
assert not ({r['case'] for r in rows}&{r['case'] for r in previous})
assert all(r['total']==r['cleared'] for r in rows+previous)
def stats(rr):
 return {'cases':len(rr),'sources':sum(r['total'] for r in rr),'cleared':sum(r['cleared'] for r in rr),'total_virtual_s':sum(r['virtual_s'] for r in rr),'pooled_s_per_source':sum(r['virtual_s'] for r in rr)/sum(r['cleared'] for r in rr),'mean_wall_s':statistics.mean(r['wall_s'] for r in rr),'case_mean_s_per_source':statistics.mean(r['seconds_per_source'] for r in rr),'actions':sum(r['actions'] for r in rr)}
summary={'new':{str(q):stats([r for r in rows if r['question']==q]) for q in (3,4)},'combined':{str(q):stats([r for r in previous+rows if r['question']==q]) for q in (3,4)},'new_independent_audits_passed':s['passed'],'source_versions_identical_to_formal':True}
times=[]
for r in rows:
 run=b/'runs'/f"{r['index']:03d}_q{r['question']}_{r['ordinal']:03d}"
 result=json.loads((run/'result.json').read_text())
 responses=[json.loads(line) for line in (run/'requests.jsonl').read_bytes().splitlines() if b'"event": "response"' in line]
 if not responses:
  responses=[x for x in map(json.loads,(run/'requests.jsonl').read_bytes().splitlines()) if x['event']=='response']
 times.append((responses[0]['response']['real_timestamp_ms'],result['exit_response']['real_timestamp_ms']))
zone=timezone(timedelta(hours=8))
summary['observed_first_enter_beijing']=datetime.fromtimestamp(min(t[0] for t in times)/1000,zone).isoformat()
summary['observed_last_exit_beijing']=datetime.fromtimestamp(max(t[1] for t in times)/1000,zone).isoformat()
summary['all_exited_before_user_deadline']=max(t[1] for t in times)<datetime(2026,9,13,15,30,tzinfo=zone).timestamp()*1000
(b/'补充及累计统计.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
lines=['# Q3、Q4 官方演练补充实验','','本次沿用正式测试时的 Q3 `v3_origin20` 原点扫描法和 Q4 V6 Lite，源码清单与正式测试版本逐文件一致。官方模拟器 1.1 在 UTM Windows 虚拟机中运行，批处理只控制虚拟机内部界面，再通过官方 HTTP 接口执行算法，没有使用 Mac 鼠标。','','用户要求北京时间 2026-09-13 15:30 前能补多少补多少。工具设置 15:29:30 起停止开新轮次，给正在收尾的案例预留时间。客户端页面自身显示的新测试截止时间为 17:30，本次采用用户给出的更早时间。', '', f"官方时间戳记录的首轮进入时间：{summary['observed_first_enter_beijing']}；最后退出时间：{summary['observed_last_exit_beijing']}。",'', '## 本次新增','','| 指标 | Q3 原点扫描法 | Q4 V6 Lite |','|---|---:|---:|']
for label,key,fmt in [('完成轮数','cases','d'),('官方源数','sources','d'),('清除数','cleared','d'),('总虚拟用时 / 总清除数（秒/源）','pooled_s_per_source','.6f'),('入口记录的平均实际用时（秒）','mean_wall_s','.6f'),('指令总数','actions','d')]:
 lines.append(f"| {label} | {summary['new']['3'][key]:{fmt}} | {summary['new']['4'][key]:{fmt}} |")
lines+=['','## 加上上一批各 100 轮后的累计统计','','| 题目 | 累计轮数 | 官方源数 / 清除数 | 虚拟秒/源 | 平均程序实际秒 |','|---|---:|---:|---:|---:|']
for q in (3,4):
 z=summary['combined'][str(q)];lines.append(f"| Q{q} | {z['cases']} | {z['sources']} / {z['cleared']} | {z['pooled_s_per_source']:.6f} | {z['mean_wall_s']:.6f} |")
lines+=['','本次新增案例编号与上一批没有交集，不包含正式测试；原始数据保存在各自批次目录中。累计统计文件同时保留本次与累计两种口径。两问题设不同，随机场景不同，不能把两题的用时直接当作同一道题的算法优劣对照。','','虚拟秒/源采用总虚拟时间除以总清除数。实际秒采用入口 `wall_seconds`，不含 Python 导入、官方准备倒计时以及界面切换等待。未命中的光学尝试属于算法动作，其成本已计入。入口 `mode=unspecified` 只表示未填写可选标签；官方演练身份由 practice 日志元数据和对应界面证据确认。','','每轮独立核对完整请求、响应和状态序列、逐步虚拟计时、最终清除数、覆盖或频道证书、版本清单、官方日志 SHA-256 以及官方结束时间。完整原始 JSONL 与 result.json、官方加密 .jlog 及 sidecar、截图和 OCR 都随包保存。','','启动期间 `utmctl` 的 ScriptingBridge 发生异常，但虚拟机和模拟器进程继续运行。改用 UTM 提供的 AppleScript 接口后成功启动补跑，没有改动算法源码。', '', '## 逐轮结果','','| 题目 | 本次轮次 | 官方案例 | 官方源数 | 清除数 | 虚拟秒 | 秒/源 | 程序实际秒 |','|---|---:|---|---:|---:|---:|---:|---:|']
for r in sorted(rows,key=lambda r:(r['question'],r['ordinal'])):
 lines.append(f"| Q{r['question']} | {r['ordinal']} | {r['case']} | {r['total']} | {r['cleared']} | {r['virtual_s']:.6f} | {r['seconds_per_source']:.6f} | {r['wall_s']:.6f} |")
lines+=['','## 按官方源数分组','','| 题目 | 源数 | 案例数 | 平均虚拟秒/轮 | 总虚拟秒/源 |','|---|---:|---:|---:|---:|']
for q in (3,4):
 for n in range(10,17):
  rr=[r for r in rows if r['question']==q and r['total']==n]
  if rr:lines.append(f"| Q{q} | {n} | {len(rr)} | {statistics.mean(r['virtual_s'] for r in rr):.3f} | {sum(r['virtual_s'] for r in rr)/sum(r['total'] for r in rr):.3f} |")
lines+=['','数量分组同时混有几何分布和方向性差异，不应单独归因为源数。','','可用随包的 `audit_practice.py` 重新复核，它只读取已存文件，不启动测试。']
(b/'结果说明.md').write_text('\n'.join(lines)+'\n')
for name in ('audit_practice.py','expected_versions.json','bulk_supplement.ps1','monitor_history.jsonl','build_supplement_report.py','utm_guest.py'):
 shutil.copy2(base/name,b/name)
shutil.copytree(base/'startup_evidence',b/'startup_evidence',dirs_exist_ok=True)
manifest={str(f.relative_to(b)):{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(b.rglob('*')) if f.is_file() and f.name!='文件校验清单.json'}
(b/'文件校验清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
