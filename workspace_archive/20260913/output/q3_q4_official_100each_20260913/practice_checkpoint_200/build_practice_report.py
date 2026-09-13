from pathlib import Path
import argparse,hashlib,json,statistics,zipfile,shutil
p=argparse.ArgumentParser();p.add_argument('batch',type=Path);p.add_argument('--expected',type=int,default=200);a=p.parse_args()
b=a.batch;s=json.loads((b/'independent_audit_summary.json').read_text());rows=json.loads((b/'completed.json').read_text('utf-8-sig'))
assert s['passed']==s['audited']==len(rows)==a.expected
lines=['# Q3 原点扫描法、Q4 V6 Lite 官方演练结果','','两版均沿用六轮正式测试中的源码与参数，使用 Windows UTM 虚拟机里的官方模拟器 1.1。通过虚拟机内部的后台输入启动演练、通过官方 HTTP 接口执行算法，没有使用 Mac 鼠标操作。','','每条记录均对应新的官方演练案例，不包含之前的演练或六轮正式测试。每轮都核对官方源数、成功清除数、正常退出和加密日志 SHA-256；随后独立核对请求—响应—状态、逐步计时、完整性证书和源码清单。','','| 指标 | Q3 原点扫描法 | Q4 V6 Lite |','|---|---:|---:|']
q3=s['questions']['3'];q4=s['questions']['4']
fields=[('演练完成数','cases','d'),('官方源总数','total_sources','d'),('成功清除总数','cleared','d'),('总虚拟用时 / 总清除数（秒/源）','pooled_s_per_source','.6f'),('逐案例秒/源的平均值','case_mean_s_per_source','.6f'),('逐案例秒/源的中位数','case_median_s_per_source','.6f'),('逐案例秒/源的标准差','case_stdev_s_per_source','.6f'),('最小秒/源','case_min_s_per_source','.6f'),('最大秒/源','case_max_s_per_source','.6f'),('入口记录的每轮实际平均用时（秒）','mean_wall_s','.6f'),('指令总数','actions','d'),('未命中目标的光学清除尝试数','failed_clear_attempts','d')]
for label,key,fmt in fields:lines.append(f"| {label} | {q3[key]:{fmt}} | {q4[key]:{fmt}} |")
lines+=['','“秒/源”的主口径为全部案例总虚拟时间除以全部成功清除数。实际用时采用各入口输出的 `wall_seconds`，不含 Python 导入、启动演练前的官方倒计时和批量工具的界面等待。未命中的光学尝试是算法正常动作，其成本已计入总用时。`result.json` 的 `mode=unspecified` 表示入口未接收可选的模式标签；实际演练身份通过官方 `practice-*` 结果和界面核对确认。','','Q3 与 Q4 的题设不同，两题使用各自的官方随机案例，因此上表用于描述各自表现，不能作为同一道题上两种算法的对照实验。','','批量过程曾因第 12 轮截图 OCR 把“10”读成“1”而暂停；原图、官方结果及全向 5 + 定向 5 的计数均确认为 10，修正后台核对逻辑后从原轮结果继续，没有重跑或替换案例。算法源码与参数未因该问题修改。第 112 轮准备阶段还发生一次官方连接超时，界面始终没有有效案例编号、机器狗未进入；保留错误截图，确认六份正式日志界面均已上传后重启登录，继续剩余演练。','','文件说明：','','- `runs/`：每轮完整 `requests.jsonl`、`result.json`、官方完成核对及独立审计。','- `official_logs/`：每轮官方 `.jlog`、`.result.json` 和存在时的 `.psum` 原件。','- `screenshots/`：演练列表、准备状态、完成提示、返回页面截图与 OCR 原文。','- `completed.json`：逐案例总表；`independent_audit_summary.json`：独立审计和统计汇总。','- `audit_practice.py`、`expected_versions.json`：复核脚本和正式测试版本清单，仅分析文件，不发起测试。','- `export_manifest.json`：原始导出文件校验；`文件校验清单.json`：交付文件校验。','','## 逐轮结果','','| 题目 | 轮次 | 官方案例 | 源数 | 清除数 | 虚拟秒 | 秒/源 | 程序实际秒 |','|---|---:|---|---:|---:|---:|---:|---:|']
for r in sorted(rows,key=lambda r:(r['question'],r['ordinal'])):
 lines.append(f"| Q{r['question']} | {r['ordinal']} | {r['case']} | {r['total']} | {r['cleared']} | {r['virtual_s']:.6f} | {r['seconds_per_source']:.6f} | {r['wall_s']:.6f} |")
group_lines=['## 按官方源数分组','','| 题目 | 源数 | 案例数 | 平均虚拟秒/轮 | 平均秒/源 |','|---|---:|---:|---:|---:|']
for q in (3,4):
 for n in range(10,17):
  rr=[r for r in rows if r['question']==q and r['total']==n]
  if rr:group_lines.append(f"| Q{q} | {n} | {len(rr)} | {statistics.mean(r['virtual_s'] for r in rr):.3f} | {statistics.mean(r['seconds_per_source'] for r in rr):.3f} |")
group_lines+=['','该分组用于观察目标数量与耗时的关系；不同分组的几何场景、方向性也不同，不能单独归因为源数。','']
ix=lines.index('## 逐轮结果');lines[ix:ix]=group_lines
(b/'结果说明.md').write_text('\n'.join(lines)+'\n')
for name in ['audit_practice.py','expected_versions.json','bulk_q3_q4_100each.ps1','monitor_history.jsonl','build_practice_report.py']:
 src=Path(__file__).with_name(name)
 if src.exists() and src.resolve()!= (b/name).resolve():shutil.copy2(src,b/name)
for recovery_name in ['recovery_012','recovery_112','wall_outlier_105']:
 recovery=Path(__file__).with_name(recovery_name)
 if recovery.is_dir():shutil.copytree(recovery,b/recovery_name,dirs_exist_ok=True)
manifest={str(f.relative_to(b)):{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(b.rglob('*')) if f.is_file() and f.name!='文件校验清单.json'}
(b/'文件校验清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(b/'结果说明.md');print('files',len(manifest)+1)
