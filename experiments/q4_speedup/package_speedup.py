"""Package only Q4 source code and this round's preserved testing evidence."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'output/q4_speedup'
ARCHIVE=OUT/'第四问_第二轮提速_代码与数据.zip'

def main():
 files=set()
 code_dirs=['code/experiments/q4_speedup','code/experiments/q4_comparison',
            'code/experiments/q4_official_practice','code/src/simulator_client','code/tests']
 for directory in code_dirs:
  for p in (ROOT/directory).rglob('*.py'):
   if '__pycache__' not in p.parts:files.add(p)
 for p in (ROOT/'code/experiments/q4_speedup').glob('*'):
  if p.is_file() and p.suffix in ('.md','.json','.txt'):files.add(p)
 for p in OUT.rglob('*'):
  if (p.is_file() and p.suffix not in ('.zip','.pyc') and '__pycache__' not in p.parts
      and p.name not in ('package_manifest.json','delivery_manifest.json')):
   files.add(p)
 manifest={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
 report_names=['第四问_第二轮提速实测报告.md','算法构造与保证.md','worst_case_audit.md','worst_case_audit.json','最终统计.json']
 with zipfile.ZipFile(ARCHIVE,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(files):z.write(p,p.relative_to(ROOT).as_posix())
  for name in report_names:
   p=OUT/name
   if p.exists():z.write(p,name)
  z.writestr('package_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
  z.writestr('README.md','''# 第四问第二轮提速
先阅读根目录的“第四问_第二轮提速实测报告.md”。源代码位于code，逐场数据位于output/q4_speedup。
保留V4动作的演练入口：code/experiments/q4_official_practice/run_speedup.py，默认v4_fast v2。
本包不含Python运行时；本机与Windows验证使用Python 3.13。
离线验证：python3.13 code/experiments/q4_speedup/test_speedup_adapter.py
复建报告：python3.13 code/experiments/q4_speedup/make_speedup_report.py
本轮新官方演练尚未进行：Mac锁屏等待手动解锁。需界面确认第四问演练后才能使用practice-confirmed标志。
旧15场官方数据保留在上一轮包里，不属于本版结果。所有新候选的数据均注明开发/合成验证身份。
其他实验策略不代表推荐采用；不能将CPU计算加速视为机器人虚拟任务省时。
package_manifest.json列出保留源文件和证据的SHA-256。
''')
 with zipfile.ZipFile(ARCHIVE) as z:
  assert z.testzip() is None
  for name,sha in manifest.items():
   assert hashlib.sha256(z.read(name)).hexdigest()==sha
 digest=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
 record={'archive':ARCHIVE.name,'bytes':ARCHIVE.stat().st_size,'sha256':digest,'hashed_files':len(manifest)}
 (OUT/'package_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
 print(json.dumps(record,ensure_ascii=False))
if __name__=='__main__':main()
