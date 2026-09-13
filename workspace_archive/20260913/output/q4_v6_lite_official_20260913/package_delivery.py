from pathlib import Path
import hashlib,json,shutil,zipfile
ROOT=Path(__file__).resolve().parent
summary=json.loads((ROOT/'summary.json').read_text());assert summary['cases']==11
package=ROOT/'Q4_V6_Lite_官方演练_20260913'
package.mkdir(exist_ok=True)
for name in ['V6_Lite_官方演练11轮报告.md','summary.json','audited_results.json','audit_results.py','build_report.py','bulk_q4.ps1','practice_assignment.json','deployment_manifest.json','windows_platform_status.json','windows_platform_console.txt','evidence_manifest.json','final_idle_status.json','archive_status.json']:
    shutil.copy2(ROOT/name,package/name)
for name in ['official_results','official_logs','additional_10','automation_events']:
    shutil.copytree(ROOT/name,package/name,dirs_exist_ok=True)
proofs=package/'evidence';proofs.mkdir(exist_ok=True)
for src,name in [('1789237370843140000_2f3b9b.png','first_practice_ready.png'),('1789237472615114000_224b77.png','first_completion_dialog.png'),('1789237709923809000_4a4435.png','first_completion_page.png')]:
    shutil.copy2(ROOT/'results'/'desktop'/src,proofs/name)
shutil.copy2(ROOT/'q4_v6_lite_official_bundle.zip',package/'source_bundle.zip')
shutil.copy2(ROOT/'windows_check_extension.zip',package/'windows_check_extension.zip')
paths=[p for p in package.rglob('*') if p.is_file() and p.name!='PACKAGE_SHA256.json']
manifest={str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
(package/'PACKAGE_SHA256.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
archive=ROOT/(package.name+'.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(package.rglob('*')):
        if p.is_file():z.write(p,Path(package.name)/p.relative_to(package))
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
print(json.dumps(dict(package=str(package),archive=str(archive),files=len(manifest),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),ensure_ascii=False,indent=2))
