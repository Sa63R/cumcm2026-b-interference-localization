"""Create a <=20 MB research support archive with explicit readiness state."""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from workflow.evidence import audit_findings


def selected_files():
    files=[]
    for directory in ("src","tests","experiments","scripts","results/study","results/validation"):
        for path in (ROOT/directory).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc"}:
                files.append(path)
    for name in ("README.md","pyproject.toml","requirements-test.txt","requirements-report.txt"):
        if (ROOT/name).exists():files.append(ROOT/name)
    files.extend(ROOT.glob("*.cmd"))
    for path in (ROOT/"论文").rglob("*"):
        if path.is_file() and path.suffix in {".md",".png",".json"}:files.append(path)
    for name in ("正式测试登记.csv","正式结果表.csv","正式结果表.md","登记与审计说明.md"):
        path=ROOT/"支撑材料"/name
        if path.exists():files.append(path)
    logs=ROOT/"支撑材料"/"正式日志"
    if logs.exists():files.extend(p for p in logs.rglob("*") if p.is_file())
    return sorted(set(files))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"交付"/"B题支撑材料.zip")
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    paths=selected_files()
    names=[str(p.relative_to(ROOT)).replace("\\","/") for p in paths]
    manifest="\n".join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+name for p,name in zip(paths,names))+"\n"
    _,missing=audit_findings(ROOT/"支撑材料/正式测试登记.csv")
    manuscript=(ROOT/"论文/B题论文.md").read_text(encoding="utf-8")
    formal_ready=not missing and "待官方测试" not in manuscript
    status={"archive_kind":"research_and_execution_support", "created_utc":datetime.now(timezone.utc).isoformat(),
            "formal_submission_ready":formal_ready, "official_missing":missing,
            "note":"Formal evidence audit runs on the original workspace; anonymous archive omits authenticated plaintext. Local research is not official testing. Final paper review and submission receipt remain manual.",
            "plain_session_logs_excluded":"May contain participant identifiers; retained locally for evidence auditing.",
            "files":len(paths)}
    # Raw authenticated request logs remain local. Do not silently send team IDs
    # in an anonymous support package; encrypted official logs retain their bytes.
    with zipfile.ZipFile(args.output,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path,name in zip(paths,names):archive.write(path,name)
        archive.writestr("MANIFEST.sha256",manifest)
        archive.writestr("ARCHIVE_STATUS.json",json.dumps(status,ensure_ascii=False,indent=2))
    size=args.output.stat().st_size
    if size>20_000_000:raise ValueError(f"Archive exceeds 20 MB: {size} bytes")
    with zipfile.ZipFile(args.output) as archive:
        bad=archive.testzip()
        if bad:raise ValueError("Archive CRC failed: "+bad)
        for path,name in zip(paths,names):
            if hashlib.sha256(archive.read(name)).hexdigest()!=hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError("Archive content mismatch: "+name)
    report={"file":args.output.name,"size_bytes":size,"file_count":len(paths)+2,
            "sha256":hashlib.sha256(args.output.read_bytes()).hexdigest(),"crc_ok":True,"all_contents_verified":True,
            "formal_submission_ready":formal_ready}
    (args.output.parent/"支撑包校验.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False))


if __name__=="__main__":main()
