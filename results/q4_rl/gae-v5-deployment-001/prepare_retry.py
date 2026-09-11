"""Version the prestage-only timestamp comparison; preserve attempt-one evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

OUT = Path(__file__).resolve().parent
raw = (OUT / "prestage.py").read_text()
changes = {
    'import time\n': 'import time\nfrom datetime import datetime\n',
    'CONTROL / "plan.json"': 'CONTROL / "plan-r2.json"',
    'sha(CONTROL / "prestage.py")': 'sha(Path(__file__))',
    'args.deadline != "2026-09-11T22:00:00+00:00"':
        'datetime.fromisoformat(args.deadline).timestamp() != datetime.fromisoformat("2026-09-11T22:00:00+00:00").timestamp()',
    'print("prestage failed: " + type(error).__name__, flush=True)':
        'print("prestage failed: " + type(error).__name__ + (": " + str(error) if isinstance(error, ValueError) else ""), flush=True)',
}
for before, after in changes.items():
    if before not in raw:
        raise ValueError("Unexpected original prestage structure")
    raw = raw.replace(before, after)
path = OUT / "prestage-r2.py"
with path.open("x", encoding="utf-8") as stream:
    stream.write(raw)
plan = json.loads((OUT / "plan.json").read_text())
plan["prestage_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
plan["prestage_revision"] = 2
plan["superseded_prestage_reason"] = "Equivalent UTC/+08:00 deadline strings need timestamp comparison; source and configuration unchanged"
plan["previous_plan_sha256"] = hashlib.sha256((OUT / "plan.json").read_bytes()).hexdigest()
with (OUT / "plan-r2.json").open("x", encoding="utf-8") as stream:
    json.dump(plan, stream, indent=2)
    stream.write("\n")
receipts = []
for name in ("prestage-r2.py", "plan-r2.json"):
    result = subprocess.run([sys.executable, "scripts/q4_object_exchange.py", "--credentials-document", "../AGENTS.md",
        "--task", "q4-rl-gae-v5-20260912", "upload", "--file", str(OUT / name),
        "--name", "launch/train-gae-v5/"+name], check=True, capture_output=True, text=True)
    receipts.append(json.loads(result.stdout))
    print(result.stdout.strip())
with (OUT / "UPLOAD_RETRY_RECEIPTS.json").open("x", encoding="utf-8") as stream:
    json.dump(receipts, stream, indent=2)
    stream.write("\n")
