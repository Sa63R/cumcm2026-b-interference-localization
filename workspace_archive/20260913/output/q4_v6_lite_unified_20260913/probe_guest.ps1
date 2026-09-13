$ErrorActionPreference='Stop'
$root='C:\Users\baiwc\Downloads\Q4V6Lite_20260913'
@{
  utc=(Get-Date).ToUniversalTime().ToString('o')
  python_processes=@(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python'} | Select-Object Name,ProcessId)
  automation_tasks_running=@(Get-ScheduledTask -TaskName 'Codex-Q4V6Lite_20260913-*' -ErrorAction SilentlyContinue | Where-Object {$_.State -eq 'Running'} | Select-Object TaskName,State)
  runner_sha=(Get-FileHash "$root\run_official.py" -Algorithm SHA256).Hash.ToLower()
  manifest_sha=(Get-FileHash "$root\source_manifest.json" -Algorithm SHA256).Hash.ToLower()
} | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 "$root\unified_entry_probe.json"
