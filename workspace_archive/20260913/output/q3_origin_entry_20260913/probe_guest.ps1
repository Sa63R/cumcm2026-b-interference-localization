$ErrorActionPreference='Stop'
$q3Root='C:\Users\baiwc\Downloads\Q3Practice'
$q4Root='C:\Users\baiwc\Downloads\Q4V6Lite_20260913'
@{
 utc=(Get-Date).ToUniversalTime().ToString('o')
 python_processes=@(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python'} | Select-Object Name,ProcessId)
 automation_tasks_running=@(Get-ScheduledTask -ErrorAction SilentlyContinue | Where-Object {$_.TaskName -like 'Codex-Q*' -and $_.State -eq 'Running'} | Select-Object TaskName,State)
 q3_root_exists=(Test-Path $q3Root)
 q4_root_exists=(Test-Path $q4Root)
 q3_new_entry_exists=(Test-Path "$q4Root\run_q3_origin.py")
 numpy_exists=(Test-Path "$q3Root\lib\numpy\__init__.py")
 numpy_lib_items=@(Get-ChildItem "$q3Root\lib" -ErrorAction SilentlyContinue | Select-Object Name)
 numpy_metadata=@(Get-ChildItem "$q3Root\lib" -Filter 'numpy*.dist-info' -ErrorAction SilentlyContinue | Select-Object Name)
 q4_runner_sha=(Get-FileHash "$q4Root\run_official.py" -Algorithm SHA256).Hash.ToLower()
} | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 'C:\Users\baiwc\Downloads\q3_origin_entry_probe.json'
