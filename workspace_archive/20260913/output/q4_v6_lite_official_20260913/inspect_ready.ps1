$ErrorActionPreference='Stop'
$root='C:\Users\baiwc\Downloads\Q4Practice\Jammers-simulator'
@{utc=(Get-Date).ToUniversalTime().ToString('o');processes=@(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^(python|jammers-simulator|powershell)'} | Select-Object Name,ProcessId,SessionId);recent_files=@(Get-ChildItem $root -Recurse -File | Where-Object {$_.LastWriteTime -gt (Get-Date).AddHours(-1)} | Select-Object FullName,Length,LastWriteTime)} | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 'C:\Users\baiwc\Downloads\Q4V6Lite_20260913\ready_probe.json'
