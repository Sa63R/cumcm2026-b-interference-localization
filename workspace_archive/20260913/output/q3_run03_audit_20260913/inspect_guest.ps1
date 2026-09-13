$ErrorActionPreference='Stop'
$q3Root='C:\Users\baiwc\Downloads\Q4V6Lite_20260913\results'
$q3Logs='C:\Users\baiwc\Downloads\Q4Practice\Jammers-simulator\JammersSimulatorData\behavior-logs'
@{
 utc=(Get-Date).ToUniversalTime().ToString('o')
 directories=@(Get-ChildItem $q3Root -Directory | Where-Object {$_.Name -match '^q3[_-]run0?3$'} | ForEach-Object {@{path=$_.FullName;files=@(Get-ChildItem $_.FullName -File | Select-Object Name,Length)}})
 logs=@(Get-ChildItem $q3Logs -File | Where-Object {$_.Name -like 'formal-p3-3-*'} | Select-Object Name,Length)
 python_processes=@(Get-Process -Name python -ErrorAction SilentlyContinue | Select-Object Id,ProcessName)
} | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 'C:\Users\baiwc\Downloads\q3_run03_inspection.json'
