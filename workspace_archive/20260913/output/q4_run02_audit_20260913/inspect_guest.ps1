$ErrorActionPreference='Stop'
$q4Root='C:\Users\baiwc\Downloads\Q4V6Lite_20260913'
$q4Logs='C:\Users\baiwc\Downloads\Q4Practice\Jammers-simulator\JammersSimulatorData\behavior-logs'
$q4Dirs=@()
foreach($q4Path in @("$q4Root\results","$q4Root\formal_results")) {
 if(Test-Path $q4Path) {$q4Dirs+=@(Get-ChildItem $q4Path -Directory | Where-Object {$_.Name -match '^(run[-_]q4[-_]0?2|q4[-_]run0?2|run0?2)$'})}
}
@{
 utc=(Get-Date).ToUniversalTime().ToString('o')
 directories=@($q4Dirs | ForEach-Object {@{path=$_.FullName;files=@(Get-ChildItem $_.FullName -File | Select-Object Name,Length)}})
 recent_directories=@(Get-ChildItem "$q4Root\results" -Directory | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 6 Name,FullName)
 logs=@(Get-ChildItem $q4Logs -File | Where-Object {$_.Name -like 'formal-p4-2-*'} | Select-Object Name,Length)
 python_processes=@(Get-Process -Name python -ErrorAction SilentlyContinue | Select-Object Id,ProcessName)
} | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 'C:\Users\baiwc\Downloads\q4_run02_inspection.json'
