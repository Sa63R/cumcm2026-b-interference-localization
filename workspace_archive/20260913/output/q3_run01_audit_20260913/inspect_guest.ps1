$ErrorActionPreference='Stop'
$auditRoot='C:\Users\baiwc\Downloads\Q4V6Lite_20260913'
$auditCandidates=@()
foreach ($auditPath in @("$auditRoot\results","$auditRoot\formal_results","$auditRoot\practice_results",'C:\Users\baiwc\Downloads\Q3Practice\practice_results')) {
 if (Test-Path $auditPath) {
  $auditCandidates+=@(Get-ChildItem $auditPath -Directory -Recurse | Where-Object {$_.Name -like '*q3_run01*'})
 }
}
if (Test-Path "$auditRoot\q3_run01") {$auditCandidates+=Get-Item "$auditRoot\q3_run01"}
$auditCandidates=@($auditCandidates | Sort-Object FullName -Unique)
$auditLogs='C:\Users\baiwc\Downloads\Q4Practice\Jammers-simulator\JammersSimulatorData\behavior-logs'
@{
 utc=(Get-Date).ToUniversalTime().ToString('o')
 python_processes=@(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python'} | Select-Object Name,ProcessId)
 matching_directories=@($auditCandidates | ForEach-Object {
  @{path=$_.FullName;files=@(Get-ChildItem $_.FullName -File | Select-Object Name,Length,LastWriteTimeUtc)}
 })
 recent_result_directories=@(Get-ChildItem "$auditRoot\results" -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 12 Name,FullName,LastWriteTimeUtc)
 recent_official_logs=@(Get-ChildItem $auditLogs -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 15 Name,Length,LastWriteTimeUtc)
} | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 'C:\Users\baiwc\Downloads\q3_run01_inspection.json'
