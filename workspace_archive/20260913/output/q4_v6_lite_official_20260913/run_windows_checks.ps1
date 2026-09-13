$ErrorActionPreference='Stop'
$status='C:\Users\baiwc\Downloads\q4_v6_lite_checks_status.json'
try {
 $active=@(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python' -and $_.CommandLine -like '*Q4V6Lite_20260913*'})
 if ($active.Count -gt 0) {throw 'Existing Lite test process still running'}
 & C:\Windows\System32\cmd.exe /c 'C:\Users\baiwc\Downloads\Q4Practice\python\python.exe C:\Users\baiwc\Downloads\Q4V6Lite_20260913\test_bridge.py --output C:\Users\baiwc\Downloads\Q4V6Lite_20260913\windows_http_checks_v2 > C:\Users\baiwc\Downloads\Q4V6Lite_20260913\windows_http_console_v2.txt 2>&1'
 @{status='checks_finished';exit_code=$LASTEXITCODE;utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Out-File $status -Encoding utf8
} catch { @{status='checks_failed';error=$_.Exception.Message} | ConvertTo-Json | Out-File $status -Encoding utf8 }
