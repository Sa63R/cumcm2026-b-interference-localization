$ErrorActionPreference='Stop'
$root='C:\Users\baiwc\Downloads\Q4V6Lite_20260913'
try {
 & C:\Windows\System32\cmd.exe /c 'C:\Users\baiwc\Downloads\Q4Practice\python\python.exe C:\Users\baiwc\Downloads\Q4V6Lite_20260913\run_official.py --robot-id 202627001104 --case-label EHKR-PC7N-PNWY-K6A9 --practice-confirmed --output C:\Users\baiwc\Downloads\Q4V6Lite_20260913\official_results\EHKR-PC7N-PNWY-K6A9 > C:\Users\baiwc\Downloads\Q4V6Lite_20260913\official_run_console.txt 2>&1'
 @{status='finished';exit_code=$LASTEXITCODE;utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Out-File "$root\official_run_status.json" -Encoding utf8
} catch { @{status='dispatch_error';error=$_.Exception.Message} | ConvertTo-Json | Out-File "$root\official_run_status.json" -Encoding utf8 }
