$root='C:\Users\baiwc\Downloads\Q4V6Lite_20260913'
& C:\Windows\System32\curl.exe --connect-timeout 8 --max-time 15 -I -sS https://cumcm2026b.shumo.net 2>&1 | ForEach-Object {[string]$_} | Out-File "$root\https_probe.txt" -Encoding utf8
@{exit_code=$LASTEXITCODE;utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Out-File "$root\https_probe_status.json" -Encoding utf8
