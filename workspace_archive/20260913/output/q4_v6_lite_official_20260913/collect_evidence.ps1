$ErrorActionPreference='Stop'
$root='C:\Users\baiwc\Downloads\Q4V6Lite_20260913'
$logs=@(Get-ChildItem 'C:\Users\baiwc\Downloads\Q4Practice\Jammers-simulator' -Recurse -Filter '*EHKR-PC7N-PNWY-K6A9*.jlog' | ForEach-Object {@{path=$_.FullName;bytes=$_.Length;sha256=(Get-FileHash $_.FullName -Algorithm SHA256).Hash;last_write_utc=$_.LastWriteTimeUtc.ToString('o')}})
@{utc=(Get-Date).ToUniversalTime().ToString('o');official_logs=$logs;python_running=@(Get-Process -Name python -ErrorAction SilentlyContinue | Select-Object Id,ProcessName);windows_checks=[string](Get-Content -Raw "$root\windows_platform_checks\summary.json")} | ConvertTo-Json -Depth 6 | Out-File "$root\evidence_manifest.json" -Encoding utf8
