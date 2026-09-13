$ErrorActionPreference = 'Stop'
$root = 'C:\Users\baiwc\Downloads\Q4Practice'
$r = [ordered]@{}
$r.utc = (Get-Date).ToUniversalTime().ToString('o')
$r.timezone = (Get-TimeZone).Id
$r.simulator_process = @(Get-Process -Name 'jammers-simulator' -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,SessionId)
$r.simulator_sha256 = (Get-FileHash "$root\Jammers-simulator\jammers-simulator.exe" -Algorithm SHA256).Hash
$r.api_listening = @(Get-NetTCPConnection -State Listen -LocalPort 2026 -ErrorAction SilentlyContinue | Select-Object LocalAddress,LocalPort,OwningProcess)
$r | ConvertTo-Json -Depth 4 | Out-File 'C:\Users\baiwc\Downloads\q4_guest_status.json' -Encoding utf8
