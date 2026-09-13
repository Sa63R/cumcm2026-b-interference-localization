$ErrorActionPreference = 'Stop'
$r = [ordered]@{}
$r.utc = (Get-Date).ToUniversalTime().ToString('o')
$r.python_exists = Test-Path 'C:\Users\baiwc\Downloads\Q4Practice\python\python.exe'
$r.program_exists = Test-Path 'C:\Users\baiwc\Downloads\Q4Practice\Jammers-simulator\jammers-simulator.exe'
$r.program_sha256 = (Get-FileHash 'C:\Users\baiwc\Downloads\Q4Practice\Jammers-simulator\jammers-simulator.exe' -Algorithm SHA256).Hash
$r.processes = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(python|pythonw|jammers-simulator)' } | ForEach-Object { [ordered]@{ Name=$_.Name; Pid=$_.ProcessId; Q3=($_.CommandLine -like '*Q3Practice*'); Q4=($_.CommandLine -like '*Q4Practice*') } })
$r.api_listening = @(Get-NetTCPConnection -State Listen -LocalPort 2026 -ErrorAction SilentlyContinue | Select-Object LocalAddress,LocalPort,OwningProcess)
$r | ConvertTo-Json -Depth 5 | Out-File 'C:\Users\baiwc\Downloads\q4_v6_lite_guest_probe.json' -Encoding utf8
