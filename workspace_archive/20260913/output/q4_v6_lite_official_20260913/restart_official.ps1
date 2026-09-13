$ErrorActionPreference='Stop'
$root='C:\Users\baiwc\Downloads\Q4V6Lite_20260913'
try {
 if(Get-Process -Name python -ErrorAction SilentlyContinue){throw 'Python process still active; do not restart simulator'}
 $sim=Get-Process -Name jammers-simulator
 if($sim.Count -ne 1 -or $sim.Path -ne 'C:\Users\baiwc\Downloads\Q4Practice\Jammers-simulator\jammers-simulator.exe'){throw 'Unexpected simulator process'}
 if(-not $sim.CloseMainWindow()){throw 'Graceful close was not accepted'}
 if(-not $sim.WaitForExit(8000)){throw 'Simulator did not close gracefully; no forced termination'}
 Start-Process -FilePath 'C:\Users\baiwc\Downloads\Q4Practice\Jammers-simulator\jammers-simulator.exe' -WorkingDirectory 'C:\Users\baiwc\Downloads\Q4Practice\Jammers-simulator'
 @{status='restarted';utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Out-File "$root\restart_status.json" -Encoding utf8
} catch {@{status='error';message=$_.Exception.Message} | ConvertTo-Json | Out-File "$root\restart_status.json" -Encoding utf8}
