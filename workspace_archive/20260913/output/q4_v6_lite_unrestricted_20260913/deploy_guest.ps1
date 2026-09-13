$ErrorActionPreference='Stop'
$q4Target='C:\Users\baiwc\Downloads\Q4V6Lite_20260913'
$q4Stage='C:\Users\baiwc\Downloads\Q4V6Lite_Unrestricted_20260913'
$q4Archive='C:\Users\baiwc\Downloads\Q4V6Lite_Unrestricted_20260913.zip'
$q4Python='C:\Users\baiwc\Downloads\Q4Practice\python\python.exe'
$q4Status="$q4Target\unrestricted_entry_deployment.json"
$q4Report=[ordered]@{status='preparing';utc=(Get-Date).ToUniversalTime().ToString('o');official_test_started=$false}
$q4Backup=$null
$q4Touched=$false
function SaveStatus { $q4Report | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $q4Status }
function AssertIdle {
  if (@(Get-Process -Name python,pythonw -ErrorAction SilentlyContinue).Count -gt 0) {throw 'Python is running; refusing to change live entry.'}
  if (@(Get-ScheduledTask -TaskName 'Codex-Q4V6Lite_20260913-*' -ErrorAction SilentlyContinue | Where-Object {$_.State -eq 'Running'}).Count -gt 0) {throw 'Practice automation is running; refusing to change live entry.'}
}
try {
  SaveStatus
  AssertIdle
  if ((Get-FileHash $q4Archive -Algorithm SHA256).Hash.ToLower() -ne '0edce7a1aec0c99f173689fb2f2a494f5d29bd041f3d4bb0b192889b37906bdd') {throw 'Staging archive hash mismatch.'}
  if (Test-Path $q4Stage) {throw 'Staging directory already exists.'}
  Expand-Archive -LiteralPath $q4Archive -DestinationPath $q4Stage
  $q4Report.status='windows_offline_checks'
  SaveStatus
  $q4Tests=Start-Process -FilePath $q4Python -ArgumentList @("$q4Stage\test_entry_modes.py") -WorkingDirectory $q4Stage -RedirectStandardOutput "$q4Stage\entry_tests_stdout.txt" -RedirectStandardError "$q4Stage\entry_tests_stderr.txt" -PassThru -Wait
  $q4Report.test_exit_code=$q4Tests.ExitCode
  if ($q4Tests.ExitCode -ne 0) {throw 'Windows entry tests failed; live entry was not changed.'}
  AssertIdle
  if ((Get-FileHash "$q4Target\run_official.py" -Algorithm SHA256).Hash.ToLower() -ne 'c0f0f1b5b0181a1e6398aa9735d9d27811150c919776906b0cd06b005692ceb0') {throw 'Live entry changed since inspection.'}
  if ((Get-FileHash "$q4Target\source_manifest.json" -Algorithm SHA256).Hash.ToLower() -ne '7bf4401c986c94dd52611a573a61d859fed669f787d04c83e0140ae7754d54c7') {throw 'Live manifest changed since inspection.'}
  $q4OldManifest=Get-Content -Raw "$q4Target\source_manifest.json" | ConvertFrom-Json
  foreach ($q4Item in $q4OldManifest.files.PSObject.Properties) {
    if ((Get-FileHash (Join-Path $q4Target $q4Item.Name) -Algorithm SHA256).Hash.ToLower() -ne $q4Item.Value) {throw "Live frozen file changed: $($q4Item.Name)"}
  }
  $q4Backup=Join-Path $q4Target ('entry_backup_'+(Get-Date -Format 'yyyyMMdd_HHmmss'))
  New-Item -ItemType Directory -Path $q4Backup -ErrorAction Stop | Out-Null
  Copy-Item "$q4Target\run_official.py","$q4Target\source_manifest.json","$q4Target\test_bridge.py","$q4Target\test_entry_modes.py","$q4Target\RUN_GUIDE.md" -Destination $q4Backup
  $q4Touched=$true
  Copy-Item "$q4Stage\run_official.py","$q4Stage\source_manifest.json","$q4Stage\test_bridge.py" -Destination $q4Target -Force
  Copy-Item "$q4Stage\test_entry_modes.py" -Destination $q4Target
  Copy-Item "$q4Stage\使用说明.md" -Destination "$q4Target\RUN_GUIDE.md"
  $q4Check=Start-Process -FilePath $q4Python -ArgumentList @('run_official.py','--check-only') -WorkingDirectory $q4Target -RedirectStandardOutput "$q4Target\unrestricted_preflight.json" -RedirectStandardError "$q4Target\unrestricted_preflight_stderr.txt" -PassThru -Wait
  if ($q4Check.ExitCode -ne 0) {throw 'Updated live entry preflight failed.'}
  $q4Report.preflight=Get-Content -Raw "$q4Target\unrestricted_preflight.json" | ConvertFrom-Json
  $q4Report.runner_sha256=(Get-FileHash "$q4Target\run_official.py" -Algorithm SHA256).Hash.ToLower()
  $q4Report.backup=$q4Backup
  $q4Report.live_root=$q4Target
  $q4Report.status='completed'
} catch {
  $q4Report.error=$_.Exception.Message
  if ($q4Touched) {
    Copy-Item "$q4Backup\*" -Destination $q4Target -Force
    $q4Report.restored_original=$true
  }
  $q4Report.status='failed'
} finally {
  $q4Report.finished_utc=(Get-Date).ToUniversalTime().ToString('o')
  SaveStatus
}
