$ErrorActionPreference='Stop'
$q3Stage='C:\Users\baiwc\Downloads\Q3Origin_20260913'
$q3Archive='C:\Users\baiwc\Downloads\Q3Origin_20260913.zip'
$q3Original='C:\Users\baiwc\Downloads\Q3Practice'
$q3Target='C:\Users\baiwc\Downloads\Q4V6Lite_20260913'
$q3Python='C:\Users\baiwc\Downloads\Q4Practice\python\python.exe'
$q3Status='C:\Users\baiwc\Downloads\q3_origin_entry_deployment.json'
$q3Report=[ordered]@{status='preparing';utc=(Get-Date).ToUniversalTime().ToString('o');test_started=$false;simulator_contacted=$false}
function SaveStatus { $q3Report | ConvertTo-Json -Depth 7 | Set-Content -Encoding UTF8 $q3Status }
try {
  SaveStatus
  if (Test-Path $q3Stage) {throw 'Staging folder already exists.'}
  if ((Test-Path "$q3Target\run_q3_origin.py") -or (Test-Path "$q3Target\q3_origin")) {throw 'Q3 destination already exists; refusing to overwrite.'}
  $q3Q4Before=(Get-FileHash "$q3Target\run_official.py" -Algorithm SHA256).Hash.ToLower()
  if ($q3Q4Before -ne '8d11b7d1ffe709947fa97414599a408fdb5dbf27eb7dc5ecc04540ce9c5d35ad') {throw 'Unexpected Q4 entry version.'}
  if ((Get-FileHash $q3Archive -Algorithm SHA256).Hash.ToLower() -ne 'd622c7261033dc881336a91a8c039abc7031f46b5d2134298f0bc164a9a6ca17') {throw 'Archive hash mismatch.'}
  Expand-Archive -LiteralPath $q3Archive -DestinationPath $q3Stage
  $q3Manifest=Get-Content -Raw "$q3Stage\q3_origin\source_manifest.json" | ConvertFrom-Json
  foreach ($q3Name in $q3Manifest.unchanged_source_files) {
    $q3Relative=$q3Name.Substring('q3_origin/'.Length)
    $q3Expected=$q3Manifest.files.PSObject.Properties[$q3Name].Value
    if ((Get-FileHash (Join-Path $q3Original $q3Relative) -Algorithm SHA256).Hash.ToLower() -ne $q3Expected) {throw "Existing Q3 source differs: $q3Relative"}
  }
  $q3Report.unchanged_existing_source_files=$q3Manifest.unchanged_source_files.Count
  New-Item -ItemType Directory -Path "$q3Stage\q3_origin\lib" | Out-Null
  foreach ($q3Name in @('numpy','numpy.libs','numpy-2.5.3.dist-info')) {
    Copy-Item (Join-Path "$q3Original\lib" $q3Name) -Destination "$q3Stage\q3_origin\lib" -Recurse
  }
  $q3Report.status='checking_files_and_imports_only'
  SaveStatus
  $q3Check=Start-Process -FilePath $q3Python -ArgumentList @("$q3Stage\run_q3_origin.py",'--check-only') -WorkingDirectory $q3Stage -RedirectStandardOutput "$q3Stage\preflight.json" -RedirectStandardError "$q3Stage\preflight_stderr.txt" -PassThru -Wait
  if ($q3Check.ExitCode -ne 0) {throw 'Q3 dependency/file preflight failed; no test was started.'}
  $q3Report.preflight=Get-Content -Raw "$q3Stage\preflight.json" | ConvertFrom-Json
  if ((Test-Path "$q3Target\run_q3_origin.py") -or (Test-Path "$q3Target\q3_origin")) {throw 'Q3 destination changed before deployment.'}
  Copy-Item "$q3Stage\q3_origin" -Destination $q3Target -Recurse
  Copy-Item "$q3Stage\run_q3_origin.py","$q3Stage\Q3_原点扫描_使用说明.md","$q3Stage\Q3_Q4_入口说明.md" -Destination $q3Target
  foreach ($q3Item in $q3Manifest.files.PSObject.Properties) {
    if ((Get-FileHash (Join-Path $q3Target $q3Item.Name) -Algorithm SHA256).Hash.ToLower() -ne $q3Item.Value) {throw "Deployed hash mismatch: $($q3Item.Name)"}
  }
  $q3SourceLibrary=Join-Path $q3Stage 'q3_origin\lib'
  $q3LibraryCount=0
  foreach ($q3File in Get-ChildItem $q3SourceLibrary -Recurse -File) {
    $q3Relative=$q3File.FullName.Substring($q3SourceLibrary.Length+1)
    $q3Deployed=Join-Path "$q3Target\q3_origin\lib" $q3Relative
    if ((Get-FileHash $q3File.FullName -Algorithm SHA256).Hash -ne (Get-FileHash $q3Deployed -Algorithm SHA256).Hash) {throw "NumPy copy mismatch: $q3Relative"}
    $q3LibraryCount++
  }
  $q3Report.verified_library_files=$q3LibraryCount
  $q3Report.q4_entry_unchanged=((Get-FileHash "$q3Target\run_official.py" -Algorithm SHA256).Hash.ToLower() -eq $q3Q4Before)
  if (-not $q3Report.q4_entry_unchanged) {throw 'Q4 entry changed during deployment.'}
  $q3Report.entry=Join-Path $q3Target 'run_q3_origin.py'
  $q3Report.entry_sha256=(Get-FileHash $q3Report.entry -Algorithm SHA256).Hash.ToLower()
  $q3Report.status='completed'
} catch {
  $q3Report.status='failed'
  $q3Report.error=$_.Exception.Message
} finally {
  $q3Report.finished_utc=(Get-Date).ToUniversalTime().ToString('o')
  SaveStatus
}
