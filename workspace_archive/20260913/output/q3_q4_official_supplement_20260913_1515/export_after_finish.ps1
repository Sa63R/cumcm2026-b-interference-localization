$ErrorActionPreference='Stop'
$root='C:\Users\baiwc\Downloads\Q4V6Lite_20260913'
$batch=Join-Path $root 'practice_supplement_20260913_1515'
try {
 while([DateTimeOffset]::UtcNow -lt [DateTimeOffset]::Parse('2026-09-13T07:33:00Z')){
  $s=Get-Content -Raw -Encoding UTF8 (Join-Path $batch 'status.json') | ConvertFrom-Json
  if($s.status -eq 'completed'){
   $rows=@(Get-Content -Raw -Encoding UTF8 (Join-Path $batch 'completed.json') | ConvertFrom-Json)
   & 'C:\Users\baiwc\Downloads\Q4Practice\python\python.exe' (Join-Path $root 'archive_supplement_guest.py') --limit $rows.Count
   if($LASTEXITCODE -ne 0){throw 'Archive command failed'}
   @{status='archive_completed';count=$rows.Count;utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $root 'supplement_export_watcher.json')
   exit
  }
  if($s.status -in @('error','stopped')){throw ('Batch stopped: '+$s.message)}
  Start-Sleep -Seconds 2
 }
 throw 'Wait deadline exceeded; preserved all data'
} catch { @{status='error';message=$_.Exception.ToString();utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $root 'supplement_export_watcher.json') }
