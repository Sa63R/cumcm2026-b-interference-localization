$ErrorActionPreference='Stop'
$root='C:\Users\baiwc\Downloads\Q3Practice'
$check=Join-Path $root 'checks\ledger_lock_20260912'
New-Item -ItemType Directory -Force $check | Out-Null
trap {@{status='failed';error=$_.Exception.ToString();where=$_.InvocationInfo.PositionMessage} | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $check 'result.json');exit 1}
$tokens=$null;$parseErrors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseInput([IO.File]::ReadAllText((Join-Path $root 'bulk_run.ps1')),[ref]$tokens,[ref]$parseErrors)
if($parseErrors.Count){throw ('Bulk PowerShell parse errors: '+($parseErrors.Message -join '; '))}
$func=$ast.Find({param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Append-Ledger'},$true)
if(-not $func){throw 'Append-Ledger function missing'}
. ([scriptblock]::Create($func.Extent.Text))
$fixture=Join-Path $check 'fixture.jsonl'
$ready=Join-Path $check ('locked_'+[guid]::NewGuid().ToString('N'))
[IO.File]::WriteAllText($fixture,"{`"index`":0}`n",[Text.Encoding]::UTF8)
$job=Start-Job -ArgumentList $fixture,$ready -ScriptBlock {
 param($fixture,$ready)
 $held=[IO.FileStream]::new($fixture,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::Read)
 try{[IO.File]::WriteAllText($ready,'locked');Start-Sleep -Seconds 4}finally{$held.Dispose()}
}
try {
 $deadline=(Get-Date).AddSeconds(15)
 while(-not (Test-Path $ready)){if((Get-Date) -gt $deadline){throw 'Lock fixture did not become ready'};Start-Sleep -Milliseconds 50}
 $watch=[Diagnostics.Stopwatch]::StartNew()
 Append-Ledger @{index=1;text='全清完成'} $fixture
 $watch.Stop()
 if($watch.Elapsed.TotalSeconds -lt 2 -or $watch.Elapsed.TotalSeconds -gt 12){throw 'Sharing lock retry timing unexpected'}
 Append-Ledger @{index=2;text='second'} $fixture
 $rows=@(Get-Content -Encoding UTF8 $fixture | ForEach-Object {$_ | ConvertFrom-Json})
 if($rows.Count -ne 3 -or ($rows.index -join ',') -ne '0,1,2' -or $rows[1].text -ne '全清完成'){throw 'Ledger missing or duplicated rows, or UTF8 corrupted'}
 $failedAsExpected=$false
 try{Append-Ledger @{index=3} (Join-Path $check 'missing_parent\ledger.jsonl')}catch{$failedAsExpected=$true}
 if(-not $failedAsExpected){throw 'Non-sharing IO error did not fail'}
 @{status='passed';parse_errors=0;sharing_retry_elapsed_s=$watch.Elapsed.TotalSeconds;rows=$rows.Count;indices=$rows.index;nonsharing_error_rejected=$failedAsExpected} | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $check 'result.json')
} catch {
 @{status='failed';error=$_.Exception.ToString()} | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $check 'result.json')
 throw
} finally {if($job.State -eq 'Running'){Stop-Job $job};Remove-Job $job}
