$root='C:\Users\baiwc\Downloads\Q4V6Lite_20260913'
$r=Get-Content -Raw "$root\additional_10\runs\03\result.json" | ConvertFrom-Json
$metas=@(Get-ChildItem 'C:\Users\baiwc\Downloads\Q4Practice\Jammers-simulator\JammersSimulatorData\behavior-logs' -Filter 'practice-p4-*.result.json' | Sort-Object LastWriteTime -Descending | Select-Object -First 3 | ForEach-Object {$m=Get-Content -Raw $_.FullName | ConvertFrom-Json;@{file=$_.FullName;metadata=$m;milliseconds=[DateTimeOffset]::Parse($m.ended_at_utc).ToUnixTimeMilliseconds()}})
@{case=$r.case_label;status=$r.status;cleared=$r.client_state.cleared_count;exit=$r.exit_response;metas=$metas} | ConvertTo-Json -Depth 8 | Out-File "$root\completed_03_probe.json" -Encoding utf8
