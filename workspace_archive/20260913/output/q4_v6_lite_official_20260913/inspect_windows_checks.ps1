$r=[ordered]@{}
$r.utc=(Get-Date).ToUniversalTime().ToString('o')
$r.processes=@(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^(python|cmd|powershell)' } | ForEach-Object { [ordered]@{Name=$_.Name;Pid=$_.ProcessId;Lite=($_.CommandLine -like '*Q4V6Lite*');IsTest=($_.CommandLine -like '*test_bridge.py*')} })
$r.files=@(Get-ChildItem 'C:\Users\baiwc\Downloads\Q4V6Lite_20260913' -Recurse -File | Where-Object {$_.FullName -match 'windows_|status'} | Select-Object FullName,Length,LastWriteTimeUtc)
$r.status_files=@(Get-ChildItem 'C:\Users\baiwc\Downloads\q4_v6_lite*status.json' | ForEach-Object { [ordered]@{Name=$_.Name;Content=(Get-Content $_.FullName -Raw)} })
[IO.File]::WriteAllText('C:\Users\baiwc\Downloads\q4_v6_lite_checks_probe.json',($r | ConvertTo-Json -Depth 7),[Text.Encoding]::UTF8)
