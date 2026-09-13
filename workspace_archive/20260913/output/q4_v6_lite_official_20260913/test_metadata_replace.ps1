$ErrorActionPreference='Stop'
$root='C:\Users\baiwc\Downloads\Q4V6Lite_20260913'
try {
[IO.File]::WriteAllText("$root\metadata_probe_source",'new')
[IO.File]::WriteAllText("$root\metadata_probe_dest",'old')
[IO.File]::Replace("$root\metadata_probe_source","$root\metadata_probe_dest",[NullString]::Value)
if([IO.File]::ReadAllText("$root\metadata_probe_dest") -ne 'new'){throw 'Replacement mismatch'}
@{status='passed'} | ConvertTo-Json | Out-File "$root\metadata_replace_check.json" -Encoding utf8
} catch {@{status='failed';error=$_.Exception.Message} | ConvertTo-Json | Out-File "$root\metadata_replace_check.json" -Encoding utf8}
