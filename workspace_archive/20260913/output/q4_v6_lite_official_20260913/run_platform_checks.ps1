$ErrorActionPreference='Stop'
$root='C:\Users\baiwc\Downloads\Q4V6Lite_20260913'
$status='C:\Users\baiwc\Downloads\q4_v6_lite_platform_status.json'
try {
 $zip='C:\Users\baiwc\Downloads\windows_check_extension.zip'
 if ((Get-FileHash $zip -Algorithm SHA256).Hash -ne 'bb32cfba82e2530d03c837165e32fcac3305d73ad7a89697d6729c06e360ce98') {throw 'Extension hash mismatch'}
 if (Test-Path "$root\check_windows.py") {throw 'Existing diagnostic preserved'}
 Expand-Archive -LiteralPath $zip -DestinationPath $root
 & C:\Windows\System32\cmd.exe /c 'C:\Users\baiwc\Downloads\Q4Practice\python\python.exe C:\Users\baiwc\Downloads\Q4V6Lite_20260913\check_windows.py > C:\Users\baiwc\Downloads\Q4V6Lite_20260913\windows_platform_console.txt 2>&1'
 @{status='platform_checks_finished';exit_code=$LASTEXITCODE;utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Out-File $status -Encoding utf8
} catch { @{status='platform_checks_failed';error=$_.Exception.Message} | ConvertTo-Json | Out-File $status -Encoding utf8 }
