$ErrorActionPreference = 'Stop'
$root = 'C:\Users\baiwc\Downloads\Q4V6Lite_20260913'
$archive = 'C:\Users\baiwc\Downloads\q4_v6_lite_official_bundle.zip'
$python = 'C:\Users\baiwc\Downloads\Q4Practice\python\python.exe'
$status = 'C:\Users\baiwc\Downloads\q4_v6_lite_setup_status.json'
try {
 if ((Get-FileHash $archive -Algorithm SHA256).Hash -ne '9295ed21c0e549f56bb1b48f76705bf9f3af07e13427b15586365c26252195ba') { throw 'Bundle hash mismatch' }
 if (Test-Path $root) { throw 'Existing Q4V6Lite directory preserved' }
 Expand-Archive -LiteralPath $archive -DestinationPath $root
 & $python "$root\test_bridge.py" --output "$root\windows_http_checks" *> "$root\windows_http_console.txt"
 $code = $LASTEXITCODE
 @{status='bridge_checks_finished';exit_code=$code;python=$python;utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Out-File $status -Encoding utf8
} catch {
 @{status='setup_failed';error=$_.Exception.Message} | ConvertTo-Json | Out-File $status -Encoding utf8
 exit 1
}
