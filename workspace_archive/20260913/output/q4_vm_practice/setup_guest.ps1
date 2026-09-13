$ErrorActionPreference = 'Stop'
$root = 'C:\Users\baiwc\Downloads\Q4Practice'
$log = 'C:\Users\baiwc\Downloads\q4_setup_result.txt'
try {
 if (Test-Path $root) { throw 'Q4Practice already exists; refusing to overwrite.' }
 Expand-Archive -LiteralPath 'C:\Users\baiwc\Downloads\q4_practice_bundle.zip' -DestinationPath $root
 $shell = New-Object -ComObject WScript.Shell
 $shortcut = $shell.CreateShortcut('C:\Users\baiwc\Desktop\Q4 Practice Simulator.lnk')
 $shortcut.TargetPath = "$root\Jammers-simulator\jammers-simulator.exe"
 $shortcut.WorkingDirectory = "$root\Jammers-simulator"
 $shortcut.Save()
 & "$root\python\python.exe" -V 2>&1 | Out-File $log -Encoding utf8
 & "$root\python\python.exe" "$root\code\experiments\q4_official_practice\test_adapter.py" 2>&1 | Out-File $log -Append -Encoding utf8
 "TEST_EXIT=$LASTEXITCODE" | Out-File $log -Append -Encoding utf8
 Get-Date -Format o | Out-File $log -Append -Encoding utf8
 Get-TimeZone | Select-Object Id,BaseUtcOffset | Out-File $log -Append -Encoding utf8
 Get-FileHash "$root\Jammers-simulator\jammers-simulator.exe" -Algorithm SHA256 | Format-List | Out-File $log -Append -Encoding utf8
} catch {
 $_ | Out-File $log -Append -Encoding utf8
 exit 1
}
