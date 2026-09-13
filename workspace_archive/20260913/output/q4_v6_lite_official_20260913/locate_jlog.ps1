$ErrorActionPreference='Stop'
$roots=@('C:\Users\baiwc\AppData\Local','C:\Users\baiwc\AppData\Roaming','C:\Users\baiwc\Downloads','C:\Users\baiwc\Documents','C:\Windows\System32')
$logs=@(foreach($dir in $roots){Get-ChildItem -LiteralPath $dir -Recurse -File -Filter '*EHKR-PC7N-PNWY-K6A9*.jlog' -ErrorAction SilentlyContinue | ForEach-Object {@{path=$_.FullName;bytes=$_.Length;sha256=(Get-FileHash $_.FullName -Algorithm SHA256).Hash}}})
@{official_logs=$logs;utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json -Depth 5 | Out-File 'C:\Users\baiwc\Downloads\Q4V6Lite_20260913\official_jlog_manifest.json' -Encoding utf8
