[CmdletBinding()]
param(
    [ValidateSet('register', 'confirm-upload', 'audit', 'export-tables')][string]$Action = 'audit',
    [ValidateSet(3, 4)][int]$Problem = 3,
    [ValidateSet(1, 2, 3)][int]$Slot = 1,
    [string]$CaseCode = '',
    [string]$Summary = '',
    [string]$OfficialLog = '',
    [Nullable[double]]$Runtime = $null,
    [Nullable[int]]$ClearedCount = $null,
    [Nullable[double]]$VirtualTime = $null,
    [switch]$Uploaded,
    [string]$Ledger = '',
    [string]$PythonPath = ''
)
. (Join-Path $PSScriptRoot 'common.ps1')
$pythonExe = Get-ProjectPython -PythonPath $PythonPath
Set-ProjectPythonEnvironment
Push-Location -LiteralPath $script:ProjectRoot
try {
    $arguments = @('-m', 'workflow', $Action)
    if ($Ledger) { $arguments += @('--ledger', $Ledger) }
    if ($Action -in @('register', 'confirm-upload')) {
        if (-not $CaseCode) { throw '-CaseCode is required.' }
        $arguments += @('--problem', "$Problem", '--slot', "$Slot", '--case-code', $CaseCode)
    }
    if ($Action -eq 'register') {
        if (-not $Summary -or -not $OfficialLog -or $null -eq $Runtime) {
            throw '-Summary, -OfficialLog and -Runtime are required for registration.'
        }
        $arguments += @('--summary', $Summary, '--official-log', $OfficialLog,
                        '--runtime', $Runtime.ToString([Globalization.CultureInfo]::InvariantCulture))
        if ($null -ne $ClearedCount) { $arguments += @('--cleared-count', "$ClearedCount") }
        if ($null -ne $VirtualTime) {
            $arguments += @('--virtual-time', $VirtualTime.ToString([Globalization.CultureInfo]::InvariantCulture))
        }
        if ($Uploaded) { $arguments += '--uploaded' }
    }
    & $pythonExe @arguments
    exit $LASTEXITCODE
} finally { Pop-Location }
