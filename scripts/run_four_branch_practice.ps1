[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$SimulatorDirectory,
    [ValidateRange(1, 1000)][int]$Repeat = 1,
    [string]$RobotId = '',
    [ValidateRange(1024, 65535)][int]$DebugPort = 19226,
    [ValidateRange(2, 1000000)][int]$MaxActions = 20000,
    [string]$DeliveryRoot = '',
    [string]$RestoredRoot = '',
    [string]$Output = '',
    [string]$PythonPath = '',
    [switch]$PreflightOnly
)
. (Join-Path $PSScriptRoot 'common.ps1')
$workspaceRoot = Split-Path -Parent $script:ProjectRoot
if (-not $PythonPath) {
    $rlPython = Join-Path $workspaceRoot 'q3-deep-rl\.venv-win\Scripts\python.exe'
    if (Test-Path -LiteralPath $rlPython -PathType Leaf) { $PythonPath = $rlPython }
}
$pythonExe = Get-ProjectPython -PythonPath $PythonPath
Set-ProjectPythonEnvironment
if (-not $DeliveryRoot) { $DeliveryRoot = Join-Path $workspaceRoot 'q3-v1-artifacts\v1-delivery' }
if (-not $RestoredRoot) { $RestoredRoot = Join-Path $workspaceRoot 'q3-v1-artifacts\v1-restored-verification' }
$profiles = @('baseline', 'state', 'rl', 'geo')
$commonArgs = @('-m', 'practice_control.branch_worker', '--delivery-root', $DeliveryRoot,
                '--restored-root', $RestoredRoot, '--debug-port', "$DebugPort")
Push-Location -LiteralPath $script:ProjectRoot
try {
    # All source/configuration/model checks finish before the first case starts.
    foreach ($profile in $profiles) {
        & $pythonExe @commonArgs --profile $profile --preflight-only
        if ($LASTEXITCODE -ne 0) { throw "Frozen $profile preflight failed; no batch cases were started." }
    }
    if ($PreflightOnly) { return }
    if (-not $RobotId) { $RobotId = $env:CUMCM_ROBOT_ID }
    if (-not $RobotId) { throw 'Pass -RobotId or set CUMCM_ROBOT_ID; no password is needed.' }
    $simulatorPath = (Resolve-Path -LiteralPath $SimulatorDirectory).ProviderPath
    if (-not (Test-Path -LiteralPath (Join-Path $simulatorPath 'jammers-simulator-full.exe') -PathType Leaf)) {
        throw 'SimulatorDirectory must contain the original installed executable.'
    }
    if (-not $Output) {
        $stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssffffffZ')
        $Output = Join-Path $script:ProjectRoot "results\practice_batches\$stamp-four-frozen-branches"
    }
    $Output = [System.IO.Path]::GetFullPath($Output)
    if (Test-Path -LiteralPath $Output) { throw 'Output directory already exists; select a new batch directory.' }
    New-Item -ItemType Directory -Path $Output | Out-Null
    [pscustomobject]@{
        mode = 'practice'; problem = 3; version_scope = 'frozen_v1_branches'
        profiles = $profiles; repeats_per_profile = $Repeat; python = $pythonExe
        delivery_root = $DeliveryRoot; restored_root = $RestoredRoot
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $Output 'batch-manifest.json') -Encoding UTF8
    for ($round = 1; $round -le $Repeat; $round++) {
        foreach ($profile in $profiles) {
            $runOutput = Join-Path $Output ('round-{0:D3}-{1}' -f $round, $profile)
            & $pythonExe @commonArgs --profile $profile --robot-id $RobotId --simulator-dir $simulatorPath --max-actions $MaxActions --output $runOutput
            if ($LASTEXITCODE -ne 0) {
                throw "Frozen $profile practice round $round failed; remaining cases were not started. Evidence: $runOutput"
            }
        }
    }
    Write-Output "Frozen four-branch Q3 practice batch completed: $Output"
} finally {
    Pop-Location
}
