[CmdletBinding()]
param(
    [ValidateSet(3, 4)][int]$Problem = 3,
    [ValidateSet('practice', 'formal')][string]$Mode = 'practice',
    [string]$RobotId = '',
    [string]$CaseCode = '',
    [ValidateSet('auto', 'baseline', 'adaptive', 'deferred', 'triangular', 'efficient', 'rollout')][string]$Variant = 'auto',
    [string]$RolloutConfig = '',
    [ValidateSet('center', 'minimax')][string]$ActivePolicy = 'center',
    [string]$BaseUrl = 'http://127.0.0.1:2026',
    [string]$PythonPath = '',
    [switch]$CheckOnly
)
. (Join-Path $PSScriptRoot 'common.ps1')
$pythonExe = Get-ProjectPython -PythonPath $PythonPath
Set-ProjectPythonEnvironment
Push-Location -LiteralPath $script:ProjectRoot
try {
    & $pythonExe -m simulator_client check --base-url $BaseUrl
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    if ($CheckOnly) { exit 0 }
    Write-Host "Declared GUI session: Problem $Problem / $Mode. HTTP cannot verify the GUI selection."
    if (-not $RobotId) { $RobotId = $env:CUMCM_ROBOT_ID }
    if (-not $RobotId) { $RobotId = Read-Host 'Team ID (the current GUI login team; no password)' }
    if (-not $RobotId.Trim()) { throw 'A team ID is required.' }
    $runArgs = @('-m', 'workflow', 'run', '--problem', "$Problem", '--mode', $Mode,
                 '--robot-id', $RobotId, '--base-url', $BaseUrl, '--active-policy', $ActivePolicy)
    if ($Variant -ne 'auto') { $runArgs += @('--variant', $Variant) }
    if ($RolloutConfig) { $runArgs += @('--rollout-config', $RolloutConfig) }
    if ($CaseCode) { $runArgs += @('--case-code', $CaseCode) }
    & $pythonExe @runArgs
    $runExit = $LASTEXITCODE
    if ($runExit -eq 0) {
        Write-Host 'Model-certified search completed. Check the official GUI and export its original encrypted log.'
    } else {
        Write-Host 'Search did not complete cleanly. Preserve the printed summary and inspect the GUI before any new test.'
    }
    exit $runExit
} finally { Pop-Location }
