[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidateSet(3, 4)][int]$Problem,
    [ValidateRange(1, 1000)][int]$Repeat = 1,
    [string]$RobotId = $env:CUMCM_ROBOT_ID,
    [ValidateSet('', 'baseline', 'adaptive', 'deferred', 'efficient', 'rollout', 'triangular')][string]$Variant = '',
    [string]$SimulatorDirectory = '',
    [int]$DebugPort = 19226,
    [string]$Output = '',
    [string]$PythonPath = ''
)
. (Join-Path $PSScriptRoot 'common.ps1')
$pythonExe = Get-ProjectPython -PythonPath $PythonPath
Set-ProjectPythonEnvironment
if (-not $SimulatorDirectory) {
    $SimulatorDirectory = Join-Path (Split-Path -Parent $script:ProjectRoot) '模拟器\数模2026\CUMCM2026B\Jammers-simulator-full'
}
if (-not $RobotId) { throw 'Provide -RobotId or CUMCM_ROBOT_ID (never a password).' }
$runArguments = @('-m', 'practice_control', '--debug-port', "$DebugPort", 'run',
                  '--problem', "$Problem", '--repeat', "$Repeat", '--robot-id', $RobotId,
                  '--simulator-dir', $SimulatorDirectory)
if ($Variant) { $runArguments += @('--variant', $Variant) }
if ($Output) { $runArguments += @('--output', $Output) }
Push-Location $script:ProjectRoot
try {
    & $pythonExe @runArguments
    if ($LASTEXITCODE -ne 0) { throw "Practice batch stopped (exit $LASTEXITCODE); inspect saved evidence before starting another run." }
} finally {
    Pop-Location
}
