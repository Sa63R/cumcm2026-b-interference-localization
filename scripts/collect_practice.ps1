[CmdletBinding()]
param(
    [ValidateRange(0, 1000000)][int]$Q3 = 500,
    [ValidateRange(0, 1000000)][int]$Q4 = 500,
    [Parameter(Mandatory = $true)][string]$Output,
    [string]$RobotId = $env:CUMCM_ROBOT_ID,
    [string]$SimulatorDirectory = '',
    [ValidateRange(1024, 65535)][int]$DebugPort = 19226,
    [ValidateRange(2, 2147483647)][int]$MaxActions = 20000,
    [double]$MaxHours = 0,
    [string]$StopFile = '',
    [switch]$Resume,
    [switch]$Background,
    [string]$PythonPath = ''
)
. (Join-Path $PSScriptRoot 'common.ps1')
$pythonExe = Get-ProjectPython -PythonPath $PythonPath
Set-ProjectPythonEnvironment
if (-not $SimulatorDirectory) {
    $SimulatorDirectory = Join-Path (Split-Path -Parent $script:ProjectRoot) '模拟器\数模2026\CUMCM2026B\Jammers-simulator-full'
}
if (-not $RobotId) { throw 'Provide -RobotId or CUMCM_ROBOT_ID (never a password).' }
if ($Q3 + $Q4 -eq 0) { throw 'Request at least one practice episode.' }
if ($MaxHours -lt 0) { throw 'MaxHours must be zero (unlimited) or positive.' }
$datasetPath = [System.IO.Path]::GetFullPath($Output)
$runArguments = @('-u', '-m', 'practice_control.collect', '--q3', "$Q3", '--q4', "$Q4",
                  '--output', $datasetPath, '--robot-id', $RobotId, '--simulator-dir', $SimulatorDirectory,
                  '--debug-port', "$DebugPort", '--max-actions', "$MaxActions")
if ($MaxHours -gt 0) { $runArguments += @('--max-hours', $MaxHours.ToString([System.Globalization.CultureInfo]::InvariantCulture)) }
if ($StopFile) { $runArguments += @('--stop-file', [System.IO.Path]::GetFullPath($StopFile)) }
if ($Resume) { $runArguments += '--resume' }

if ($Background) {
    # Start-Process joins ArgumentList; quote each argument for Windows argv,
    # including trailing backslashes, without evaluating a command string.
    $quotedArguments = foreach ($argument in $runArguments) {
        '"' + [regex]::Replace([regex]::Replace($argument, '(\\*)"', '$1$1\"'), '(\\+)$', '$1$1') + '"'
    }
    $parentDirectory = Split-Path -Parent $datasetPath
    [System.IO.Directory]::CreateDirectory($parentDirectory) | Out-Null
    $launchStamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ')
    $stdoutPath = "$datasetPath.$launchStamp.stdout.log"
    $stderrPath = "$datasetPath.$launchStamp.stderr.log"
    $process = Start-Process -FilePath $pythonExe -ArgumentList ($quotedArguments -join ' ') `
        -WorkingDirectory $script:ProjectRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    [pscustomobject]@{
        pid = $process.Id; dataset = $datasetPath; stdout = $stdoutPath; stderr = $stderrPath
        mode = 'practice'; q3_target = $Q3; q4_target = $Q4
    } | ConvertTo-Json -Compress
    return
}
Push-Location $script:ProjectRoot
try {
    & $pythonExe @runArguments
    if ($LASTEXITCODE -eq 3) { Write-Output 'Practice collection paused cleanly; use -Resume to continue.' }
    elseif ($LASTEXITCODE -ne 0) { throw "Practice collection stopped (exit $LASTEXITCODE); inspect saved progress before resuming." }
} finally {
    Pop-Location
}
