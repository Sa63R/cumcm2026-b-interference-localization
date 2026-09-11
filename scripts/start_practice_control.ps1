[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$SimulatorExe,
    [ValidateRange(1024, 65535)][int]$DebugPort = 19226,
    [string]$PythonPath = '',
    [switch]$PrepareOnly
)
. (Join-Path $PSScriptRoot 'common.ps1')

function Assert-SimulatorStopped {
    $simulatorProcesses = @(Get-CimInstance Win32_Process -Filter "Name = 'jammers-simulator-full.exe'")
    if ($simulatorProcesses.Count -gt 0) {
        $processIds = ($simulatorProcesses | ForEach-Object { $_.ProcessId }) -join ', '
        throw "Simulator is already running (PID $processIds). Finish the active work and close it before using this launcher. No process was stopped."
    }
}

function Assert-DebugPortFree {
    $listeners = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
    if (@($listeners | Where-Object { $_.Port -eq $DebugPort }).Count -gt 0) {
        throw "Debug port $DebugPort is already in use."
    }
    $probe = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $DebugPort)
    try {
        $probe.Server.ExclusiveAddressUse = $true
        $probe.Start()
    } finally {
        $probe.Stop()
    }
}

function Ensure-CheckedJunction {
    param([string]$CopyDirectory, [string]$OriginalDirectory, [string]$Name)
    $targetPath = [System.IO.Path]::GetFullPath((Join-Path $OriginalDirectory $Name))
    $linkPath = [System.IO.Path]::GetFullPath((Join-Path $CopyDirectory $Name))
    $targetItem = Get-Item -LiteralPath $targetPath -Force
    if (-not $targetItem.PSIsContainer -or ($targetItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
        throw "Expected the original installed directory without redirection: $targetPath"
    }
    if ((Split-Path -Parent $targetPath) -ine $OriginalDirectory -or
        (Split-Path -Parent $linkPath) -ine $CopyDirectory) {
        throw 'Runtime/data junction paths do not remain in their expected directories.'
    }
    if (Test-Path -LiteralPath $linkPath) {
        $linkItem = Get-Item -LiteralPath $linkPath -Force
        if (-not $linkItem.PSIsContainer -or $linkItem.LinkType -ne 'Junction') {
            throw "Existing runtime/data link is not a junction: $linkPath"
        }
        $linkTargets = @($linkItem.Target)
        if ($linkTargets.Count -ne 1 -or [System.IO.Path]::GetFullPath($linkTargets[0]) -ine $targetPath) {
            throw "Existing junction points to a different directory: $linkPath"
        }
    } else {
        New-Item -ItemType Junction -Path $linkPath -Target $targetPath | Out-Null
    }
}

$resolvedExe = (Resolve-Path -LiteralPath $SimulatorExe).ProviderPath
if ((Split-Path -Leaf $resolvedExe) -ine 'jammers-simulator-full.exe') {
    throw 'Pass the original installed jammers-simulator-full.exe path.'
}
Assert-SimulatorStopped
Assert-DebugPortFree
$pythonExe = Get-ProjectPython -PythonPath $PythonPath
Set-ProjectPythonEnvironment
$preparationOutput = & $pythonExe -m practice_control.runtime --simulator-exe $resolvedExe
if ($LASTEXITCODE -ne 0) { throw 'Audited runtime-copy preparation failed.' }
$manifest = $preparationOutput | ConvertFrom-Json
$originalDirectory = Split-Path -Parent $manifest.original_exe
$copyDirectory = Split-Path -Parent $manifest.copy_exe
Ensure-CheckedJunction -CopyDirectory $copyDirectory -OriginalDirectory $originalDirectory -Name 'WebView2Runtime'
Ensure-CheckedJunction -CopyDirectory $copyDirectory -OriginalDirectory $originalDirectory -Name 'JammersSimulatorData'

if ($PrepareOnly) {
    $manifest | ConvertTo-Json -Depth 4
    return
}

Assert-SimulatorStopped
Assert-DebugPortFree
$savedArguments = [Environment]::GetEnvironmentVariable('WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS', 'Process')
$savedUnusedArguments = [Environment]::GetEnvironmentVariable('JAMMERS_PRACTICE_BROWSER_UNUSED_ARGS_', 'Process')
try {
    [Environment]::SetEnvironmentVariable('JAMMERS_PRACTICE_BROWSER_UNUSED_ARGS_', $null, 'Process')
    [Environment]::SetEnvironmentVariable('WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS',
        "--remote-debugging-address=127.0.0.1 --remote-debugging-port=$DebugPort", 'Process')
    $startedProcess = Start-Process -FilePath $manifest.copy_exe -WorkingDirectory $copyDirectory -WindowStyle Hidden -PassThru
} finally {
    [Environment]::SetEnvironmentVariable('WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS', $savedArguments, 'Process')
    [Environment]::SetEnvironmentVariable('JAMMERS_PRACTICE_BROWSER_UNUSED_ARGS_', $savedUnusedArguments, 'Process')
}
[pscustomobject]@{
    process_id = $startedProcess.Id
    debug_endpoint = "http://127.0.0.1:$DebugPort"
    original_exe = $manifest.original_exe
    copy_exe = $manifest.copy_exe
    original_sha256 = $manifest.original_sha256
    copy_sha256 = $manifest.copy_sha256
    manifest = Join-Path $copyDirectory 'preparation-manifest.json'
} | ConvertTo-Json
