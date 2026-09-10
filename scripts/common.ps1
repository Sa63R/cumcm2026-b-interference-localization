# Shared local Python discovery; the historical Linux .venv is untouched.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$script:ProjectRoot = Split-Path -Parent $PSScriptRoot

function Get-ProjectPython {
    param([string]$PythonPath = '')
    $candidates = @()
    if ($PythonPath) { $candidates += $PythonPath }
    $candidates += (Join-Path $script:ProjectRoot '.venv-win\Scripts\python.exe')
    $candidates += (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe')
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand) { $candidates += $pythonCommand.Source }
    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            & $candidate -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>$null
            if ($LASTEXITCODE -eq 0) { return $candidate }
        }
    }
    throw 'Python 3.10+ was not found. Install Python or pass -PythonPath to setup.ps1.'
}

function Set-ProjectPythonEnvironment {
    $env:PYTHONUTF8 = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    $sourcePath = Join-Path $script:ProjectRoot 'src'
    $env:PYTHONPATH = if ($env:PYTHONPATH) { "$sourcePath;$env:PYTHONPATH" } else { $sourcePath }
}
