[CmdletBinding()]
param(
    [string]$PythonPath = '',
    [switch]$WithTests,
    [switch]$WithReports
)
. (Join-Path $PSScriptRoot 'common.ps1')
$pythonExe = Get-ProjectPython -PythonPath $PythonPath
$venvPython = Join-Path $script:ProjectRoot '.venv-win\Scripts\python.exe'
Push-Location -LiteralPath $script:ProjectRoot
try {
    if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        & $pythonExe -m venv (Join-Path $script:ProjectRoot '.venv-win')
        if ($LASTEXITCODE -ne 0) { throw 'Creating the Windows virtual environment failed.' }
    }
    Set-ProjectPythonEnvironment
    if ($WithTests) {
        & $venvPython -m pip install -r requirements-test.txt
        if ($LASTEXITCODE -ne 0) { throw 'Installing test dependencies failed.' }
    }
    if ($WithReports) {
        & $venvPython -m pip install 'matplotlib>=3.6' 'reportlab>=4' 'pypdf>=5'
        if ($LASTEXITCODE -ne 0) { throw 'Installing report dependencies failed.' }
    }
    & $venvPython -c 'import simulator_client, strategies, workflow'
    if ($LASTEXITCODE -ne 0) { throw 'Source import check failed.' }
    Write-Host 'Windows Python and source imports are ready.'
    if ($WithTests) {
        & $venvPython -m pytest -q
        if ($LASTEXITCODE -ne 0) { throw 'Project tests failed.' }
    }
    Write-Host "Python: $venvPython"
    Write-Host 'Run from an already-started GUI session: .\scripts\run_session.ps1 -Problem 3 -Mode practice'
} finally { Pop-Location }
