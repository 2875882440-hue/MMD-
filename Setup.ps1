$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    if (Test-Path -LiteralPath $bundledPython) {
        & $bundledPython -m venv .venv
    } else {
        py -3.12 -m venv .venv
    }
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12, then retry.' }
}
$requirementsFile = 'requirements.txt'
if (Test-Path -LiteralPath 'requirements-lock.txt') { $requirementsFile = 'requirements-lock.txt' }
& $pythonExe -m pip install -r $requirementsFile --prefer-binary --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { throw 'Package installation failed.' }
& $pythonExe scripts/download_models.py
if ($LASTEXITCODE -ne 0) { throw 'Model download failed.' }
& $pythonExe scripts/download_hand_model.py
if ($LASTEXITCODE -ne 0) { Write-Warning 'Optional hand model was not installed. Body capture remains available; retry scripts/download_hand_model.py later.' }
Write-Host 'MotionBridge is ready. Run Start.ps1.'
