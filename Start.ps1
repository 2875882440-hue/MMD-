param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$runFile = Join-Path $projectRoot 'run.py'
$logRoot = Join-Path $projectRoot 'logs'
$appUrl = 'http://127.0.0.1:8766'
$expectedInstance = 'motionbridge-physics-dev'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Python environment missing. Run Setup.ps1 in THIS development copy.' }
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$existingHealth = $null
try { $existingHealth = Invoke-RestMethod -Uri "$appUrl/api/health" -TimeoutSec 2 } catch {}
if ($existingHealth) {
    if ($existingHealth.instance_id -ne $expectedInstance -or $existingHealth.project_root -ne $projectRoot) {
        throw 'Port 8766 belongs to another project. No process was stopped or reused.'
    }
    if (-not $NoBrowser) { Start-Process $appUrl }
    exit 0
}
$worker = Start-Process -FilePath $pythonExe -ArgumentList @(('"' + $runFile + '"'), '--no-browser', '--port', '8766') -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logRoot 'physics-server.stdout.log') -RedirectStandardError (Join-Path $logRoot 'physics-server.stderr.log')
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    Start-Sleep -Milliseconds 500
    $health = $null
    try { $health = Invoke-RestMethod -Uri "$appUrl/api/health" -TimeoutSec 1 } catch {}
    if ($health -and $health.instance_id -eq $expectedInstance -and $health.project_root -eq $projectRoot) {
        @{ pid=$health.pid; launcher_pid=$worker.Id; project=$projectRoot; instance_id=$expectedInstance; port=8766; started_at=(Get-Date).ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logRoot 'physics-server-process.json') -Encoding utf8
        if (-not $NoBrowser) { Start-Process $appUrl }
        exit 0
    }
    if ($worker.HasExited) { throw 'Development server stopped. See logs/physics-server.stderr.log.' }
}
throw 'Development startup timed out; see physics-server logs.'
