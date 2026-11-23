$ErrorActionPreference = 'Stop'
$recordPath = Join-Path $PSScriptRoot 'logs\physics-server-process.json'
if (-not (Test-Path -LiteralPath $recordPath)) { Write-Host 'No managed Physics development server.'; exit 0 }
$record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
if ($record.project -ne $PSScriptRoot -or $record.instance_id -ne 'motionbridge-physics-dev') { throw 'Process record belongs to another project.' }
$expectedRunFile = Join-Path $PSScriptRoot 'run.py'
$ownedIds = @($record.pid,$record.launcher_pid) | Select-Object -Unique
foreach ($ownedId in $ownedIds) {
    $managedProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $ownedId"
    if (-not $managedProcess) { continue }
    if (-not $managedProcess.CommandLine -or -not $managedProcess.CommandLine.Contains($expectedRunFile)) { throw 'PID belongs to another command; refusing to stop it.' }
    Stop-Process -Id $ownedId -ErrorAction SilentlyContinue
}
Remove-Item -LiteralPath $recordPath
Write-Host 'Physics development server stopped; original server untouched.'
