param(
    [string]$Video = '',
    [string]$Output = '',
    [ValidateSet('fp16','fp32')][string]$FeaturePrecision = 'fp16',
    [ValidateSet('raw','temporal')][string]$KeypointIdentity = 'raw',
    [switch]$StaticCamera
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$pythonPath = Join-Path $projectRoot '.venv-gvhmr\Scripts\python.exe'
if (-not $Video) { $Video = Join-Path $projectRoot 'research\global-motion-probe\dance_12_18s.mp4' }
if (-not $Output) { $Output = Join-Path $projectRoot 'research\global-motion-probe\gvhmr-windows' }
$Video = (Resolve-Path -LiteralPath $Video).Path
$Output = [IO.Path]::GetFullPath($Output)
New-Item -ItemType Directory -Path $Output -Force | Out-Null
$common = @('-X','utf8',(Join-Path $PSScriptRoot 'run_gvhmr_probe.py'),'--video',$Video,'--output',$Output,'--feature-precision',$FeaturePrecision)
$common += @('--keypoint-identity',$KeypointIdentity)
if ($StaticCamera) { $common += '--static-camera' }
$stages = @('preflight','track','pose','features','camera','infer')
$files = @{track='bbx.pt'; pose='vitpose.pt'; features='vit_features.pt'; camera='slam_results.pt'; infer='motion_world.npz'}
foreach ($stage in $stages) {
    $reportPath = Join-Path $Output ($stage + '_telemetry.json')
    # Preflight verifies identical input/settings before any stage can be reused.
    if ($stage -ne 'preflight' -and (Test-Path -LiteralPath $reportPath)) {
        $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
        if ($report.success -and (Test-Path -LiteralPath (Join-Path $Output $files[$stage]))) {
            Write-Host "Reusing completed stage: $stage"
            continue
        }
    }
    Write-Host "Running stage: $stage"
    & $pythonPath @common --stage $stage 2>&1 | Tee-Object -FilePath (Join-Path $Output ($stage + '.log'))
    if ($LASTEXITCODE -ne 0) { throw "GVHMR stage failed: $stage. See log in $Output" }
}
$productionPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$exportDir = Join-Path $Output 'export'
& $productionPython -X utf8 (Join-Path $PSScriptRoot 'convert_gvhmr_motion.py') (Join-Path $Output 'motion_world.npz') --output $exportDir
if ($LASTEXITCODE -ne 0) { throw 'World-motion conversion failed' }
& $productionPython -X utf8 (Join-Path $PSScriptRoot 'review_gvhmr_probe.py') --motion (Join-Path $Output 'motion_world.npz') --video $Video --output (Join-Path $Output 'review')
if ($LASTEXITCODE -ne 0) { throw 'Source overlay review generation failed' }
Write-Host "Research output ready for visual validation: $Output"
