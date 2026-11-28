param([switch]$SkipDemos)
# Regenerate 16-dim BC demos into p3_bc_dataset_v5, then retrain BC checkpoint.
# Run from any directory; paths are absolute.
$ErrorActionPreference = 'Stop'
$root   = 'C:\Users\dash\Downloads\NewProject\safety-v2x-pipeline\SafetyOffloading\workflow_v2'
$python = 'C:\Users\dash\AppData\Local\Programs\Python\Python311\python.exe'
$matlab = 'C:\Program Files\MATLAB\R2026a\bin\matlab.exe'
$outDir = Join-Path $root 'outputs\p3_bc_dataset_v5'
$cfg    = Join-Path $root 'config\p3_bc_selected.json'

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

if (-not $SkipDemos) {
    $seeds = 1101, 1102, 1103, 1104, 1105
    foreach ($seed in $seeds) {
        $seedDir = Join-Path $outDir "seed$seed"
        if (Test-Path (Join-Path $seedDir 'COMPLETE.mat')) {
            Write-Output "seed $seed already complete, skipping"
            continue
        }
        Write-Output "Collecting 16-dim demos: seed $seed ..."
        $rootFwd   = $root.Replace('\','/')
        $seedFwd   = $seedDir.Replace('\','/')
        $cfgFwd    = $cfg.Replace('\','/')
        $log = Join-Path $outDir "seed${seed}_matlab.log"
        # Use call operator (not Start-Process) so semicolons in MATLAB code are preserved.
        & $matlab -batch "addpath(genpath('$rootFwd')); runR3PilotCase('$seedFwd','workflow_drop_low_congest',$seed,'config','$cfgFwd','cpu_offered_load',0.85);" *> $log
        if ($LASTEXITCODE -ne 0) { throw "MATLAB failed for seed $seed (exit $LASTEXITCODE)" }
        Write-Output "seed $seed done"
    }
    Write-Output "All demo seeds complete."
}

# Verify demos exist and check dimension
$demoFiles = Get-ChildItem $outDir -Recurse -Filter 'fresh_demos.json' -ErrorAction SilentlyContinue
if ($demoFiles.Count -eq 0) { throw "No fresh_demos.json found in $outDir" }
Write-Output "Found $($demoFiles.Count) demo file(s). Checking feature dim..."
$sample = Get-Content $demoFiles[0].FullName | ConvertFrom-Json
$dim = $sample[0].candidate_features[0].Count
Write-Output "Feature dim = $dim (expected 16)"
if ($dim -ne 16) { throw "Wrong feature dim $dim; expected 16. Check WorkflowFreshCandidateFeatures.m." }

# Train BC
Write-Output "Training BC checkpoint (p3_bc_dataset_v5)..."
$bcScript = Join-Path $root 'drl\train_fresh_candidate_bc.py'
& $python $bcScript
if ($LASTEXITCODE -ne 0) { throw "BC training failed" }

Write-Output "BC checkpoint written to drl\outputs\fresh_candidate_bc.json"
Write-Output "Run 2 is ready to start: .\run_p4_online80_training.ps1 -RunId 2"
