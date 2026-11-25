param(
    [ValidateRange(1,5)][int]$RunId = 1
)
$ErrorActionPreference = 'Stop'
$root = 'C:\Users\dash\Downloads\NewProject\safety-v2x-pipeline\SafetyOffloading\workflow_v2'
$python = 'C:\Users\dash\AppData\Local\Programs\Python\Python311\python.exe'
$matlab = 'C:\Program Files\MATLAB\R2026a\bin\matlab.exe'
$matlabRoot = 'C:/Users/dash/Downloads/NewProject/safety-v2x-pipeline/SafetyOffloading/workflow_v2'
$runName = ('p4_online80_dueling_run{0:D2}' -f $RunId)
$runRoot = Join-Path $root (Join-Path 'outputs' $runName)
$weights = Join-Path $root 'drl\outputs\fresh_candidate_bc.json'
$seedBase = 3000 + 10000 * ($RunId - 1)

function Invoke-Rollout([int]$rollout) {
    $weightMatlab = $weights.Replace('\','/')
    & $matlab -batch "addpath(genpath('$matlabRoot')); runP4Online80Episode($rollout,'$weightMatlab','$runName',$seedBase);"
    if ($LASTEXITCODE -ne 0) { throw "WiLab rollout $rollout failed" }
    & $python (Join-Path $root 'drl\build_online80_ledger.py') '--source' (Join-Path $runRoot 'rollouts') '--output' (Join-Path $runRoot 'rollout_ledger.json')
    if ($LASTEXITCODE -ne 0) { throw "ledger failed after rollout $rollout" }
}
function Invoke-Update([int]$update,[int]$epochs,[string]$phase) {
    $updateRoot = Join-Path $runRoot ("training\update{0:D2}" -f $update)
    New-Item -ItemType Directory -Force -Path $updateRoot | Out-Null
    $replay = Join-Path $updateRoot 'replay.npz'; $policy = Join-Path $updateRoot 'policy.json'
    $report = Join-Path $updateRoot 'optimizer_updates.json'; $checkpoints = Join-Path $updateRoot 'optimizer_checkpoints'
    Copy-Item -LiteralPath $weights -Destination (Join-Path $updateRoot 'input_policy.json') -Force
    & $python (Join-Path $root 'drl\build_fresh_p4_replay.py') '--source' (Join-Path $runRoot 'rollouts') '--output' $replay
    if ($LASTEXITCODE -ne 0) { throw "replay build failed at update $update" }
    & $python (Join-Path $root 'drl\train_fresh_p4_ddqn.py') '--seed' (100000*$RunId+$update) '--replay' $replay '--init' $weights '--output' $policy '--report' $report '--checkpoint-dir' $checkpoints '--checkpoint-every' 1 '--epochs' $epochs
    if ($LASTEXITCODE -ne 0) { throw "Dueling DDQN update $update failed" }
    & $matlab -batch "addpath(genpath('$matlabRoot')); runP4OnlineBlockProbes($update,'$runName','$phase');"
    if ($LASTEXITCODE -ne 0) { throw "fixed evaluation failed at update $update" }
    $weights = $policy
}

New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
@{architecture='dueling_ddqn';run_id=$RunId;physical_seed_base=$seedBase;original_rollouts=80;original_optimizer_updates=1280;extension_rollouts=8;extension_optimizer_updates=128;fixed_probe_seeds=@(1301,1302,1303,1304,1305)} | ConvertTo-Json | Set-Content (Join-Path $runRoot 'run_manifest.json')
for ($rollout = 1; $rollout -le 80; $rollout++) {
    Invoke-Rollout $rollout
    if (($rollout % 5) -eq 0) { Invoke-Update ([int]($rollout / 5)) 80 'main' }
}
# Exact 10% continuation: 8 physical rollouts and 128 optimizer updates.
for ($rollout = 81; $rollout -le 88; $rollout++) {
    Invoke-Rollout $rollout
    if ($rollout -eq 85) { Invoke-Update 17 80 'extension' }
    if ($rollout -eq 88) { Invoke-Update 18 48 'extension' }
}
& $python (Join-Path $root 'drl\evaluate_dueling_convergence.py') '--run-root' $runRoot '--output' (Join-Path $runRoot 'convergence_diagnostics.json')
if ($LASTEXITCODE -ne 0) { throw 'convergence diagnostics failed' }
