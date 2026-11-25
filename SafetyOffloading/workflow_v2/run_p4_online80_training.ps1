$root = 'C:\Users\dash\Downloads\NewProject\safety-v2x-pipeline\SafetyOffloading\workflow_v2'
$python = 'C:\Users\dash\AppData\Local\Programs\Python\Python311\python.exe'
$matlab = 'C:\Program Files\MATLAB\R2026a\bin\matlab.exe'
$matlabRoot = 'C:/Users/dash/Downloads/NewProject/safety-v2x-pipeline/SafetyOffloading/workflow_v2'
$runRoot = Join-Path $root 'outputs\p4_online80'
$weights = Join-Path $root 'drl\outputs\fresh_candidate_bc.json'

New-Item -ItemType Directory -Force -Path (Join-Path $runRoot 'training') | Out-Null
for ($episode = 1; $episode -le 80; $episode++) {
    $weightMatlab = $weights.Replace('\','/')
    & $matlab -batch "addpath(genpath('$matlabRoot')); runP4Online80Episode($episode,'$weightMatlab');"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $python (Join-Path $root 'drl\build_online80_ledger.py')
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    if (($episode % 5) -ne 0) { continue }
    $update = [int]($episode / 5)
    $updateRoot = Join-Path $runRoot ("training\update{0:D2}" -f $update)
    New-Item -ItemType Directory -Force -Path $updateRoot | Out-Null
    $replay = Join-Path $updateRoot 'replay.npz'
    $policy = Join-Path $updateRoot 'policy.json'
    $report = Join-Path $updateRoot 'epochs.json'
    $checkpoints = Join-Path $updateRoot 'epoch_checkpoints'
    & $python (Join-Path $root 'drl\build_fresh_p4_replay.py') '--source' (Join-Path $runRoot 'episodes') '--output' $replay
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $python (Join-Path $root 'drl\train_fresh_p4_ddqn.py') '--seed' (100 + $update) '--replay' $replay '--init' $weights '--output' $policy '--report' $report '--checkpoint-dir' $checkpoints '--checkpoint-every' 1
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $matlab -batch "addpath(genpath('$matlabRoot')); runP4OnlineBlockProbes($update);"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $weights = $policy
}
