$collectorPid = 47752
while (Get-Process -Id $collectorPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 10 }

$root = 'C:\Users\dash\Downloads\NewProject\safety-v2x-pipeline\SafetyOffloading\workflow_v2'
$python = 'C:\Users\dash\AppData\Local\Programs\Python\Python311\python.exe'
& $python (Join-Path $root 'drl\build_fresh_p4_replay.py')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python (Join-Path $root 'drl\train_fresh_p4_ddqn.py') '--checkpoint-dir' (Join-Path $root 'drl\outputs\p4_cycle02_checkpoints') '--checkpoint-every' '20'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$matlab = 'C:\Program Files\MATLAB\R2026a\bin\matlab.exe'
$matlabRoot = 'C:/Users/dash/Downloads/NewProject/safety-v2x-pipeline/SafetyOffloading/workflow_v2'
& $matlab -batch "addpath(genpath('$matlabRoot')); runP4ConvergenceProbes('cycle02');"
exit $LASTEXITCODE
