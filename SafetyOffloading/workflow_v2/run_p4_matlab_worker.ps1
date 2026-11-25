param([string]$RunName,[string]$WorkerRoot)
$matlab='C:\Program Files\MATLAB\R2026a\bin\matlab.exe'
$matlabRoot='C:/Users/dash/Downloads/NewProject/safety-v2x-pipeline/SafetyOffloading/workflow_v2'
$wr=$WorkerRoot.Replace('\','/')
& $matlab -batch "addpath(genpath('$matlabRoot')); runP4PersistentWorker('$RunName','$wr');"
exit $LASTEXITCODE
