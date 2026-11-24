$trainerPid = 3416
while (Get-Process -Id $trainerPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 10 }

$matlab = 'C:\Program Files\MATLAB\R2026a\bin\matlab.exe'
$root = 'C:/Users/dash/Downloads/NewProject/safety-v2x-pipeline/SafetyOffloading/workflow_v2'
$out = "$root/outputs/p4_ddqn_multiseed_eval"
$config = "$root/config/p3_bc_selected.json"
$code = "addpath(genpath('$root')); for s=[1101 1102 1103 1104 1105], runR3PilotCase(sprintf('$out/seed%d',s),'workflow_fresh_ddqn',s,'config','$config','cpu_offered_load',0.85); end"
$arg = "-batch `"$code`""
Start-Process -FilePath $matlab -ArgumentList $arg -RedirectStandardOutput 'C:\Users\dash\Downloads\NewProject\safety-v2x-pipeline\SafetyOffloading\workflow_v2\outputs\p4_ddqn_multiseed_eval.stdout.txt' -RedirectStandardError 'C:\Users\dash\Downloads\NewProject\safety-v2x-pipeline\SafetyOffloading\workflow_v2\outputs\p4_ddqn_multiseed_eval.stderr.txt' -WindowStyle Hidden
