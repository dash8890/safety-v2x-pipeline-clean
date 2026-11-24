$trainerPid = 55264
while (Get-Process -Id $trainerPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 10 }
$matlab = 'C:\Program Files\MATLAB\R2026a\bin\matlab.exe'
$code = "addpath(genpath('C:/Users/dash/Downloads/NewProject/safety-v2x-pipeline/SafetyOffloading/workflow_v2')); for s=[1101 1102 1103 1104 1105], runR3PilotCase(sprintf('C:/Users/dash/Downloads/NewProject/safety-v2x-pipeline/SafetyOffloading/workflow_v2/outputs/p4_ddqn_stabilized_eval/seed%d',s),'workflow_fresh_ddqn',s,'config','C:/Users/dash/Downloads/NewProject/safety-v2x-pipeline/SafetyOffloading/workflow_v2/config/p3_bc_selected.json','cpu_offered_load',0.85); end"
$arg = "-batch `"$code`""
Start-Process -FilePath $matlab -ArgumentList $arg -RedirectStandardOutput 'C:\Users\dash\Downloads\NewProject\safety-v2x-pipeline\SafetyOffloading\workflow_v2\outputs\p4_ddqn_stabilized_eval.stdout.txt' -RedirectStandardError 'C:\Users\dash\Downloads\NewProject\safety-v2x-pipeline\SafetyOffloading\workflow_v2\outputs\p4_ddqn_stabilized_eval.stderr.txt' -WindowStyle Hidden
