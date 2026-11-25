param([ValidateRange(1,5)][int]$RunId = 1)
$ErrorActionPreference='Stop'
$root='C:\Users\dash\Downloads\NewProject\safety-v2x-pipeline\SafetyOffloading\workflow_v2'
$python='C:\Users\dash\AppData\Local\Programs\Python\Python311\python.exe'
$matlab='C:\Program Files\MATLAB\R2026a\bin\matlab.exe'
$matlabRoot='C:/Users/dash/Downloads/NewProject/safety-v2x-pipeline/SafetyOffloading/workflow_v2'
$runName=('p4_persistent_dueling_run{0:D2}' -f $RunId)
$runRoot=Join-Path $root (Join-Path 'outputs' $runName)
$weights=Join-Path $root 'drl\outputs\fresh_candidate_bc.json'
$seedBase=3000+10000*($RunId-1)
$workerRoot=Join-Path $runRoot 'worker'
$worker=$null

function Write-Timing([string]$kind,[int]$id,[double]$seconds) {
 [pscustomobject]@{utc=(Get-Date).ToUniversalTime().ToString('o');kind=$kind;id=$id;seconds=$seconds}|ConvertTo-Json -Compress|Add-Content (Join-Path $runRoot 'timings.jsonl')
}
function Start-Worker {
 New-Item -ItemType Directory -Force -Path $workerRoot|Out-Null
 $ready=Join-Path $workerRoot 'READY'; Remove-Item -LiteralPath $ready -Force -ErrorAction SilentlyContinue
 $wrapper=Join-Path $root 'run_p4_matlab_worker.ps1'
 $worker=Start-Process -FilePath 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',$wrapper,'-RunName',$runName,'-WorkerRoot',$workerRoot) -WindowStyle Hidden -RedirectStandardOutput (Join-Path $workerRoot 'matlab.log') -RedirectStandardError (Join-Path $workerRoot 'matlab.error.log') -PassThru
 $deadline=(Get-Date).AddMinutes(3); while(!(Test-Path $ready)){if((Get-Date)-gt $deadline){throw 'persistent MATLAB worker did not become ready'};Start-Sleep -Milliseconds 250}
}
function Invoke-Rollout([int]$rollout) {
 $watch=[System.Diagnostics.Stopwatch]::StartNew(); $done=Join-Path $workerRoot ('done_{0:D3}.json' -f $rollout); $failed=Join-Path $workerRoot ('failed_{0:D3}.json' -f $rollout)
 Remove-Item -LiteralPath $done,$failed -Force -ErrorAction SilentlyContinue
 $job=@{rollout_id=$rollout;weights_json=$weights.Replace('\','/');seed_base=$seedBase}|ConvertTo-Json -Compress
 $tmp=Join-Path $workerRoot 'job.tmp'; $jobPath=Join-Path $workerRoot 'job.json'; Set-Content -LiteralPath $tmp -Value $job -NoNewline; Move-Item -LiteralPath $tmp -Destination $jobPath -Force
 while(!(Test-Path $done)){if(Test-Path $failed){throw "WiLab rollout $rollout failed: $(Get-Content $failed -Raw)"}; if($worker.HasExited){throw "persistent MATLAB worker exited during rollout $rollout"}; Start-Sleep -Milliseconds 200}
 $watch.Stop();Write-Timing 'training_rollout' $rollout $watch.Elapsed.TotalSeconds
 & $python (Join-Path $root 'drl\build_online80_ledger.py') '--source' (Join-Path $runRoot 'rollouts') '--output' (Join-Path $runRoot 'rollout_ledger.json');if($LASTEXITCODE -ne 0){throw "ledger failed after rollout $rollout"}
}
function Invoke-Validation([int]$block) {
 $watch=[System.Diagnostics.Stopwatch]::StartNew();$tag=('validation_{0:D2}' -f $block);$done=Join-Path $workerRoot ('done_'+$tag+'.json');$failed=Join-Path $workerRoot ('failed_'+$tag+'.json')
 Remove-Item -LiteralPath $done,$failed -Force -ErrorAction SilentlyContinue
 $job=@{kind='validation';block=$block}|ConvertTo-Json -Compress;$tmp=Join-Path $workerRoot 'job.tmp';$jobPath=Join-Path $workerRoot 'job.json';Set-Content -LiteralPath $tmp -Value $job -NoNewline;Move-Item -LiteralPath $tmp -Destination $jobPath -Force
 while(!(Test-Path $done)){if(Test-Path $failed){throw "validation block $block failed: $(Get-Content $failed -Raw)"};if($worker.HasExited){throw "persistent MATLAB worker exited during validation $block"};Start-Sleep -Milliseconds 200}
 $watch.Stop();Write-Timing 'block_validation' $block $watch.Elapsed.TotalSeconds
}
function Invoke-Update([int]$update,[int]$epochs,[bool]$validate) {
 $updateRoot=Join-Path $runRoot ("training\update{0:D2}" -f $update);New-Item -ItemType Directory -Force -Path $updateRoot|Out-Null
 $replay=Join-Path $updateRoot 'replay.npz';$policy=Join-Path $updateRoot 'policy.json';$report=Join-Path $updateRoot 'optimizer_updates.json';$checkpoints=Join-Path $updateRoot 'optimizer_checkpoints'
 Copy-Item -LiteralPath $weights -Destination (Join-Path $updateRoot 'input_policy.json') -Force
 $watch=[System.Diagnostics.Stopwatch]::StartNew(); & $python (Join-Path $root 'drl\build_fresh_p4_replay.py') '--source' (Join-Path $runRoot 'rollouts') '--output' $replay;if($LASTEXITCODE -ne 0){throw "replay build failed at update $update"}
 & $python (Join-Path $root 'drl\train_fresh_p4_ddqn.py') '--seed' (100000*$RunId+$update) '--replay' $replay '--init' $weights '--output' $policy '--report' $report '--checkpoint-dir' $checkpoints '--checkpoint-every' $epochs '--epochs' $epochs;if($LASTEXITCODE -ne 0){throw "Dueling DDQN update $update failed"};$watch.Stop();Write-Timing 'optimizer_block' $update $watch.Elapsed.TotalSeconds
 if($validate){Invoke-Validation $update}
 $weights=$policy
}
New-Item -ItemType Directory -Force -Path $runRoot|Out-Null
@{architecture='dueling_ddqn';executor='persistent_matlab_worker';run_id=$RunId;physical_seed_base=$seedBase;training_rollouts=88;optimizer_updates=1280;validation_simulations=8}|ConvertTo-Json|Set-Content (Join-Path $runRoot 'run_manifest.json')
try {
 Start-Worker
 for($rollout=1;$rollout -le 80;$rollout++){Invoke-Rollout $rollout;if(($rollout%5)-eq 0){$block=[int]($rollout/5);Invoke-Update $block 80 ($block -in @(4,8,12,16))}}
 for($rollout=81;$rollout -le 88;$rollout++){Invoke-Rollout $rollout;if($rollout-eq 85){Invoke-Update 17 80 $false};if($rollout-eq 88){Invoke-Update 18 48 $false}}
 & $python (Join-Path $root 'drl\evaluate_dueling_convergence.py') '--run-root' $runRoot '--output' (Join-Path $runRoot 'convergence_diagnostics.json');if($LASTEXITCODE -ne 0){throw 'convergence diagnostics failed'}
} finally {
 New-Item -ItemType File -Force -Path (Join-Path $workerRoot 'STOP')|Out-Null
 if($worker){$worker.WaitForExit(15000)|Out-Null;if(!$worker.HasExited){Stop-Process -Id $worker.Id -Force}}
}
