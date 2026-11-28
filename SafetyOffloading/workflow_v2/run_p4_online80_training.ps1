param([ValidateRange(1,5)][int]$RunId = 1)
$ErrorActionPreference='Stop'
$root='C:\Users\dash\Downloads\NewProject\safety-v2x-pipeline\SafetyOffloading\workflow_v2'
$python='C:\Users\dash\AppData\Local\Programs\Python\Python311\python.exe'
$matlab='C:\Program Files\MATLAB\R2026a\bin\matlab.exe'
$matlabRoot='C:/Users/dash/Downloads/NewProject/safety-v2x-pipeline/SafetyOffloading/workflow_v2'
$runName=('p4_corrected_run{0:D2}' -f $RunId)
$runRoot=Join-Path $root (Join-Path 'outputs' $runName)
$weights=Join-Path $root 'drl\outputs\fresh_candidate_bc.json'
$seedBase=3000+10000*($RunId-1)
$workerRoot=Join-Path $runRoot 'worker'
$worker=$null
# Total optimizer updates for the full 16-block main training budget.
# Used by the corrected BC-retention schedule in train_fresh_p4_ddqn.py.
$totalUpdates=2560

function Write-Timing([string]$kind,[int]$id,[double]$seconds) {
 [pscustomobject]@{utc=(Get-Date).ToUniversalTime().ToString('o');kind=$kind;id=$id;seconds=$seconds}|ConvertTo-Json -Compress|Add-Content (Join-Path $runRoot 'timings.jsonl')
}
function Wait-ForMemory([double]$minFreeGB=1.2,[int]$timeoutMin=20) {
 # Block until the OS reports at least minFreeGB of free physical RAM.
 # Prevents simultaneous peak-memory WiLab sims from thrashing the pagefile.
 $deadline=(Get-Date).AddMinutes($timeoutMin)
 while((Get-Date) -lt $deadline){
  $freeGB=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
  if($freeGB -ge $minFreeGB){return}
  $msg=("Run {0}: waiting for memory ({1:F1} GB free, need {2} GB)..." -f $RunId,$freeGB,$minFreeGB)
  Write-Output $msg
  [pscustomobject]@{utc=(Get-Date).ToUniversalTime().ToString('o');kind='memory_wait';free_gb=[math]::Round($freeGB,2)}|ConvertTo-Json -Compress|Add-Content (Join-Path $runRoot 'timings.jsonl')
  Start-Sleep -Seconds 20
 }
 Write-Output ("Run {0}: memory wait timed out after {1} min, proceeding anyway" -f $RunId,$timeoutMin)
}
function Start-Worker {
 New-Item -ItemType Directory -Force -Path $workerRoot|Out-Null
 # Kill previous worker process if PID file exists, preventing orphaned MATLAB processes.
 $pidFile=Join-Path $workerRoot 'worker.pid'
 if(Test-Path $pidFile){
  $oldPid=[int](Get-Content $pidFile -Raw)
  Stop-Process -Id $oldPid -Force -ErrorAction SilentlyContinue
  Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
 }
 # Clean up stale state from any previous session.
 Remove-Item -LiteralPath (Join-Path $workerRoot 'READY') -Force -ErrorAction SilentlyContinue
 Remove-Item -LiteralPath (Join-Path $workerRoot 'STOP') -Force -ErrorAction SilentlyContinue
 Remove-Item -LiteralPath (Join-Path $workerRoot 'job.json') -Force -ErrorAction SilentlyContinue
 Get-ChildItem $workerRoot -Filter 'failed_*.json' -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
 $ready=Join-Path $workerRoot 'READY'
 $wrapper=Join-Path $root 'run_p4_matlab_worker.ps1'
 $worker=Start-Process -FilePath 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',$wrapper,'-RunName',$runName,'-WorkerRoot',$workerRoot) -WindowStyle Hidden -RedirectStandardOutput (Join-Path $workerRoot 'matlab.log') -RedirectStandardError (Join-Path $workerRoot 'matlab.error.log') -PassThru
 $worker.Id | Set-Content $pidFile
 $deadline=(Get-Date).AddMinutes(3); while(!(Test-Path $ready)){if((Get-Date)-gt $deadline){throw 'persistent MATLAB worker did not become ready'};Start-Sleep -Milliseconds 250}
}
function Invoke-Rollout([int]$rollout,[int]$maxRetries=3) {
 $done=Join-Path $workerRoot ('done_{0:D3}.json' -f $rollout)
 $failed=Join-Path $workerRoot ('failed_{0:D3}.json' -f $rollout)
 for($attempt=1;$attempt -le $maxRetries;$attempt++){
  Remove-Item -LiteralPath $done,$failed -Force -ErrorAction SilentlyContinue
  $job=@{rollout_id=$rollout;weights_json=$weights.Replace('\','/');seed_base=$seedBase}|ConvertTo-Json -Compress
  $tmp=Join-Path $workerRoot 'job.tmp'; $jobPath=Join-Path $workerRoot 'job.json'
  Set-Content -LiteralPath $tmp -Value $job -NoNewline; Move-Item -LiteralPath $tmp -Destination $jobPath -Force
  $watch=[System.Diagnostics.Stopwatch]::StartNew()
  $ok=$false
  while($true){
   if(Test-Path $done){$ok=$true;break}
   if(Test-Path $failed){
    $msg=Get-Content $failed -Raw
    Write-Output ("Run {0}: rollout {1} attempt {2}/{3} failed: {4}" -f $RunId,$rollout,$attempt,$maxRetries,$msg)
    [pscustomobject]@{utc=(Get-Date).ToUniversalTime().ToString('o');kind='rollout_retry';rollout=$rollout;attempt=$attempt;message=$msg}|ConvertTo-Json -Compress|Add-Content (Join-Path $runRoot 'timings.jsonl')
    if($worker.HasExited){Start-Worker}
    break
   }
   if($worker.HasExited){
    Write-Output ("Run {0}: MATLAB worker exited during rollout {1} attempt {2}, restarting worker" -f $RunId,$rollout,$attempt)
    Start-Worker; break
   }
   Start-Sleep -Milliseconds 200
  }
  if($ok){
   $watch.Stop();Write-Timing 'training_rollout' $rollout $watch.Elapsed.TotalSeconds
   & $python (Join-Path $root 'drl\build_online80_ledger.py') '--source' (Join-Path $runRoot 'rollouts') '--output' (Join-Path $runRoot 'rollout_ledger.json')
   if($LASTEXITCODE -ne 0){throw "ledger failed after rollout $rollout"}
   return
  }
 }
 throw "WiLab rollout $rollout failed after $maxRetries attempts"
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
 # globalUpdateStart: cumulative updates completed before this block.
 # For the 16 main blocks (each 80 epochs): block B starts at (B-1)*80.
 # Continuation blocks 17 and 18 are after the 1280-update budget; pass 1280
 # so BC retention is zero (fully decayed) during those extra updates.
 $globalUpdateStart = if ($update -le 16) { ($update - 1) * 80 } else { $totalUpdates }
 $watch=[System.Diagnostics.Stopwatch]::StartNew(); & $python (Join-Path $root 'drl\build_fresh_p4_replay.py') '--source' (Join-Path $runRoot 'rollouts') '--output' $replay;if($LASTEXITCODE -ne 0){throw "replay build failed at update $update"}
 & $python (Join-Path $root 'drl\train_fresh_p4_ddqn.py') '--seed' (100000*$RunId+$update) '--replay' $replay '--init' $weights '--output' $policy '--report' $report '--checkpoint-dir' $checkpoints '--checkpoint-every' $epochs '--epochs' $epochs '--global-update-start' $globalUpdateStart '--total-updates' $totalUpdates;if($LASTEXITCODE -ne 0){throw "Dueling DDQN update $update failed"};$watch.Stop();Write-Timing 'optimizer_block' $update $watch.Elapsed.TotalSeconds
 if($validate){Invoke-Validation $update}
 $weights=$policy
}
New-Item -ItemType Directory -Force -Path $runRoot|Out-Null
@{architecture='dueling_ddqn';executor='persistent_matlab_worker';run_id=$RunId;physical_seed_base=$seedBase;training_rollouts=88;optimizer_updates=1280;validation_simulations=8}|ConvertTo-Json|Set-Content (Join-Path $runRoot 'run_manifest.json')

# Startup stagger: offset runs by ~3 rollouts each so they never peak simultaneously.
# Skip stagger on resume (run is already offset from prior progress).
$staggerSec=($RunId-1)*420
$staggerFlag=Join-Path $runRoot 'stagger_done.flag'
if($staggerSec -gt 0 -and !(Test-Path $staggerFlag)){
 Write-Output ("Run {0}: startup stagger {1}s to avoid memory contention with other runs" -f $RunId,$staggerSec)
 Start-Sleep -Seconds $staggerSec
 New-Item -ItemType File -Force -Path $staggerFlag|Out-Null
}

$resumeBlock=0; $resumeRollout=0
for($b=18;$b -ge 1;$b--){
 $pol=Join-Path $runRoot ("training\update{0:D2}\policy.json" -f $b)
 if(Test-Path $pol){$resumeBlock=$b;$weights=$pol;break}
}
if($resumeBlock -gt 0){
 if($resumeBlock -le 16){$resumeRollout=$resumeBlock*5}
 elseif($resumeBlock -eq 17){$resumeRollout=85}
 else{$resumeRollout=88}
}
$rolloutDir=Join-Path $runRoot 'rollouts'
if(Test-Path $rolloutDir){
 for($r=$resumeRollout+1;$r -le 88;$r++){
  if(Test-Path (Join-Path $rolloutDir ("rollout{0:D3}\COMPLETE.mat" -f $r))){$resumeRollout=$r}else{break}
 }
}
if($resumeRollout -gt 0 -or $resumeBlock -gt 0){
 $msg=("RESUME run {0}: block={1} rollout={2} weights={3}" -f $RunId,$resumeBlock,$resumeRollout,$weights)
 Write-Output $msg; [pscustomobject]@{utc=(Get-Date).ToUniversalTime().ToString('o');kind='resume';block=$resumeBlock;rollout=$resumeRollout}|ConvertTo-Json -Compress|Add-Content (Join-Path $runRoot 'timings.jsonl')
}

try {
 Start-Worker
 for($rollout=1;$rollout -le 80;$rollout++){
  if($rollout -le $resumeRollout){
   if(($rollout%5)-eq 0){$block=[int]($rollout/5);if($block -le $resumeBlock){continue};Invoke-Update $block 80 ($block -in @(4,8,12,16))}
   continue
  }
  Invoke-Rollout $rollout;if(($rollout%5)-eq 0){$block=[int]($rollout/5);Invoke-Update $block 80 ($block -in @(4,8,12,16))}
 }
 for($rollout=81;$rollout -le 88;$rollout++){
  if($rollout -le $resumeRollout){
   if($rollout-eq 85 -and $resumeBlock -lt 17){Invoke-Update 17 80 $false}
   if($rollout-eq 88 -and $resumeBlock -lt 18){Invoke-Update 18 48 $false}
   continue
  }
  Invoke-Rollout $rollout;if($rollout-eq 85){Invoke-Update 17 80 $false};if($rollout-eq 88){Invoke-Update 18 48 $false}
 }
 & $python (Join-Path $root 'drl\evaluate_dueling_convergence.py') '--run-root' $runRoot '--output' (Join-Path $runRoot 'convergence_diagnostics.json');if($LASTEXITCODE -ne 0){throw 'convergence diagnostics failed'}
} finally {
 New-Item -ItemType File -Force -Path (Join-Path $workerRoot 'STOP')|Out-Null
 if($worker){$worker.WaitForExit(15000)|Out-Null;if(!$worker.HasExited){Stop-Process -Id $worker.Id -Force}}
}
