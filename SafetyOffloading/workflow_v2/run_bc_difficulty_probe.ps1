param([int]$SeedBase = 83000)
# BC difficulty probe: run BC policy (fixed) on the same seeds as run09 (83001-83060)
# using the same epsilon/coverage annealing schedule. No optimizer blocks.
# Purpose: determine whether declining rollout TU is scenario difficulty or policy degradation.
$ErrorActionPreference = 'Stop'
$root   = 'C:\Users\dash\Downloads\NewProject\safety-v2x-pipeline\SafetyOffloading\workflow_v2'
$matlab = 'C:\Program Files\MATLAB\R2026a\bin\matlab.exe'
$python = 'C:\Users\dash\AppData\Local\Programs\Python\Python311\python.exe'
$runName    = 'p4_bc_difficulty_probe'
$runRoot    = Join-Path $root (Join-Path 'outputs' $runName)
$bcWeights  = Join-Path $root 'drl\outputs\fresh_candidate_bc.json'
$workerRoot = Join-Path $runRoot 'worker'
$worker     = $null

function Write-Timing([string]$kind, [int]$id, [double]$seconds) {
    [pscustomobject]@{utc=(Get-Date).ToUniversalTime().ToString('o');kind=$kind;id=$id;seconds=$seconds} |
        ConvertTo-Json -Compress | Add-Content (Join-Path $runRoot 'timings.jsonl')
}

function Start-Worker {
    New-Item -ItemType Directory -Force -Path $workerRoot | Out-Null
    $pidFile = Join-Path $workerRoot 'worker.pid'
    if (Test-Path $pidFile) {
        $oldPid = [int](Get-Content $pidFile -Raw)
        Stop-Process -Id $oldPid -Force -ErrorAction SilentlyContinue
        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    }
    Remove-Item -LiteralPath (Join-Path $workerRoot 'READY') -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath (Join-Path $workerRoot 'STOP')  -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath (Join-Path $workerRoot 'job.json') -Force -ErrorAction SilentlyContinue
    $ready   = Join-Path $workerRoot 'READY'
    $wrapper = Join-Path $root 'run_p4_matlab_worker.ps1'
    $script:worker = Start-Process -FilePath 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' `
        -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',$wrapper,'-RunName',$runName,'-WorkerRoot',$workerRoot) `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $workerRoot 'matlab.log') `
        -RedirectStandardError  (Join-Path $workerRoot 'matlab.error.log') `
        -PassThru
    $script:worker.Id | Set-Content $pidFile
    $deadline = (Get-Date).AddMinutes(3)
    while (!(Test-Path $ready)) {
        if ((Get-Date) -gt $deadline) { throw 'MATLAB worker did not become ready' }
        Start-Sleep -Milliseconds 250
    }
}

function Invoke-Rollout([int]$rollout, [int]$maxRetries = 3) {
    $done   = Join-Path $workerRoot ('done_{0:D3}.json'   -f $rollout)
    $failed = Join-Path $workerRoot ('failed_{0:D3}.json' -f $rollout)
    for ($attempt = 1; $attempt -le $maxRetries; $attempt++) {
        Remove-Item -LiteralPath $done, $failed -Force -ErrorAction SilentlyContinue
        $job = @{rollout_id=$rollout; weights_json=$bcWeights.Replace('\','/'); seed_base=$SeedBase} | ConvertTo-Json -Compress
        $tmp     = Join-Path $workerRoot 'job.tmp'
        $jobPath = Join-Path $workerRoot 'job.json'
        Set-Content -LiteralPath $tmp -Value $job -NoNewline
        Move-Item -LiteralPath $tmp -Destination $jobPath -Force
        $watch = [System.Diagnostics.Stopwatch]::StartNew()
        $ok = $false
        while ($true) {
            if (Test-Path $done)   { $ok = $true; break }
            if (Test-Path $failed) {
                $msg = Get-Content $failed -Raw
                Write-Output ("Probe: rollout {0} attempt {1}/{2} failed: {3}" -f $rollout,$attempt,$maxRetries,$msg)
                if ($script:worker.HasExited) { Start-Worker }
                break
            }
            if ($script:worker.HasExited) {
                Write-Output ("Probe: worker exited during rollout {0} attempt {1}, restarting" -f $rollout,$attempt)
                Start-Worker; break
            }
            Start-Sleep -Milliseconds 200
        }
        if ($ok) {
            $watch.Stop()
            Write-Timing 'rollout' $rollout $watch.Elapsed.TotalSeconds
            return
        }
    }
    throw "Probe rollout $rollout failed after $maxRetries attempts"
}

New-Item -ItemType Directory -Force -Path $runRoot | Out-Null

try {
    Start-Worker
    for ($r = 1; $r -le 60; $r++) {
        Invoke-Rollout $r
        $pct = [math]::Round($r / 60 * 100)
        Write-Output ("Probe: rollout {0}/60 done ({1}%)" -f $r,$pct)
    }

    # Collect TU from each rollout's summary.json
    $results = @()
    for ($r = 1; $r -le 60; $r++) {
        $summaryPath = Join-Path $runRoot ('rollouts\rollout{0:D3}\workflow_v2\summary.json' -f $r)
        if (Test-Path $summaryPath) {
            $s  = Get-Content $summaryPath -Raw | ConvertFrom-Json
            $tu = [int]$s.total_value
        } else {
            $tu = $null
            Write-Output ("Probe: WARNING missing summary for rollout {0}" -f $r)
        }
        $results += [pscustomobject]@{rollout=$r; seed=$SeedBase+$r; tu=$tu}
    }

    $outPath = Join-Path $runRoot 'bc_difficulty_results.json'
    $results | ConvertTo-Json | Set-Content $outPath
    Write-Output ("Probe complete. Results: {0}" -f $outPath)

    # Print quick summary
    $valid = $results | Where-Object { $null -ne $_.tu }
    $mean  = ($valid | Measure-Object -Property tu -Average).Average
    Write-Output ("Mean TU across 60 seeds: {0:F1}" -f $mean)
    $q1  = ($valid | Select-Object -First  20 | Measure-Object -Property tu -Average).Average
    $q2  = ($valid | Select-Object -Skip 20 -First 20 | Measure-Object -Property tu -Average).Average
    $q3  = ($valid | Select-Object -Last  20 | Measure-Object -Property tu -Average).Average
    Write-Output ("Rollouts 1-20 avg: {0:F1}  |  21-40 avg: {1:F1}  |  41-60 avg: {2:F1}" -f $q1,$q2,$q3)

} finally {
    New-Item -ItemType File -Force -Path (Join-Path $workerRoot 'STOP') | Out-Null
    if ($script:worker) {
        $script:worker.WaitForExit(15000) | Out-Null
        if (!$script:worker.HasExited) { Stop-Process -Id $script:worker.Id -Force }
    }
}
