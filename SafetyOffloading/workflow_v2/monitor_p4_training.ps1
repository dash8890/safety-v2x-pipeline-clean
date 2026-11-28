param(
    [int]$RefreshSeconds = 30,
    [switch]$Once
)
$ErrorActionPreference = 'Stop'
$root = 'C:\Users\dash\Downloads\NewProject\safety-v2x-pipeline\SafetyOffloading\workflow_v2'
$totalRuns = 3
$rolloutsPerRun = 88
$updatesPerRun = 1280
$validationsPerRun = 8
$milestoneBlocks = @(4, 8, 12, 16)

function Get-RunState([int]$runId) {
    $name = 'p4_persistent_dueling_run{0:D2}' -f $runId
    $dir = Join-Path $root (Join-Path 'outputs' $name)
    $state = [ordered]@{
        run_id      = $runId
        name        = $name
        exists      = (Test-Path $dir)
        rollouts    = 0
        blocks      = 0
        validations = 0
        updates     = 0
        status      = 'not_started'
        rollout_times = @()
        block_times   = @()
        val_times     = @()
        elapsed_sec   = 0
        last_td_error = $null
        last_return   = $null
    }
    if (-not $state.exists) { return $state }

    $rolloutDir = Join-Path $dir 'rollouts'
    if (Test-Path $rolloutDir) {
        $completeDirs = @(Get-ChildItem $rolloutDir -Directory | Where-Object {
            Test-Path (Join-Path $_.FullName 'COMPLETE.mat')
        })
        $state.rollouts = $completeDirs.Count
    }

    $trainingDir = Join-Path $dir 'training'
    if (Test-Path $trainingDir) {
        $doneBLocks = @(Get-ChildItem $trainingDir -Directory | Where-Object {
            Test-Path (Join-Path $_.FullName 'policy.json')
        })
        $state.blocks = $doneBLocks.Count
    }

    $valDir = Join-Path $dir 'validation'
    if (Test-Path $valDir) {
        $state.validations = @(Get-ChildItem $valDir -Directory | Where-Object {
            Test-Path (Join-Path $_.FullName 'validation_manifest.json')
        }).Count
    }

    $state.updates = [math]::Min($state.blocks * 80, $updatesPerRun)

    $timingsFile = Join-Path $dir 'timings.jsonl'
    if (Test-Path $timingsFile) {
        $lines = Get-Content $timingsFile -ErrorAction SilentlyContinue
        foreach ($line in $lines) {
            if ([string]::IsNullOrWhiteSpace($line)) { continue }
            try {
                $entry = $line | ConvertFrom-Json
                switch ($entry.kind) {
                    'training_rollout' { $state.rollout_times += $entry.seconds }
                    'optimizer_block'  { $state.block_times += $entry.seconds }
                    'block_validation' { $state.val_times += $entry.seconds }
                }
            } catch {}
        }
    }

    $runnerLog = Join-Path $dir 'runner.log'
    if (Test-Path $runnerLog) {
        $tailLines = Get-Content $runnerLog -Tail 5 -ErrorAction SilentlyContinue
        if ($tailLines) {
            for ($i = $tailLines.Count - 1; $i -ge 0; $i--) {
                try {
                    $last = $tailLines[$i] | ConvertFrom-Json
                    if ($last.mean_absolute_td_error) {
                        $state.last_td_error = $last.mean_absolute_td_error
                        $state.last_return = $last.mean_sampled_reward
                        break
                    }
                } catch {}
            }
        }
    }

    $workerDir = Join-Path $dir 'worker'
    $workerReady = Test-Path (Join-Path $workerDir 'READY')
    $workerStop = Test-Path (Join-Path $workerDir 'STOP')

    if ($state.rollouts -ge $rolloutsPerRun -and $state.blocks -ge 18) {
        $state.status = 'complete'
    } elseif ($workerStop) {
        $state.status = 'stopped'
    } elseif ($workerReady) {
        $state.status = 'running'
    } else {
        $state.status = 'starting'
    }

    if ($state.rollout_times.Count -gt 0) {
        $firstLine = Get-Content (Join-Path $dir 'timings.jsonl') -TotalCount 1 -ErrorAction SilentlyContinue
        if ($firstLine) {
            try {
                $utcStr = ($firstLine | ConvertFrom-Json).utc
                $firstUtc = [datetimeoffset]::Parse($utcStr).UtcDateTime
                $state.elapsed_sec = ((Get-Date).ToUniversalTime() - $firstUtc).TotalSeconds
            } catch {}
        }
    }

    return $state
}

function Format-Duration([double]$seconds) {
    if ($seconds -le 0) { return '--:--:--' }
    $ts = [timespan]::FromSeconds([math]::Round($seconds))
    return '{0:D2}:{1:D2}:{2:D2}' -f [int]$ts.TotalHours, $ts.Minutes, $ts.Seconds
}

function Show-Progress {
    $states = @()
    for ($id = 1; $id -le $totalRuns; $id++) { $states += Get-RunState $id }

    Clear-Host
    $now = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    Write-Host "=== P4 Dueling DDQN Training Monitor === $now ===" -ForegroundColor Cyan
    Write-Host ""
    Write-Host ("{0,-6} {1,-10} {2,-12} {3,-12} {4,-12} {5,-10} {6,-10} {7,-12} {8,-12}" -f `
        'Run', 'Status', 'Rollouts', 'Blocks', 'Updates', 'TD Error', 'Return', 'Elapsed', 'ETA')
    Write-Host ("{0,-6} {1,-10} {2,-12} {3,-12} {4,-12} {5,-10} {6,-10} {7,-12} {8,-12}" -f `
        '---', '------', '--------', '------', '-------', '--------', '------', '-------', '---')

    foreach ($s in $states) {
        $rolloutStr = '{0}/88' -f $s.rollouts
        $blockStr = '{0}/18' -f $s.blocks
        $updateStr = '{0}/1280' -f $s.updates
        $tdStr = if ($s.last_td_error) { '{0:F2}' -f $s.last_td_error } else { '--' }
        $retStr = if ($s.last_return) { '{0:F2}' -f $s.last_return } else { '--' }
        $elapsedStr = Format-Duration $s.elapsed_sec

        $etaStr = '--:--:--'
        if ($s.rollouts -gt 0 -and $s.rollouts -lt $rolloutsPerRun -and $s.elapsed_sec -gt 0) {
            $rate = $s.elapsed_sec / $s.rollouts
            $remaining = ($rolloutsPerRun - $s.rollouts) * $rate
            $blockOverhead = ($s.block_times | Measure-Object -Average -ErrorAction SilentlyContinue).Average
            if (-not $blockOverhead) { $blockOverhead = 0 }
            $remainingBlocks = 18 - $s.blocks
            $remaining += $remainingBlocks * $blockOverhead
            $etaStr = Format-Duration $remaining
        }

        $color = switch ($s.status) {
            'running'  { 'Green' }
            'complete' { 'Cyan' }
            'stopped'  { 'Red' }
            default    { 'Gray' }
        }
        Write-Host ("{0,-6} {1,-10} {2,-12} {3,-12} {4,-12} {5,-10} {6,-10} {7,-12} {8,-12}" -f `
            $s.run_id, $s.status, $rolloutStr, $blockStr, $updateStr, $tdStr, $retStr, $elapsedStr, $etaStr) -ForegroundColor $color
    }

    Write-Host ""
    Write-Host "--- Aggregate ---" -ForegroundColor Yellow

    $totalRolloutsDone = ($states | ForEach-Object { $_.rollouts } | Measure-Object -Sum).Sum
    $totalUpdatesDone = ($states | ForEach-Object { $_.updates } | Measure-Object -Sum).Sum
    $totalValsDone = ($states | ForEach-Object { $_.validations } | Measure-Object -Sum).Sum
    $completedRuns = @($states | Where-Object { $_.status -eq 'complete' }).Count
    $runningRuns = @($states | Where-Object { $_.status -eq 'running' }).Count

    $allRolloutTimes = @()
    $allBlockTimes = @()
    foreach ($s in $states) {
        $allRolloutTimes += $s.rollout_times
        $allBlockTimes += $s.block_times
    }
    $meanRollout = if ($allRolloutTimes.Count -gt 0) { ($allRolloutTimes | Measure-Object -Average).Average } else { 0 }
    $meanBlock = if ($allBlockTimes.Count -gt 0) { ($allBlockTimes | Measure-Object -Average).Average } else { 0 }

    $totalPhysical = $totalRolloutsDone + $totalValsDone * 2
    $targetPhysical = $totalRuns * ($rolloutsPerRun + $validationsPerRun)

    Write-Host "  Completed runs:          $completedRuns / $totalRuns"
    Write-Host "  Running workers:         $runningRuns / 3"
    Write-Host "  Training rollouts:       $totalRolloutsDone / $($totalRuns * $rolloutsPerRun)"
    Write-Host "  Optimizer updates:       $totalUpdatesDone / $($totalRuns * $updatesPerRun)"
    Write-Host "  Validation simulations:  $($totalValsDone * 2) / $($totalRuns * $validationsPerRun)"
    Write-Host "  Total physical sims:     $totalPhysical / $targetPhysical"
    Write-Host ("  Mean rollout duration:   {0:F1}s" -f $meanRollout)
    Write-Host ("  Mean block duration:     {0:F1}s" -f $meanBlock)

    if ($meanRollout -gt 0 -and $runningRuns -gt 0) {
        $remainingRollouts = ($totalRuns * $rolloutsPerRun) - $totalRolloutsDone
        $remainingBlocks = ($totalRuns * 18) - ($states | ForEach-Object { $_.blocks } | Measure-Object -Sum).Sum
        $remainingVal = ($totalRuns * 4) - $totalValsDone
        $parallelFactor = [math]::Min($runningRuns, 3)
        $estRemaining = (($remainingRollouts * $meanRollout) + ($remainingBlocks * $meanBlock) + ($remainingVal * $meanRollout * 2)) / $parallelFactor
        Write-Host ("  Est. remaining:          {0}" -f (Format-Duration $estRemaining))
    }

    $matlabProcs = @(Get-Process MATLAB -ErrorAction SilentlyContinue)
    $activeMatlabCpu = ($matlabProcs | Where-Object { $_.CPU -gt 10 }).Count
    Write-Host "  MATLAB processes:        $($matlabProcs.Count) ($activeMatlabCpu active)"
    Write-Host ""
}

do {
    Show-Progress
    if (-not $Once) {
        Write-Host "Refreshing in $RefreshSeconds seconds... (Ctrl+C to stop)" -ForegroundColor DarkGray
        Start-Sleep -Seconds $RefreshSeconds
    }
} while (-not $Once)
