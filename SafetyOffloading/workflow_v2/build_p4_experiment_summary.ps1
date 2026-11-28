param([switch]$Force)
$ErrorActionPreference = 'Stop'
$root = 'C:\Users\dash\Downloads\NewProject\safety-v2x-pipeline\SafetyOffloading\workflow_v2'
$outputsDir = Join-Path $root 'outputs'
$totalRuns = 3
$summaryDir = Join-Path $outputsDir 'p4_experiment_summary'
New-Item -ItemType Directory -Force -Path $summaryDir | Out-Null

$runs = @()
$allRolloutTimes = @()
$allBlockTimes = @()
$allValTimes = @()
$allComplete = $true

for ($id = 1; $id -le $totalRuns; $id++) {
    $name = 'p4_persistent_dueling_run{0:D2}' -f $id
    $dir = Join-Path $outputsDir $name
    $run = [ordered]@{
        run_id = $id
        name = $name
        exists = (Test-Path $dir)
        rollouts_complete = 0
        blocks_complete = 0
        validations_complete = 0
        has_convergence_diagnostics = $false
        seed_base = 3000 + 10000 * ($id - 1)
        total_rollout_seconds = 0
        total_block_seconds = 0
        total_validation_seconds = 0
        total_wall_seconds = 0
        mean_rollout_seconds = 0
        checkpoints = @()
    }

    if (-not $run.exists) { $allComplete = $false; $runs += $run; continue }

    $rolloutDir = Join-Path $dir 'rollouts'
    if (Test-Path $rolloutDir) {
        $run.rollouts_complete = @(Get-ChildItem $rolloutDir -Directory | Where-Object {
            Test-Path (Join-Path $_.FullName 'COMPLETE.mat')
        }).Count
    }

    $trainingDir = Join-Path $dir 'training'
    if (Test-Path $trainingDir) {
        $blocks = @(Get-ChildItem $trainingDir -Directory | Sort-Object Name)
        foreach ($b in $blocks) {
            if (Test-Path (Join-Path $b.FullName 'policy.json')) {
                $run.blocks_complete++
                $run.checkpoints += $b.Name
            }
        }
    }

    $valDir = Join-Path $dir 'validation'
    if (Test-Path $valDir) {
        $run.validations_complete = @(Get-ChildItem $valDir -Directory | Where-Object {
            Test-Path (Join-Path $_.FullName 'validation_manifest.json')
        }).Count
    }

    $run.has_convergence_diagnostics = Test-Path (Join-Path $dir 'convergence_diagnostics.json')

    $timingsFile = Join-Path $dir 'timings.jsonl'
    if (Test-Path $timingsFile) {
        $firstUtc = $null; $lastUtc = $null
        foreach ($line in (Get-Content $timingsFile)) {
            if ([string]::IsNullOrWhiteSpace($line)) { continue }
            try {
                $entry = $line | ConvertFrom-Json
                $ts = [datetime]$entry.utc
                if (-not $firstUtc -or $ts -lt $firstUtc) { $firstUtc = $ts }
                if (-not $lastUtc -or $ts -gt $lastUtc) { $lastUtc = $ts }
                switch ($entry.kind) {
                    'training_rollout' {
                        $run.total_rollout_seconds += $entry.seconds
                        $allRolloutTimes += $entry.seconds
                    }
                    'optimizer_block' {
                        $run.total_block_seconds += $entry.seconds
                        $allBlockTimes += $entry.seconds
                    }
                    'block_validation' {
                        $run.total_validation_seconds += $entry.seconds
                        $allValTimes += $entry.seconds
                    }
                }
            } catch {}
        }
        if ($firstUtc -and $lastUtc) {
            $run.total_wall_seconds = ($lastUtc - $firstUtc).TotalSeconds
        }
        if ($run.rollouts_complete -gt 0) {
            $run.mean_rollout_seconds = [math]::Round($run.total_rollout_seconds / $run.rollouts_complete, 2)
        }
    }

    if ($run.rollouts_complete -lt 88 -or $run.blocks_complete -lt 18) { $allComplete = $false }
    $runs += $run
}

$totalRollouts = ($runs | ForEach-Object { $_.rollouts_complete } | Measure-Object -Sum).Sum
$totalBlocks = ($runs | ForEach-Object { $_.blocks_complete } | Measure-Object -Sum).Sum
$totalVals = ($runs | ForEach-Object { $_.validations_complete } | Measure-Object -Sum).Sum
$completedRuns = @($runs | Where-Object { $_.rollouts_complete -ge 88 -and $_.blocks_complete -ge 18 }).Count

$profiling = [ordered]@{
    total_rollout_seconds = [math]::Round(($allRolloutTimes | Measure-Object -Sum).Sum, 2)
    total_block_seconds = [math]::Round(($allBlockTimes | Measure-Object -Sum).Sum, 2)
    total_validation_seconds = [math]::Round(($allValTimes | Measure-Object -Sum).Sum, 2)
    mean_rollout_seconds = if ($allRolloutTimes.Count -gt 0) { [math]::Round(($allRolloutTimes | Measure-Object -Average).Average, 2) } else { 0 }
    median_rollout_seconds = if ($allRolloutTimes.Count -gt 0) {
        $sorted = $allRolloutTimes | Sort-Object
        $mid = [math]::Floor($sorted.Count / 2)
        if ($sorted.Count % 2 -eq 0) { [math]::Round(($sorted[$mid-1] + $sorted[$mid]) / 2, 2) }
        else { [math]::Round($sorted[$mid], 2) }
    } else { 0 }
    mean_block_seconds = if ($allBlockTimes.Count -gt 0) { [math]::Round(($allBlockTimes | Measure-Object -Average).Average, 2) } else { 0 }
    mean_validation_seconds = if ($allValTimes.Count -gt 0) { [math]::Round(($allValTimes | Measure-Object -Average).Average, 2) } else { 0 }
    rollout_count = $allRolloutTimes.Count
    block_count = $allBlockTimes.Count
    validation_count = $allValTimes.Count
    compute_seconds = [math]::Round(($allRolloutTimes | Measure-Object -Sum).Sum + ($allBlockTimes | Measure-Object -Sum).Sum + ($allValTimes | Measure-Object -Sum).Sum, 2)
}

$maxWall = ($runs | ForEach-Object { $_.total_wall_seconds } | Measure-Object -Maximum).Maximum
if ($maxWall -gt 0 -and $profiling.compute_seconds -gt 0) {
    $profiling['worker_utilization'] = [math]::Round($profiling.compute_seconds / ($maxWall * 3), 4)
} else {
    $profiling['worker_utilization'] = 0
}

$manifest = [ordered]@{
    experiment = 'p4_dueling_ddqn_convergence'
    architecture = 'dueling_ddqn'
    total_independent_runs = $totalRuns
    workers = 3
    per_run = [ordered]@{
        training_rollouts = 88
        optimizer_updates = 1280
        training_blocks = 18
        milestone_validation_blocks = @(4, 8, 12, 16)
        validation_seeds = @(1301, 1302)
        physical_validation_sims = 8
    }
    totals = [ordered]@{
        training_rollouts = $totalRuns * 88
        optimizer_updates = $totalRuns * 1280
        physical_validation_sims = $totalRuns * 8
        total_physical_sims = $totalRuns * 96
    }
    progress = [ordered]@{
        completed_runs = $completedRuns
        total_training_rollouts_done = $totalRollouts
        total_optimizer_blocks_done = $totalBlocks
        total_validation_milestones_done = $totalVals
        all_complete = $allComplete
    }
    runs = $runs
    profiling = $profiling
    generated_utc = (Get-Date).ToUniversalTime().ToString('o')
}

$manifestPath = Join-Path $summaryDir 'experiment_manifest.json'
$manifest | ConvertTo-Json -Depth 5 | Set-Content $manifestPath -Encoding utf8
Write-Output "Experiment manifest written to: $manifestPath"

$reportPath = Join-Path $summaryDir 'profiling_report.txt'
$lines = @()
$lines += '=== P4 Dueling DDQN Profiling Report ==='
$lines += "Generated: $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))"
$lines += ''
$lines += '--- Per-Run Summary ---'
$lines += ('{0,-8} {1,-12} {2,-10} {3,-8} {4,-14} {5,-14} {6,-14} {7,-12}' -f 'Run','Rollouts','Blocks','Vals','Rollout(s)','Block(s)','Val(s)','Wall(s)')
foreach ($r in $runs) {
    $lines += ('{0,-8} {1,-12} {2,-10} {3,-8} {4,-14} {5,-14} {6,-14} {7,-12}' -f $r.run_id,
        "$($r.rollouts_complete)/88", "$($r.blocks_complete)/18", $r.validations_complete,
        [math]::Round($r.total_rollout_seconds,1), [math]::Round($r.total_block_seconds,1),
        [math]::Round($r.total_validation_seconds,1), [math]::Round($r.total_wall_seconds,1))
}
$lines += ''
$lines += '--- Aggregate Profiling ---'
$lines += "Total compute time:        $([math]::Round($profiling.compute_seconds / 3600, 2)) hours"
$lines += "  Rollout time:            $([math]::Round($profiling.total_rollout_seconds / 3600, 2)) hours ($($profiling.rollout_count) rollouts)"
$lines += "  Optimizer block time:    $([math]::Round($profiling.total_block_seconds / 3600, 2)) hours ($($profiling.block_count) blocks)"
$lines += "  Validation time:         $([math]::Round($profiling.total_validation_seconds / 3600, 2)) hours ($($profiling.validation_count) milestones)"
$lines += "Mean rollout duration:     $($profiling.mean_rollout_seconds)s"
$lines += "Median rollout duration:   $($profiling.median_rollout_seconds)s"
$lines += "Mean block duration:       $($profiling.mean_block_seconds)s"
$lines += "Mean validation duration:  $($profiling.mean_validation_seconds)s"
$lines += "Max wall-clock (any run):  $([math]::Round($maxWall / 3600, 2)) hours"
$lines += "Worker utilization (3w):   $([math]::Round($profiling.worker_utilization * 100, 1))%"
$lines += ''
$lines += '--- Completion Status ---'
$lines += "Completed runs:            $completedRuns / $totalRuns"
$lines += "Total rollouts:            $totalRollouts / $($totalRuns * 88)"
$lines += "Total optimizer blocks:    $totalBlocks / $($totalRuns * 18)"
$lines += "Total validations:         $totalVals / $($totalRuns * 4)"
$lines += "All complete:              $allComplete"

$lines | Set-Content $reportPath -Encoding utf8
Write-Output "Profiling report written to: $reportPath"
Get-Content $reportPath
