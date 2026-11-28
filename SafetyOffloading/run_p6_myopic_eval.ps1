param(
    [string]$Config = "SafetyOffloading/config/training_data.json",
    [string]$BaselineRoot = "",
    [string]$RunRoot = "",
    [ValidateRange(1, 6)][int]$Workers = 3,
    [int[]]$ConfigurationIds = @(1,5,6,22,31),
    [int[]]$Seeds = @(1,2,3),
    [double]$WarmupSeconds = 2,
    [double]$ObservationSeconds = 20
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$configPath = (Resolve-Path (Join-Path $repo $Config)).Path
if (-not $BaselineRoot) { $BaselineRoot = Join-Path $PSScriptRoot "outputs\opportunity_sweep_p4" }
$BaselineRoot = [IO.Path]::GetFullPath($BaselineRoot)
if (-not $RunRoot) { $RunRoot = Join-Path $PSScriptRoot "outputs\p6_myopic_eval" }
$RunRoot = [IO.Path]::GetFullPath($RunRoot)
New-Item -ItemType Directory -Path $RunRoot -Force | Out-Null

# Spec lookup from baseline matrix
$matrix = Import-Csv (Join-Path $BaselineRoot "configuration_matrix.csv")
function Quote-Matlab([string]$Value) { $Value.Replace("\", "/").Replace("'", "''") }

$jobs = @()
foreach ($cfgId in $ConfigurationIds) {
    $m = $matrix | Where-Object { [int]$_.configuration_id -eq $cfgId } | Select-Object -First 1
    if (-not $m) { throw "Missing configuration $cfgId in baseline matrix" }
    foreach ($seed in $Seeds) {
        $baseCase = Join-Path $BaselineRoot ("cfg{0:D2}_seed{1}" -f $cfgId, $seed)
        if (-not (Test-Path (Join-Path $baseCase "CASE_COMPLETE.mat"))) {
            Write-Output "[skip-missing-baseline] cfg=$cfgId seed=$seed"; continue
        }
        $jobs += [pscustomobject]@{ configuration_id=[int]$cfgId; seed=[int]$seed; density=[double]$m.density;
            origin_cpu_ghz=[double]$m.origin_cpu_ghz; helper_cpu_ghz=[double]$m.helper_cpu_ghz;
            arrival_rate=[double]$m.arrival_rate; helper_background_utilization=0.20;
            selection_window_slots=40; warmup_s=$WarmupSeconds; observation_s=$ObservationSeconds;
            baseline_case=$baseCase }
    }
}
Write-Output "run_root=$RunRoot jobs=$($jobs.Count) workers=$Workers"

$pending = [Collections.Queue]::new(); foreach ($j in $jobs) { $pending.Enqueue($j) }
$active = @(); $failed = @(); $started = Get-Date
while ($pending.Count -gt 0 -or $active.Count -gt 0) {
    while ($pending.Count -gt 0 -and $active.Count -lt $Workers) {
        $job = $pending.Dequeue()
        $caseRoot = Join-Path $RunRoot ("cfg{0:D2}_seed{1}" -f $job.configuration_id, $job.seed)
        $marker = Join-Path $caseRoot "CASE_COMPLETE.mat"
        if (Test-Path $marker) { Write-Output "[resume] $caseRoot"; continue }
        New-Item -ItemType Directory -Path $caseRoot -Force | Out-Null
        $repoQ = Quote-Matlab $repo
        $cfgQ = Quote-Matlab $configPath
        $caseQ = Quote-Matlab $caseRoot
        $baseQ = Quote-Matlab $BaselineRoot
        $runner = Join-Path $caseRoot "run_case.m"
        @"
addpath(genpath('$repoQ'));
spec=struct('configuration_id',$($job.configuration_id),'seed',$($job.seed),'density',$($job.density),'origin_cpu_ghz',$($job.origin_cpu_ghz),'helper_cpu_ghz',$($job.helper_cpu_ghz),'arrival_rate',$($job.arrival_rate),'helper_background_utilization',$($job.helper_background_utilization),'selection_window_slots',$($job.selection_window_slots),'warmup_s',$WarmupSeconds,'observation_s',$ObservationSeconds);
try
  runP6MyopicCase('$cfgQ','$caseQ',spec,'$baseQ');
catch ME
  disp(getReport(ME,'extended'));
  exit(1);
end
exit(0);
"@ | Set-Content -Path $runner -Encoding ASCII
        $runnerQ = Quote-Matlab $runner
        $stdout = Join-Path $caseRoot "matlab.log"; $stderr = Join-Path $caseRoot "matlab.err.log"
        $process = Start-Process -FilePath "matlab" -ArgumentList @("-singleCompThread","-batch","run('$runnerQ')") `
            -WorkingDirectory $repo -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
            -WindowStyle Hidden -PassThru
        $active += [pscustomobject]@{ Process=$process; Job=$job; Root=$caseRoot; Marker=$marker; Start=Get-Date }
        Write-Output "[started pid=$($process.Id)] config=$($job.configuration_id) seed=$($job.seed)"
    }
    if ($active.Count -gt 0) {
        Start-Sleep -Seconds 5
        $remaining = @()
        foreach ($item in $active) {
            try { $item.Process.Refresh() } catch { }
            $alive = $false
            try { $alive = -not $item.Process.HasExited } catch { $alive = $false }
            # Fallback: PID may outlive the Process object handle
            if ($alive) {
                $alive = $null -ne (Get-Process -Id $item.Process.Id -ErrorAction SilentlyContinue)
            }
            if (-not $alive) {
                $minutes = [math]::Round(((Get-Date) - $item.Start).TotalMinutes, 1)
                if (-not (Test-Path $item.Marker)) {
                    $failed += $item.Root
                    $errTail = ""
                    if (Test-Path (Join-Path $item.Root "matlab.err.log")) {
                        $errTail = (Get-Content (Join-Path $item.Root "matlab.err.log") -Tail 8) -join " | "
                    }
                    Write-Output "[FAILED $minutes min] $($item.Root) $errTail"
                } else { Write-Output "[complete $minutes min] config=$($item.Job.configuration_id) seed=$($item.Job.seed)" }
            } else { $remaining += $item }
        }
        $active = $remaining
        try {
            $status = Join-Path $RunRoot "eval_status.txt"
            "$((Get-Date).ToString('s')) pending=$($pending.Count) active=$($active.Count) failed=$($failed.Count)" | Set-Content -Path $status -Force -ErrorAction Stop
        } catch {
            # Status file is best-effort; never abort the eval on a lock/IO race.
        }
    }
}

$rows = @()
foreach ($job in $jobs) {
    $csv = Join-Path $RunRoot ("cfg{0:D2}_seed{1}\case_summary.csv" -f $job.configuration_id, $job.seed)
    if (-not (Test-Path $csv)) { continue }
    foreach ($r in Import-Csv $csv) {
        $rows += [pscustomobject]@{ configuration_id=$job.configuration_id; seed=$job.seed; density=$job.density;
            origin_cpu_ghz=$job.origin_cpu_ghz; helper_cpu_ghz=$job.helper_cpu_ghz; arrival_rate=$job.arrival_rate;
            method=$r.method; completed=$r.completed; tasks=$r.tasks; relative_gain=$r.relative_gain;
            offloaded=$r.offloaded; harmful_failure_probability=$r.harmful_failure_probability;
            absolute_safety_failure=$r.absolute_safety_failure; harm_constraint_met=$r.harm_constraint_met;
            local_completed=$r.local_completed; safety_search_completed=$r.safety_search_completed;
            learning_headroom=$r.learning_headroom }
    }
}
$rows | Export-Csv (Join-Path $RunRoot "p6_summary.csv") -NoTypeInformation
Write-Output "done elapsed_min=$([math]::Round(((Get-Date)-$started).TotalMinutes,1)) completed=$($rows.Count) failed=$($failed.Count)"
