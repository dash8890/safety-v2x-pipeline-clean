param(
    [string]$Config = "SafetyOffloading/config/training_data.json",
    [string]$RunRoot = "",
    [ValidateRange(1, 12)][int]$Workers = 2,
    [double]$WarmupSeconds = 2,
    [double]$ObservationSeconds = 20,
    [int[]]$Seeds = @(1,2,3,4,5),
    [int]$FirstConfiguration = 1,
    [int]$LastConfiguration = 72,
    [int[]]$ConfigurationIds = @(),
    [double]$ReceiverSampleFraction = 0.10
)
# P4 opportunity matrix.  Full 50/100 veh/km grids are retained.  At
# 150 veh/km only four representative stress-test configs are run (H1-H4);
# exhaustive high-density evaluation is deferred to P10.  Instance
# directories keep the original configuration_id numbering so batches
# share one RunRoot and resume without renumbering.
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$configPath = (Resolve-Path (Join-Path $repo $Config)).Path
if (-not $RunRoot) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $RunRoot = Join-Path $PSScriptRoot "outputs\opportunity_sweep_$stamp"
}
$RunRoot = [System.IO.Path]::GetFullPath($RunRoot)
New-Item -ItemType Directory -Path $RunRoot -Force | Out-Null
Copy-Item -LiteralPath $configPath -Destination (Join-Path $RunRoot "base_config.json") -Force

function Quote-Matlab([string]$Value) { $Value.Replace("\", "/").Replace("'", "''") }

# Representative 150 veh/km stress-test slice (H1-H4).
$highDensityKeep = @(
    @{ origin=0.5; helper=2.0; arrival=6  },
    @{ origin=0.5; helper=2.0; arrival=12 },
    @{ origin=0.7; helper=2.0; arrival=6  },
    @{ origin=0.7; helper=2.0; arrival=12 }
)

$matrix = @(); $p4Active = @(); $configurationId = 0
foreach ($density in @(50,100,150)) {
    foreach ($originCpu in @(0.5,0.6,0.7,1.0)) {
        foreach ($helperCpu in @(1.5,2.0)) {
            foreach ($arrivalRate in @(6,12,16)) {
                $configurationId++
                $row = [pscustomobject]@{ configuration_id=$configurationId; density=$density;
                    origin_cpu_ghz=$originCpu; helper_cpu_ghz=$helperCpu; arrival_rate=$arrivalRate;
                    helper_background_utilization=0.20; selection_window_slots=40; p4_active=$false }
                $keep = $density -lt 150
                if (-not $keep) {
                    foreach ($h in $highDensityKeep) {
                        if ($originCpu -eq $h.origin -and $helperCpu -eq $h.helper -and $arrivalRate -eq $h.arrival) {
                            $keep = $true; break
                        }
                    }
                }
                $row.p4_active = $keep
                $matrix += $row
                if ($keep) { $p4Active += $row }
            }
        }
    }
}
$matrix | Export-Csv -LiteralPath (Join-Path $RunRoot "configuration_matrix.csv") -NoTypeInformation
$p4Active | Export-Csv -LiteralPath (Join-Path $RunRoot "configuration_matrix_p4_active.csv") -NoTypeInformation

function Get-CaseRoot($job) { Join-Path $RunRoot ("cfg{0:D2}_seed{1}" -f $job.configuration_id, $job.seed) }

$selected = @()
foreach ($m in $p4Active) {
    if ($m.configuration_id -lt $FirstConfiguration -or $m.configuration_id -gt $LastConfiguration) { continue }
    if ($ConfigurationIds.Count -gt 0 -and ($ConfigurationIds -notcontains [int]$m.configuration_id)) { continue }
    foreach ($seed in $Seeds) {
        $selected += [pscustomobject]@{ configuration_id=$m.configuration_id; seed=$seed; density=$m.density;
            origin_cpu_ghz=$m.origin_cpu_ghz; helper_cpu_ghz=$m.helper_cpu_ghz; arrival_rate=$m.arrival_rate;
            helper_background_utilization=$m.helper_background_utilization;
            selection_window_slots=$m.selection_window_slots }
    }
}
$byDensity = $selected | Group-Object density | ForEach-Object { "density_$($_.Name)=$($_.Count)" }
$idNote = if ($ConfigurationIds.Count -gt 0) { "configuration_ids=$($ConfigurationIds -join ',')" } else { "configuration_ids=all_active" }
Write-Output "run_root=$RunRoot full_matrix=$($matrix.Count) p4_active_configs=$($p4Active.Count) selected_instances=$($selected.Count) workers=$Workers $idNote $($byDensity -join ' ')"

$pending = [System.Collections.Queue]::new()
foreach ($job in $selected) { $pending.Enqueue($job) }
$active = @(); $failed = @(); $started = Get-Date
while ($pending.Count -gt 0 -or $active.Count -gt 0) {
    while ($pending.Count -gt 0 -and $active.Count -lt $Workers) {
        $job = $pending.Dequeue(); $caseRoot = Get-CaseRoot $job
        $marker = Join-Path $caseRoot "CASE_COMPLETE.mat"
        if (Test-Path -LiteralPath $marker) { Write-Output "[resume] $caseRoot"; continue }
        New-Item -ItemType Directory -Path $caseRoot -Force | Out-Null
        $spec = "struct('configuration_id',$($job.configuration_id),'seed',$($job.seed),'density',$($job.density),'origin_cpu_ghz',$($job.origin_cpu_ghz),'helper_cpu_ghz',$($job.helper_cpu_ghz),'arrival_rate',$($job.arrival_rate),'helper_background_utilization',$($job.helper_background_utilization),'selection_window_slots',$($job.selection_window_slots),'warmup_s',$WarmupSeconds,'observation_s',$ObservationSeconds,'receiver_sample_fraction',$ReceiverSampleFraction)"
        $command = "addpath(genpath(pwd)); spec=$spec; runWiLabOpportunitySweepCase('$(Quote-Matlab $configPath)','$(Quote-Matlab $caseRoot)',spec);"
        $batchArgument = '"' + $command.Replace('"','\"') + '"'
        $stdout = Join-Path $caseRoot "matlab.log"; $stderr = Join-Path $caseRoot "matlab.err.log"
        $process = Start-Process -FilePath "matlab" -ArgumentList @("-singleCompThread","-batch",$batchArgument) `
            -WorkingDirectory $repo -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
            -WindowStyle Hidden -PassThru
        $active += [pscustomobject]@{ Process=$process; Job=$job; Root=$caseRoot; Marker=$marker; Start=Get-Date }
        Write-Output "[started pid=$($process.Id)] config=$($job.configuration_id) seed=$($job.seed)"
    }
    if ($active.Count -gt 0) {
        Start-Sleep -Seconds 5
        $remaining = @()
        foreach ($item in $active) {
            if ($item.Process.HasExited) {
                $minutes = [math]::Round(((Get-Date) - $item.Start).TotalMinutes, 1)
                if (-not (Test-Path -LiteralPath $item.Marker)) {
                    $failed += $item.Root
                    Write-Output "[FAILED after $minutes min] $($item.Root) (see matlab.log / matlab.err.log)"
                } else { Write-Output "[complete $minutes min] config=$($item.Job.configuration_id) seed=$($item.Job.seed)" }
            } else { $remaining += $item }
        }
        $active = $remaining
    }
}

# Cumulative summary over every finished instance in the run root.
$summary = foreach ($m in $matrix) {
    foreach ($dir in (Get-ChildItem -LiteralPath $RunRoot -Directory -Filter ("cfg{0:D2}_seed*" -f $m.configuration_id))) {
        $resultPath = Join-Path $dir.FullName "case_summary.csv"
        if (-not (Test-Path -LiteralPath $resultPath)) { continue }
        $seed = [int]($dir.Name -replace '.*_seed','')
        foreach ($row in (Import-Csv -LiteralPath $resultPath)) {
            $out = [ordered]@{ configuration_id=$m.configuration_id; seed=$seed; density=$m.density;
                origin_cpu_ghz=$m.origin_cpu_ghz; helper_cpu_ghz=$m.helper_cpu_ghz; arrival_rate=$m.arrival_rate;
                helper_background_utilization=$m.helper_background_utilization;
                selection_window_slots=$m.selection_window_slots }
            foreach ($p in $row.PSObject.Properties) { $out[$p.Name] = $p.Value }
            [pscustomobject]$out
        }
    }
}
if ($summary) { $summary | Export-Csv -LiteralPath (Join-Path $RunRoot "sweep_summary.csv") -NoTypeInformation }
$totalMin = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)
Write-Output "done run_root=$RunRoot elapsed_min=$totalMin completed=$((Get-ChildItem -Path $RunRoot -Filter CASE_COMPLETE.mat -Recurse).Count) failed=$($failed.Count)"
foreach ($f in $failed) { Write-Output "failed: $f" }
