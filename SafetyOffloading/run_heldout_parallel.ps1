param(
    [string]$Config = "SafetyOffloading/config/heldout.json",
    [string]$RunRoot = "",
    [ValidateRange(1, 8)][int]$Workers = 4
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$configPath = (Resolve-Path (Join-Path $repo $Config)).Path
$cfg = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
if (-not $RunRoot) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $RunRoot = Join-Path $PSScriptRoot (Join-Path $cfg.experiment.output_root "heldout_$($cfg.project.profile)_$stamp")
}
$RunRoot = [System.IO.Path]::GetFullPath($RunRoot)
New-Item -ItemType Directory -Path $RunRoot -Force | Out-Null
Copy-Item -LiteralPath $configPath -Destination (Join-Path $RunRoot "config.json") -Force

function Quote-Matlab([string]$Value) { return $Value.Replace("\", "/").Replace("'", "''") }

$scenarios = @(); $scenarioId = 0
for ($di = 0; $di -lt $cfg.mobility.density_vehicles_per_km.Count; $di++) {
    $density = $cfg.mobility.density_vehicles_per_km[$di]
    $speed = $cfg.mobility.speed_kmh_mean[[Math]::Min($di, $cfg.mobility.speed_kmh_mean.Count - 1)]
    $speedStd = $cfg.mobility.speed_kmh_std[[Math]::Min($di, $cfg.mobility.speed_kmh_std.Count - 1)]
    foreach ($seed in $cfg.experiment.seeds) {
        $scenarioId++
        $scenarios += [pscustomobject]@{ ScenarioId=$scenarioId; Seed=$seed; Density=$density; Speed=$speed; SpeedStd=$speedStd }
    }
}

function Invoke-Phase([array]$Jobs, [string]$Name) {
    Write-Output "phase=$Name jobs=$($Jobs.Count) workers=$Workers"
    $pending = [System.Collections.Queue]::new()
    foreach ($job in $Jobs) { $pending.Enqueue($job) }
    $active = @()
    while ($pending.Count -gt 0 -or $active.Count -gt 0) {
        while ($pending.Count -gt 0 -and $active.Count -lt $Workers) {
            $job = $pending.Dequeue()
            $caseRoot = Join-Path $RunRoot ("scenario_{0:D4}" -f $job.ScenarioId) | Join-Path -ChildPath $job.Policy
            $marker = Join-Path $caseRoot "COMPLETE.mat"
            if (Test-Path -LiteralPath $marker) { Write-Output "[resume] $caseRoot"; continue }
            New-Item -ItemType Directory -Path $caseRoot -Force | Out-Null
            $command = "addpath(genpath(pwd)); runWiLabHeldoutCase('$(Quote-Matlab $configPath)','$(Quote-Matlab $RunRoot)',$($job.ScenarioId),$($job.Seed),$($job.Density),$($job.Speed),$($job.SpeedStd),'$($job.Policy)');"
            $batchArgument = '"' + $command.Replace('"','\"') + '"'
            $stdout = Join-Path $caseRoot "matlab.log"; $stderr = Join-Path $caseRoot "matlab.err.log"
            $process = Start-Process -FilePath "matlab" -ArgumentList @("-singleCompThread","-batch",$batchArgument) `
                -WorkingDirectory $repo -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
                -WindowStyle Hidden -PassThru
            $active += [pscustomobject]@{ Process=$process; Job=$job; Root=$caseRoot; Marker=$marker }
            Write-Output "[started pid=$($process.Id)] $caseRoot"
        }
        if ($active.Count -gt 0) {
            Start-Sleep -Seconds 2
            $remaining = @()
            foreach ($item in $active) {
                if ($item.Process.HasExited) {
                    if ($item.Process.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $item.Marker)) {
                        throw "Case failed: $($item.Root). See matlab.log and matlab.err.log."
                    }
                    Write-Output "[complete] $($item.Root)"
                } else { $remaining += $item }
            }
            $active = $remaining
        }
    }
}

$baselines = foreach ($scenario in $scenarios) {
    [pscustomobject]@{ ScenarioId=$scenario.ScenarioId; Seed=$scenario.Seed; Density=$scenario.Density; Speed=$scenario.Speed; SpeedStd=$scenario.SpeedStd; Policy="no_offload" }
}
$policies = foreach ($scenario in $scenarios) { foreach ($policy in $cfg.experiment.policies) {
    [pscustomobject]@{ ScenarioId=$scenario.ScenarioId; Seed=$scenario.Seed; Density=$scenario.Density; Speed=$scenario.Speed; SpeedStd=$scenario.SpeedStd; Policy=$policy }
}}

Write-Output "run_root=$RunRoot"
Invoke-Phase $baselines "baselines"
Invoke-Phase $policies "policies"
$summary = foreach ($job in $policies) {
    $resultPath = Join-Path $RunRoot ("scenario_{0:D4}" -f $job.ScenarioId) | Join-Path -ChildPath $job.Policy | Join-Path -ChildPath "case_result.csv"
    $result = Import-Csv -LiteralPath $resultPath
    [pscustomobject]@{ scenario_id=$job.ScenarioId; seed=$job.Seed; density=$job.Density; policy=$job.Policy;
        safety_opportunities=$result.safety_opportunities; harm_probability=$result.harm_probability;
        policy_failure_probability=$result.policy_failure_probability; baseline_failure_probability=$result.baseline_failure_probability;
        avoided_failure_probability=$result.avoided_failure_probability; tasks_completed=$result.tasks_completed;
        fragment_count=$result.fragment_count; status="complete" }
}
$summary | Export-Csv -LiteralPath (Join-Path $RunRoot "summary.csv") -NoTypeInformation
Write-Output "complete=$RunRoot"
