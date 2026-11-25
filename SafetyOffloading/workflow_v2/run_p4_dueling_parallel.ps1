param(
    [ValidateRange(1,2)][int]$MaxParallel = 2
)
# Independent runs have separate output roots and seeds.  Two concurrent
# MATLAB jobs are deliberately capped on this 8-core host until timings show
# that a higher level preserves the single-run baseline throughput.
$root='C:\Users\dash\Downloads\NewProject\safety-v2x-pipeline\SafetyOffloading\workflow_v2'
$runner=Join-Path $root 'run_p4_online80_training.ps1'
$active=@()
foreach($id in 1..5) {
    while(@($active | Where-Object { -not $_.HasExited }).Count -ge $MaxParallel) { Start-Sleep -Seconds 15; $active=@($active | Where-Object { -not $_.HasExited }) }
    $log=Join-Path $root ("outputs\p4_online80_dueling_run{0:D2}\runner.log" -f $id)
    New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
    $active += Start-Process -FilePath 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',$runner,'-RunId',$id) -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError ($log+'.error') -PassThru
}
$active | Wait-Process
