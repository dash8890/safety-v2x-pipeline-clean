$ErrorActionPreference='Stop'
$root='C:\Users\dash\Downloads\NewProject\safety-v2x-pipeline\SafetyOffloading\workflow_v2'
$runner=Join-Path $root 'run_p4_online80_training.ps1'
foreach($id in 4..5) {
    while(@(Get-Process MATLAB -ErrorAction SilentlyContinue).Count -ge 3) { Start-Sleep -Seconds 15 }
    $dir=Join-Path $root ("outputs\p4_compact_validation_dueling_run{0:D2}" -f $id)
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    Start-Process -FilePath 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',$runner,'-RunId',$id) -WindowStyle Hidden -RedirectStandardOutput (Join-Path $dir 'runner.log') -RedirectStandardError (Join-Path $dir 'runner.error.log') | Out-Null
}
