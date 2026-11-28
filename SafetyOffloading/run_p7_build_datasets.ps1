# Build P7 datasets from finished P4/P6 outputs. Does NOT launch simulations.
# Prefers the Python builder so it never contends for MATLAB licenses with P6.
param(
  [string]$OutputRoot = "",
  [string]$Python = "python",
  [switch]$UseMatlab,
  [string]$Matlab = "matlab"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if ($UseMatlab) {
  if ($OutputRoot -ne "") {
    $cmd = "try, buildP7Datasets('OutputRoot','$OutputRoot'); catch ME, disp(getReport(ME,'extended')); exit(1); end; exit(0);"
  } else {
    $cmd = "try, buildP7Datasets(); catch ME, disp(getReport(ME,'extended')); exit(1); end; exit(0);"
  }
  & $Matlab -batch $cmd
  if ($LASTEXITCODE -ne 0) { throw "buildP7Datasets failed with exit $LASTEXITCODE" }
} else {
  $args = @((Join-Path $root "python\build_p7_datasets.py"))
  if ($OutputRoot -ne "") { $args += @("--output-root", $OutputRoot) }
  & $Python @args
  if ($LASTEXITCODE -ne 0) { throw "build_p7_datasets.py failed with exit $LASTEXITCODE" }
}
Write-Host "P7 dataset build finished."
