# Calibrate shared P7 estimators from assembled datasets. Train seeds only.
# Does NOT launch simulations or DRL training.
param(
  [string]$DataRoot = "",
  [string]$ModelRoot = "",
  [string]$Python = "python"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
if ($DataRoot -eq "") { $DataRoot = Join-Path $root "outputs\p7_datasets" }
if ($ModelRoot -eq "") { $ModelRoot = Join-Path $root "models" }
& $Python (Join-Path $root "python\calibrate_p7_estimators.py") --data-root $DataRoot --model-root $ModelRoot
if ($LASTEXITCODE -ne 0) { throw "calibrate_p7_estimators failed with exit $LASTEXITCODE" }
Write-Host "P7 calibration finished -> $ModelRoot"
