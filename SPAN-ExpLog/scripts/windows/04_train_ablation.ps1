param([Parameter(Mandatory=$true)][string]$Config)
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..\..")
if (-not (Test-Path $Config)) { throw "Config not found: $Config" }
conda run -n span-explog python basicsr/train.py -opt $Config --launcher none
