param(
    [Parameter(Mandatory=$true)][string]$DatasetRoot,
    [Parameter(Mandatory=$true)][string]$Checkpoint
)
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..\..")
conda run -n span-explog python tools/prepare_windows_configs.py --dataset-root $DatasetRoot --checkpoint $Checkpoint
