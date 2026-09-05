$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..\..")
$Config = "options/experiments/generated/test_SPAN_x4_windows.yml"
if (-not (Test-Path $Config)) {
    throw "Run 02_prepare_configs.ps1 with your dataset root and checkpoint first."
}
conda run -n span-explog python basicsr/test.py -opt $Config
