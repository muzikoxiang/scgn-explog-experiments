$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..\..")
conda run -n span-explog python tools/smoke_test_explog.py --device cuda --size 24
