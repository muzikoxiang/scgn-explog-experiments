param(
    [ValidateSet('setup','verify','profile','train','evaluate')][string]$Action = 'verify',
    [string]$Backbone = 'scgn',
    [string]$Variant = 'baseline',
    [int]$Seed = 10,
    [string]$Config = 'configs/screen.json',
    [string]$Output = '',
    [string]$Checkpoint = '',
    [string]$TestRoot = '',
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
function Check-Exit { if ($LASTEXITCODE -ne 0) { throw "Command failed: $LASTEXITCODE" } }
if ($Action -eq 'setup') {
    py -3.11 -m venv .venv
    Check-Exit
    & .\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
    Check-Exit
    & .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    Check-Exit
    exit
}
$Python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (!(Test-Path $Python)) { throw 'Run windows.ps1 -Action setup first' }
if ($Action -eq 'verify') { & $Python verify.py }
elseif ($Action -eq 'profile') {
    & $Python run.py --device cuda profile --output profile_windows.json
}
elseif ($Action -eq 'train') {
    if (!$Output) { $Output = "runs/${Backbone}_${Variant}_seed${Seed}" }
    $Arguments = @('run.py','--device','cuda','train','--config',$Config,'--backbone',$Backbone,'--variant',$Variant,'--seed',"$Seed",'--output',$Output)
    if ($Resume) { $Arguments += '--resume' }
    & $Python @Arguments
}
elseif ($Action -eq 'evaluate') {
    if (!$Checkpoint -or !$TestRoot -or !$Output) { throw 'Specify Checkpoint, TestRoot and Output' }
    & $Python run.py --device cuda evaluate --checkpoint $Checkpoint --test-root $TestRoot --output $Output
}
Check-Exit
