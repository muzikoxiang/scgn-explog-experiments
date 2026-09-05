$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..\..")

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found. Install Miniconda/Anaconda and reopen PowerShell."
}

conda create -n span-explog python=3.9 -y
conda run -n span-explog python -m pip install --upgrade pip
Write-Host "Install a CUDA-compatible PyTorch build next, following https://pytorch.org/get-started/locally/"
Write-Host "Then run: conda run -n span-explog pip install -r requirements-windows.txt"
Write-Host "Then run: conda run -n span-explog pip install -e ."
