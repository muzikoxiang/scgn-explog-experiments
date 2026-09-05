$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..\..")
conda run -n span-explog python -c "import torch; print('torch=',torch.__version__); print('cuda_build=',torch.version.cuda); print('cuda_available=',torch.cuda.is_available()); print('gpu=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
conda run -n span-explog python tools/smoke_test_explog.py --device cuda
