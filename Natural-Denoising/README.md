# 自然图像去噪：SCGN / RAMiT × ExpLog

状态：独立实验代码包已在 Windows/CUDA 完成首轮六组 10k 初筛，并在 Set12 上评估。结果见 `EXPERIMENT_RESULTS.md`。这不是论文指标复现，也不能称为 SOTA。

## 这次比较什么

输入/输出均为灰度图。训练加入 σ∈[0,50] 的高斯噪声；测试固定 σ=15、25、50。六组均从头训练，使用相同数据、步数、裁剪、Adam、L1 和学习率计划。

| 主干 | baseline 参数 | matched 参数 | explog 参数 |
|---|---:|---:|---:|
| SCGN | 1,640,641 | 1,668,513 | 1,668,513 |
| RAMiT | 932,271 | 1,072,895 | 1,072,895 |

- baseline：保留原主干；SCGN 使用原始 8 块、BN 配置。
- matched：局部模块替换为两个带 GN 和残差的普通卷积块。
- explog：同位置、同参数、同初始化，将 matched 的两个分支差改为历史 ExpLog 非线性。
- SCGN 替换第一块的第一个局部特征提取模块；RAMiT 替换 stage1.block0 的 MobiVari 特征混合模块。其余部分不变。
- C/D 比较用于隔离 ExpLog 非线性；A/D 同时改变了局部模块结构，不能单独归因于 ExpLog。

**尚未满足师兄的跨主干等预算条件。** SCGN 和 RAMiT 参数量不相同；C/D 仅参数量与卷积 MACs 相同，ExpLog 额外运算仍需考虑。`profile` 只统计卷积 MACs，不包含 FFT、注意力矩阵乘、exp/log 等，不得当作总 FLOPs。正式研究还需要跨主干预算匹配、多个 seed 和充分训练。

RAMiT 是 CVPR 2024 Workshop 的轻量候选，不是已确认的 2026 最新 SOTA。本包为统一协议受控实验，不复制两篇论文各自完整训练方案。

## Windows 使用

将整个 `Natural-Denoising` 文件夹复制到 Windows。先安装 Python 3.11 和适用的 NVIDIA 显卡驱动；环境安装需要联网。打开此文件夹的 PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File .\windows.ps1 -Action setup
powershell -ExecutionPolicy Bypass -File .\windows.ps1 -Action verify
powershell -ExecutionPolicy Bypass -File .\windows.ps1 -Action profile
```

setup 在独立 `.venv` 中安装 PyTorch 2.5.1 CUDA 11.8，不改变已有实验环境。verify 是 CPU 功能测试；profile 才检查 CUDA 前向与实际显存/延迟。安装或 CUDA 失败时先保存报错，不要跳过检查直接长训。

编辑 `configs/screen.json` 的 `train_roots`，填实际干净自然图像目录，例如 `D:/datasets/denoising/train_clean`。目录可以递归包含 PNG/JPG 等。无需预先生成噪声，图片短边应至少为 patch 大小。

数据建议按官方数据划分准备 DIV2K/Flickr2K/BSD500/WED；**不能把 BSD68 测试图或其他测试图放入训练集**。程序能拦截字节完全相同的训练/测试文件，不能替代数据划分审计。验证集另留，不用测试集调参。不同数据子集结果不可冒充论文复现。

先运行一组确认显存、速度和损失正常：

```powershell
powershell -ExecutionPolicy Bypass -File .\windows.ps1 -Action train -Backbone scgn -Variant baseline
```

默认 10,000 步、patch 64、batch 4，只用于初筛，不能证明收敛性能。显存不足时可降低 batch，但六组必须使用同一新配置、重新训练并记录变化。确认稳定后，逐组运行六个组合：

```powershell
foreach ($b in @('scgn','ramit')) {
  foreach ($v in @('baseline','matched','explog')) {
    powershell -ExecutionPolicy Bypass -File .\windows.ps1 -Action train -Backbone $b -Variant $v
    if ($LASTEXITCODE -ne 0) { throw 'Training failed; stop and inspect logs' }
  }
}
```

已运行的组须跳过，或显式加 `-Resume`。程序拒绝覆盖非空训练目录；续训校验数据、配置、seed 和模型一致。只加载自己可信来源的 checkpoint。

```powershell
powershell -ExecutionPolicy Bypass -File .\windows.ps1 -Action train -Backbone scgn -Variant baseline -Resume
powershell -ExecutionPolicy Bypass -File .\windows.ps1 -Action evaluate -Checkpoint runs/scgn_baseline_seed10/latest.pt -TestRoot D:/datasets/Set12 -Output runs/scgn_baseline_seed10/eval_Set12
```

其余模型/测试集替换路径，使用同一步数的 checkpoint。评估输出 `metrics.json` 和 `per_image.csv`；每张图的噪声按文件内容固定，跨模型相同。输入不截断，输出截断到 [0,1]，全图 PSNR、Gaussian SSIM，无边界裁剪/自集成。此评估口径不保证与原论文完全一致。相同评估输出目录会被更新，需保留旧结果时另设目录。

训练输出含 `manifest.json`、`train.jsonl`、`latest.pt`、阶段 checkpoint。正式阶段预先确定预算与验证方案，再增加 seed 20、30；不要根据测试分数挑 checkpoint。

## 搬回 Mac 的内容

带回 `profile_windows.json`、各组 manifest、训练日志、评估 JSON/CSV、实际使用的配置，以及 Windows 的报错或环境版本。权重自行备份，不提交 Git。首轮 Windows 结果已通过聊天中的终端输出确认；原始结果文件搬回 Mac 并核对后，再提交代码和实验结果。当前包未自动提交或推送。

来源和分发限制见 `PROVENANCE.md`，本地验证记录见 `VALIDATION.md`。
