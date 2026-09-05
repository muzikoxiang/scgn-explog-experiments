# SPAN-ExpLog Windows 运行手册

## 状态

`prepared` / `mac-smoke-tested` / **not yet windows-verified**

本目录基于 SPAN 官方仓库的 `c77a591` 准备。现阶段的数值不得标记为 Windows/CUDA 复现结果。

## 1. 拷贝与环境

将整个 `SPAN-ExpLog` 目录拷到 Windows，在 Anaconda PowerShell Prompt 中运行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\windows\00_setup.ps1
```

`00_setup.ps1` 不会自动猜测 CUDA 版本。按 PyTorch 官网选择与驱动匹配的 CUDA wheel，然后执行：

```powershell
conda run -n span-explog pip install -r requirements-windows.txt
conda run -n span-explog pip install -e .
.\scripts\windows\01_check_env.ps1
```

## 2. 数据与官方基线

1. 按 `options/experiments/test_SPAN_x4_template.yml` 的目录约定准备 Set5、Set14、B100、Urban100、Manga109。
2. 用一条命令生成基线和六组冒烟实验配置：

```powershell
.\scripts\windows\02_prepare_configs.ps1 -DatasetRoot D:\datasets -Checkpoint D:\models\SPAN\SPAN_x4.pth
```

3. 执行 `.\scripts\windows\03_test_baseline.ps1`。

只有基线指标与官方结果基本对齐后，才开始完整训练。

## 3. 消融配置

`02_prepare_configs.ps1` 会在 `options/experiments/generated/` 自动生成下面的配置，每个实验都有唯一的 `name`。

自动生成的 1000 iter 冒烟配置使用 `ema_decay: 0`，确保验证当前
`net_g`。正式长训练时再恢复 `ema_decay: 0.999`。

| 实验 | type | enhance_blocks | enhancer |
|---|---|---|---|
| A 官方基线 | `SPAN` | 删除 | 删除 |
| B GN对照 | `SPANExpLog` | `[3]` | `gn` |
| C 参数匹配卷积 | `SPANExpLog` | `[3]` | `matched_conv` |
| D 单点 ExpLog | `SPANExpLog` | `[3]` | `explog` |
| E 半数块 ExpLog | `SPANExpLog` | `[2, 4, 6]` | `explog` |
| F 全部 ExpLog | `SPANExpLog` | `[1, 2, 3, 4, 5, 6]` | `explog` |

注：6 个 SPAB 无法精确实现 25%，所以使用可复现的 1/6、3/6、6/6，不在报告中写成“25%”。

运行示例：

```powershell
.\scripts\windows\04_train_ablation.ps1 -Config options/experiments/generated/train_D_explog_b3_x4_smoke.yml
```

## 4. 第一次只做冒烟训练

将配置中 `total_iter` 改为 `1000`、`val_freq` 改为 `500`、`save_checkpoint_freq` 改为 `500`。检查 loss、NaN/Inf、显存和输出图像后，再恢复完整训练。

## 5. Windows 验收记录

运行时记录：GPU、驱动、CUDA build、PyTorch、配置文件、git/ZIP 版本、随机种子、峰值显存、训练时间、五个数据集的 PSNR/SSIM。

## 6. 当前执行顺序

A 与 D 已通过 1000 iter 冒烟训练。下一步先跑 B，再跑 C，以判断 D
的早期提升来自 ExpLog、GN，还是单纯增加参数/计算量。B/C 完成前暂缓 E/F。
