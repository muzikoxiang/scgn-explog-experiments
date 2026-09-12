# 代码验证记录

初始代码验证日期：2026-09-09。环境：macOS CPU，Python 3.14，PyTorch 2.14.0，FP32。

已实际执行并通过：

- 六组模型 1×1×33×35 输入输出尺寸一致，输出有限，所有可训练参数均获得有限梯度。
- 每个主干的 matched/explog 参数数量及初始 state_dict 完全一致。
- 两种主干的 ExpLog 版各执行合成数据两步训练；连续训练与一步后断点续训的最终模型张量逐一完全相等。
- 保存/加载 checkpoint、三档噪声评估、训练测试相同文件拦截、非空训练目录覆盖保护。
- RAMiT baseline 严格加载官方 RAMiT_GDN.pth，所有键匹配；权重不随包分发，实验默认从头训练。
- CPU profile 六组执行成功。32×32 输入下，SCGN C/D 卷积 MACs 均为 1,345,536,000；RAMiT C/D 均为 723,623,168。该计数不包括其他算子，不代表总计算量相同。短计时仅用于检查程序，不作速度结论。

2026-09-11 至 2026-09-12，Windows 终端记录显示 PowerShell 入口、CUDA 训练和 Set12 评估已成功完成六组 seed=10、10k 实验，并观察到 ExpLog 相对 matched 的提升，详见 `EXPERIMENT_RESULTS.md`。但 Windows 原始目录尚未同步回 Mac，因此还未核验 manifest、逐图结果、完整日志、CUDA/profile 文件和目标环境版本。

尚未验证：断点续训在 Windows 上的逐张量一致性、实际 profile 文件、多个 seed、BSD68、充分训练收敛、论文指标复现、统计显著性、跨主干参数/总计算预算匹配。Mac 合成测试的 PSNR/SSIM 不属于科研实验结果。

注意：上游 meshgrid 调用在当前 PyTorch 给出未来需显式指定 indexing 的警告，目前不影响测试。
