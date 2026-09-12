# 来源与边界

- `nets/scgn.py`：来自本工作区 `models/scgn_variants.py`，保留历史实现。当前 factory 只以 baseline 配置作为起点。
- `nets/ramit.py`：来自 https://github.com/rami0205/RAMiT ，commit `193ead604a1ab01dd548eb808bf6a11f873fec39`。保留网络源码，仅替换依赖导入；`compat.py` 提供本配置所需的 padding、归一化与装饰器兼容实现。不支持该仓库全部任务/配置。
- `factory.py`、训练/评估入口与 Windows 入口：本次新写的单设备实验适配，不是作者官方训练入口。

检查上游文件时未找到 RAMiT 的明确 LICENSE。这里的本地研究副本不因此自动受工作区根目录 MIT 许可证覆盖；公开再分发前应核实上游许可或取得授权。本次不推送该副本，不打包预训练权重或数据。
