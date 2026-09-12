# 自然图像去噪 SOTA 主干筛选

更新日期：2026-09-09

## 目标

按照师兄的建议，在自然图像去噪任务上分别验证 SCGN 和一个轻量先进网络，判断 ExpLog 的正向结果是来自显微镜噪声、SCGN 结构，还是能够推广到其他去噪网络。

筛选条件：

1. 直接支持自然图像去噪，输入和输出尺寸一致。
2. 模型较小，能够在 RTX 4060 Laptop GPU 上完成多组对照。
3. 公开训练代码、测试代码和预训练权重。
4. 能定位清楚的局部特征提取或特征混合模块，便于设置 ExpLog 和参数匹配普通卷积对照。
5. 有正式论文和标准测试集结果，不只依赖作者自行报告。

## 筛选结论

主选模型为 **RAMiT（Reciprocal Attention Mixing Transformer）**。

代码准备阶段补充：RAMiT 是可操作的轻量先进基线候选，尚不能称为 2026 最新 SOTA。当前独立实现位于 `Natural-Denoising/`，Windows 实验尚待验证。

- 论文：Reciprocal Attention Mixing Transformer for Lightweight Image Restoration。
- 发表：CVPR 2024 Workshop，NTIRE。
- 任务：彩色去噪、灰度去噪、超分辨率、低光增强和去雨。
- 灰度去噪参数量：约 0.932M；彩色去噪参数量：约 0.935M。
- 论文报告的去噪计算量：620.8G Mult-Adds，口径为 1280×720 图像；按像素数折算到 256×256 约为 44.1G，最终应使用与 SCGN 相同的脚本重新测量。
- 官方仓库包含完整训练与测试入口，以及彩色和灰度去噪预训练权重。
- 官方自然图像去噪训练集为 DFBW，即 DIV2K、Flickr2K、BSD500 和 WED。
- 彩色测试集包括 CBSD68、Kodak24、McMaster 和 Urban100；灰度测试集包括 Set12、BSD68 和 Urban100。
- 官方设置支持盲高斯去噪，噪声标准差范围为 0–50，并报告 σ=15、25、50 的结果。

RAMiT 论文在约 1M 参数的统一轻量对照中超过 SwinIR-light、Restormer-light、CAT-light、ART-light 和 NGswin。以 σ=25 为例：

| 任务 | 测试集 | RAMiT PSNR / SSIM |
|---|---|---|
| 彩色去噪 | CBSD68 | 31.59 / 0.8902 |
| 彩色去噪 | Kodak24 | 32.76 / 0.8887 |
| 彩色去噪 | McMaster | 33.02 / 0.9008 |
| 彩色去噪 | Urban100 | 32.36 / 0.9244 |
| 灰度去噪 | Set12 | 30.79 / 0.8694 |
| 灰度去噪 | BSD68 | 29.37 / 0.8346 |
| 灰度去噪 | Urban100 | 30.71 / 0.9013 |

论文与代码：

- [CVPR 论文](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Choi_Reciprocal_Attention_Mixing_Transformer_for_Lightweight_Image_Restoration_CVPRW_2024_paper.pdf)
- [官方 GitHub](https://github.com/rami0205/RAMiT)

## 为什么 RAMiT 符合本实验

RAMiT 与 SCGN 的规模处于同一量级。历史汇报中的原始 SCGN 为 1.64M 参数、83.7G FLOPs，统计输入为 1×256×256；RAMiT 灰度去噪版约 0.932M 参数，按同分辨率估算约 44.1G Mult-Adds。两者都属于小模型，可以在相同自然图像去噪协议下进行多随机种子实验。

RAMiT 的 `DRAMiTransformer` 中包含明确的局部特征混合模块 `MobiVari`，并带有残差连接。它与 SCGN 中被修改的局部特征提取部分具有可比的功能位置，适合构造以下三组：

| 组别 | SCGN | RAMiT |
|---|---|---|
| A 原始主干 | 原始 SCGN | 原始 RAMiT |
| C 参数匹配对照 | 普通卷积局部模块 | 普通卷积局部模块 |
| D ExpLog | ExpLog 局部模块 | ExpLog 局部模块 |

第一阶段先保证同一主干内部 C/D 参数匹配，使用完全一致的数据、损失、训练预算和随机种子。但这不代替师兄提出的跨主干等参数量/计算量条件：后续还要构造并验证预算匹配版本。原始宽度作为参考保留。当前代码仅统计卷积 MACs，不包含 FFT、注意力矩阵乘及逐元素算子，不能称为总 FLOPs；历史 SCGN FLOPs 与论文 RAMiT Mult-Adds 不能直接当作同口径数据比较。

## 推荐的第一阶段实验

先做灰度自然图像盲高斯去噪，因为它与原始 SCGN 的单通道输入输出一致，可以减少任务迁移带来的额外结构改动。

| 项目 | 设置 |
|---|---|
| 训练数据 | DFBW 的灰度版本 |
| 噪声 | 训练时 σ∈[0,50]；测试 σ=15、25、50 |
| 测试集 | Set12、BSD68、Urban100 |
| 主干 | SCGN、RAMiT-GDN |
| 每个主干的版本 | Baseline、Matched Conv、ExpLog |
| 初筛 | 单 seed、短预算，检查稳定性和早期趋势 |
| 正式比较 | 3 seeds、统一最终 checkpoint |
| 指标 | PSNR、SSIM、参数量、FLOPs、延迟、峰值显存 |

如果 ExpLog 在灰度自然图像上出现稳定优势，再扩展到彩色去噪；如果两个主干都没有超过参数匹配卷积，则没有必要立即投入其他恢复任务或长训练。

## 候选模型比较

| 模型 | 年份与出处 | 规模与任务 | 代码状态 | 结论 |
|---|---|---|---|---|
| RAMiT | CVPRW 2024 NTIRE | 约 0.93M；彩色/灰度去噪及多种恢复任务 | 训练、测试、权重完整 | **主选** |
| MTSNet | 2025 论文 | 约 0.46–1.49M；去噪、去雨、去模糊 | BasicSR 代码、配置和权重完整 | 备选；更新但论文影响与复现成熟度较弱 |
| DnLUT | CVPR 2025 | 超轻量彩色去噪 | 仓库仍标注训练与 LUT 转换代码待补齐 | 不选；结构特殊且不便做同构 ExpLog 对照 |
| EMTS-MHGNet | CVPRW 2026 NTIRE | 0.76M/3.46M；主任务为去模糊，附带去噪实验 | 未找到完整官方训练仓库 | 不选；任务与复现条件不匹配 |
| Compacter | ACM MM 2024 | 约 0.407M；多种轻量恢复任务 | 未找到公开完整官方代码 | 不选；无法立即复现 |
| NAFNet | ECCV 2022 | 去噪和去模糊效果强 | 代码成熟 | 备选基准；常用去噪版明显大于当前实验预算，且发布时间较早 |

## 结果判定

| 观察结果 | 支持的解释 |
|---|---|
| ExpLog 在 SCGN 和 RAMiT 上都超过 Matched Conv | ExpLog 具有跨网络的去噪通用性 |
| 只在 SCGN 上超过 Matched Conv | ExpLog 更可能依赖 SCGN 的局部统计结构 |
| 只在显微镜数据上有效 | ExpLog 更可能适配 HRTEM 噪声或纹理分布 |
| 两个自然图像主干都不超过 Matched Conv | 早期 SCGN 提升可能来自组合改动，不能归因于 ExpLog |

## 下一步实现位置

RAMiT 的第一处实现应放在 `DRAMiTransformer.mobivari` 对应的局部混合位置。先只替换 Stage 1 的一个块，建立 Baseline、Matched Conv 和 ExpLog 三个参数匹配版本；通过形状、有限值、反向传播和参数量检查后，再生成 Windows 训练配置。
