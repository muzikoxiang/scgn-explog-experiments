"""Reconstructed SCGN/ExpLog experiment variants.

The baseline follows HeasonLee/SCGN.  ExpLog variants are reconstructed from
the surviving exp8 source, checkpoints/results, and the user's experiment chat.
See EXPERIMENTS.md for the evidence level of each mapping.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Variant:
    blocks: int
    local: str
    outer_norm: str
    spectral: bool = False
    explog_attention: bool = False


VARIANTS = {
    "baseline": Variant(8, "conv", "bn"),
    "exp1_raw": Variant(8, "raw_abs", "bn"),
    "exp2_softplus": Variant(8, "raw_softplus", "bn"),
    "exp3_layernorm": Variant(8, "layernorm", "bn"),
    "exp5_spatial_main": Variant(4, "residual_gn4_x2", "gn"),
    "exp7_frequency": Variant(4, "residual_gn4_x2", "gn", spectral=True),
    "exp8_full_sdgw": Variant(
        4, "residual_gn4_x2", "gn", explog_attention=True
    ),
}


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x))


class FourierUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels * 2 + 2, out_channels * 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)
        self.attn = ChannelAttention(out_channels * 2)
        self.conv2 = nn.Conv2d(out_channels * 2, out_channels * 2, 1, bias=False)

    def forward(self, x):
        batch = x.shape[0]
        ffted = torch.fft.rfftn(x, dim=(-2, -1), norm="ortho")
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1) + ffted.size()[3:])
        height, width = ffted.shape[-2:]
        vertical = torch.linspace(0, 1, height, device=x.device, dtype=x.dtype)
        horizontal = torch.linspace(0, 1, width, device=x.device, dtype=x.dtype)
        vertical = vertical[None, None, :, None].expand(batch, 1, height, width)
        horizontal = horizontal[None, None, None, :].expand(batch, 1, height, width)
        ffted = torch.cat((vertical, horizontal, ffted), dim=1)
        ffted = self.relu(self.bn(self.conv1(ffted)))
        ffted = self.conv2(ffted * self.attn(ffted))
        ffted = ffted.view((batch, -1, 2) + ffted.size()[2:])
        ffted = ffted.permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        return torch.fft.irfftn(ffted, s=x.shape[-2:], dim=(-2, -1), norm="ortho")


class ExpLogLite(nn.Module):
    def __init__(self, channels: int, groups: int = 4, alpha: float = 0.1):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 2, 1)
        self.norm = nn.GroupNorm(groups, channels)
        self.alpha = alpha

    def forward(self, x):
        x1, x2 = torch.chunk(self.conv(x), 2, dim=1)
        nonlinear = torch.exp(torch.clamp(x1, -3, 3)) - torch.log(x2.abs() + 1e-6)
        return x + self.alpha * self.norm(nonlinear)


class SpectralTransform(nn.Module):
    def __init__(self, channels: int, use_explog_lite: bool = False):
        super().__init__()
        half = channels // 2
        self.pre = ExpLogLite(half) if use_explog_lite else nn.Conv2d(half, half, 3, 1, 1)
        self.fourier = FourierUnit(half, half)
        # Kept as Conv: replacing it with a same-width ExpLogLite changes 64 -> 32.
        self.fuse = nn.Conv2d(channels, half, 3, 1, 1)

    def forward(self, x):
        transformed = self.fourier(self.pre(x))
        return self.fuse(torch.cat((x, transformed), dim=1))


class WindowStd(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, eps: float = 1e-5):
        super().__init__()
        self.kernel_size = kernel_size
        self.eps = eps
        kernel = torch.ones(channels, 1, kernel_size, kernel_size) / (kernel_size**2)
        self.register_buffer("mean_kernel", kernel)

    def forward(self, x):
        pad = self.kernel_size // 2
        padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        mean = F.conv2d(padded, self.mean_kernel, groups=x.shape[1])
        squared = F.pad(x.square(), (pad, pad, pad, pad), mode="reflect")
        mean_squared = F.conv2d(squared, self.mean_kernel, groups=x.shape[1])
        return torch.sqrt(torch.clamp(mean_squared - mean.square(), min=self.eps))


class ExpLogConv(nn.Module):
    def __init__(self, channels: int, domain: str, norm_groups: int | None = None):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 2, 3, 1, 1)
        self.domain = domain
        self.norm = nn.Identity() if norm_groups is None else nn.GroupNorm(norm_groups, channels)

    def forward(self, x):
        x1, x2 = torch.chunk(self.conv(x), 2, dim=1)
        x1 = torch.clamp(x1, -5, 5)
        if self.domain == "softplus":
            x2 = F.softplus(x2) + 1e-6
        else:
            x2 = x2.abs() + 1e-6
        return self.norm(torch.exp(x1) - torch.log(x2))


class ExpLogBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 4, alpha: float = 0.1):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 2, 3, 1, 1)
        self.norm = nn.GroupNorm(groups, channels)
        self.alpha = alpha

    def forward(self, x):
        x1, x2 = torch.chunk(self.conv(x), 2, dim=1)
        nonlinear = torch.exp(torch.clamp(x1, -5, 5)) - torch.log(x2.abs() + 1e-6)
        return x + self.alpha * self.norm(nonlinear)


class ExpLogAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 2, 1)
        self.norm = nn.GroupNorm(4, channels)

    def forward(self, x):
        x1, x2 = torch.chunk(self.conv(x), 2, dim=1)
        nonlinear = torch.exp(torch.clamp(x1, -3, 3)) - torch.log(x2.abs() + 1e-6)
        return torch.sigmoid(self.norm(nonlinear))


def _local_extractor(channels: int, name: str) -> nn.Module:
    if name == "conv":
        return nn.Conv2d(channels, channels, 3, 1, 1)
    if name == "raw_abs":
        return ExpLogConv(channels, "abs")
    if name == "raw_softplus":
        return ExpLogConv(channels, "softplus")
    if name == "layernorm":
        return ExpLogConv(channels, "softplus", norm_groups=1)
    if name == "residual_gn4_x2":
        return nn.Sequential(ExpLogBlock(channels), ExpLogBlock(channels))
    raise ValueError(f"Unknown local extractor: {name}")


class FFC(nn.Module):
    def __init__(self, channels: int, variant: Variant):
        super().__init__()
        half = channels // 2
        self.local = _local_extractor(half, variant.local)
        self.std = WindowStd(half)
        self.attention = (
            ExpLogAttention(half)
            if variant.explog_attention
            else nn.Sequential(nn.Conv2d(half, half, 1, bias=False), nn.Sigmoid())
        )
        self.global_branch = SpectralTransform(channels, variant.spectral)

    def forward(self, inputs):
        local, global_ = inputs
        feature = self.local(local)
        weight = self.attention(self.std(local))
        return feature * weight, self.global_branch(global_)


class FFCNormAct(nn.Module):
    def __init__(self, channels: int, variant: Variant):
        super().__init__()
        half = channels // 2
        self.ffc = FFC(channels, variant)
        norm = nn.BatchNorm2d if variant.outer_norm == "bn" else lambda c: nn.GroupNorm(4, c)
        self.local_norm = norm(half)
        self.global_norm = norm(half)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        local, global_ = self.ffc(inputs)
        return self.relu(self.local_norm(local)), self.relu(self.global_norm(global_))


class FFCResidualBlock(nn.Module):
    def __init__(self, channels: int, variant: Variant):
        super().__init__()
        self.first = FFCNormAct(channels, variant)
        self.second = FFCNormAct(channels, variant)
        self.fuse = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        local, global_ = torch.chunk(x, 2, dim=1)
        identity_local, identity_global = local, global_
        local, global_ = self.first((local, global_))
        local, global_ = self.second((local, global_))
        local = identity_local + local
        global_ = identity_global + global_
        return self.fuse(torch.cat((local, global_), dim=1))


class SCGN(nn.Module):
    def __init__(self, variant: str = "baseline", channels: int = 64):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
        config = VARIANTS[variant]
        self.variant = variant
        self.head = nn.Conv2d(1, channels, 3, 1, 1)
        self.body = nn.Sequential(
            *(FFCResidualBlock(channels, config) for _ in range(config.blocks))
        )
        self.tail = nn.Conv2d(channels, 1, 3, 1, 1)

    def forward(self, x):
        x = self.head(x)
        return self.tail(self.body(x) + x)


def build_model(variant: str = "baseline") -> SCGN:
    return SCGN(variant=variant)

