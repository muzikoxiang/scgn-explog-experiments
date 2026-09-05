"""CPU/CUDA smoke test for all prepared SPAN-ExpLog ablations."""

import argparse
import importlib.util
from pathlib import Path
import sys
import types

import torch


def load_architecture_without_basicsr_side_effects():
    """Load span_arch directly; the upstream package eagerly imports optional modules."""

    class NoopRegistry:
        def register(self):
            return lambda cls: cls

    basicsr = types.ModuleType('basicsr')
    utils = types.ModuleType('basicsr.utils')
    registry = types.ModuleType('basicsr.utils.registry')
    registry.ARCH_REGISTRY = NoopRegistry()
    sys.modules.update({
        'basicsr': basicsr,
        'basicsr.utils': utils,
        'basicsr.utils.registry': registry,
    })
    path = Path(__file__).resolve().parents[1] / 'basicsr' / 'archs' / 'span_arch.py'
    spec = importlib.util.spec_from_file_location('_span_arch_smoke', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SPAN, module.SPANExpLog


SPAN, SPANExpLog = load_architecture_without_basicsr_side_effects()


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def assert_finite_gradients(model):
    missing = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all()):
            missing.append(name)
    if missing:
        raise RuntimeError(f'Missing or non-finite gradients: {missing[:8]}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--size', type=int, default=24)
    args = parser.parse_args()
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    if device == 'auto':
        device = 'cpu'

    variants = {
        'baseline': SPAN(3, 3, feature_channels=48, upscale=4),
        'gn_b3': SPANExpLog(3, 3, feature_channels=48, upscale=4, enhance_blocks=[3], enhancer='gn'),
        'matched_b3': SPANExpLog(3, 3, feature_channels=48, upscale=4, enhance_blocks=[3], enhancer='matched_conv'),
        'explog_b3': SPANExpLog(3, 3, feature_channels=48, upscale=4, enhance_blocks=[3], enhancer='explog'),
        'explog_b246': SPANExpLog(3, 3, feature_channels=48, upscale=4, enhance_blocks=[2, 4, 6], enhancer='explog'),
        'explog_all': SPANExpLog(3, 3, feature_channels=48, upscale=4, enhance_blocks=[1, 2, 3, 4, 5, 6], enhancer='explog'),
    }
    sample = torch.rand(1, 3, args.size, args.size, device=device)
    for name, model in variants.items():
        model = model.to(device).train()
        output = model(sample)
        expected = (1, 3, args.size * 4, args.size * 4)
        if tuple(output.shape) != expected:
            raise RuntimeError(f'{name}: shape {tuple(output.shape)} != {expected}')
        if not torch.isfinite(output).all():
            raise RuntimeError(f'{name}: output contains NaN/Inf')
        output.square().mean().backward()
        assert_finite_gradients(model)
        print(f'{name:16s} params={count_parameters(model):,} output={tuple(output.shape)} finite=yes gradients=yes')

    print(f'PASS device={device} torch={torch.__version__} cuda={torch.version.cuda}')


if __name__ == '__main__':
    main()
