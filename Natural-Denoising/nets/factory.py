import torch
from torch import nn
from .scgn import SCGN, ExpLogBlock
from .ramit import RAMiT


class MatchedBlock(ExpLogBlock):
    """Same weights, GN and residual scaling as ExpLog, without exp/log."""
    def forward(self, x):
        first, second = self.conv(x).chunk(2, dim=1)
        return x + self.alpha * self.norm(first - second)


def build(backbone, variant, seed=10):
    # Separate initialization streams keep unchanged backbone weights identical.
    torch.manual_seed(seed)
    if backbone == 'scgn':
        model = SCGN('baseline')
        parent, key, channels = model.body[0].first.ffc, 'local', 32
    elif backbone == 'ramit':
        model = RAMiT(in_chans=1, target_mode='light_graydn', mv_ver=2,
                      helper=False, img_norm=True)
        parent, key, channels = model.stage1.blocks.block0, 'mobivari', 64
    else:
        raise ValueError(backbone)
    if variant != 'baseline':
        if variant not in ('matched', 'explog'):
            raise ValueError(variant)
        torch.manual_seed(seed + 10000)
        block = ExpLogBlock if variant == 'explog' else MatchedBlock
        setattr(parent, key, nn.Sequential(block(channels), block(channels)))
    return model
