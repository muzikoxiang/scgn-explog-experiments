"""Synthetic functional tests; no research accuracy claims."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from nets.factory import build

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parent


def main():
    for backbone in ('scgn', 'ramit'):
        c, d = build(backbone, 'matched'), build(backbone, 'explog')
        assert sum(p.numel() for p in c.parameters()) == sum(p.numel() for p in d.parameters())
        assert all(torch.equal(c.state_dict()[k], v) for k, v in d.state_dict().items())
        for variant in ('baseline', 'matched', 'explog'):
            model = build(backbone, variant)
            x = torch.randn(1, 1, 33, 35)
            y = model(x)
            assert y.shape == x.shape and torch.isfinite(y).all()
            y.square().mean().backward()
            grads = [p.grad for p in model.parameters() if p.requires_grad]
            assert all(g is not None and torch.isfinite(g).all() for g in grads)
            print(backbone, variant, sum(p.numel() for p in model.parameters()), 'shape/backward PASS', flush=True)
    with tempfile.TemporaryDirectory(prefix='denoise-test-') as td:
        td = Path(td)
        train_dir, test_dir = td/'train', td/'test'
        train_dir.mkdir()
        test_dir.mkdir()
        rng = np.random.default_rng(12)
        for directory in (train_dir, test_dir):
            Image.fromarray(rng.integers(0,256,(64,64),dtype=np.uint8)).save(directory/'fixture.png')
        cfg = dict(train_roots=[str(train_dir)], steps=2, patch=32, batch=1,
                   lr=.0002, min_lr=1e-6, save_every=1, log_every=1)
        config = td/'config.json'
        config.write_text(json.dumps(cfg))
        def run(*args, success=True):
            result = subprocess.run([sys.executable, str(ROOT/'run.py'), '--device','cpu',*args],
                                    capture_output=True, text=True)
            if success and result.returncode:
                raise AssertionError(result.stdout + result.stderr)
            if not success:
                assert result.returncode != 0
            return result
        for backbone in ('scgn','ramit'):
            full, resumed = td/f'{backbone}-full', td/f'{backbone}-resume'
            common = ['train','--config',str(config),'--backbone',backbone,'--variant','explog']
            run(*common,'--output',str(full))
            run(*common,'--output',str(resumed),'--stop-after','1')
            run(*common,'--output',str(resumed),'--resume')
            a = torch.load(full/'latest.pt', weights_only=True)
            b = torch.load(resumed/'latest.pt', weights_only=True)
            assert all(torch.equal(v,b['model'][k]) for k,v in a['model'].items())
            run('evaluate','--checkpoint',str(full/'latest.pt'),'--test-root',str(test_dir),'--output',str(td/'eval'))
            assert len(json.loads((td/'eval/metrics.json').read_text())['metrics']) == 3
            run('evaluate','--checkpoint',str(full/'latest.pt'),'--test-root',str(train_dir),'--output',str(td/'leak'),success=False)
            run(*common,'--output',str(full),success=False)
            print(backbone, 'resume exact / evaluate / leakage / overwrite protection PASS', flush=True)


if __name__ == '__main__':
    main()
