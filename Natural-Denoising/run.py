"""Single-device, FP32 grayscale Gaussian-denoising experiments."""
import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity
from nets.factory import build

ROOT = Path(__file__).resolve().parent
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def images(roots):
    files = sorted({p.resolve() for root in roots for p in Path(root).rglob('*')
                    if p.is_file() and p.suffix.lower() in EXTENSIONS})
    if not files:
        raise ValueError('No images found; check data paths')
    return files


def gray(path):
    with Image.open(path) as image:
        array = np.array(ImageOps.exif_transpose(image).convert('L'), dtype=np.float32) / 255
    return torch.from_numpy(array.copy()).unsqueeze(0)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def batch(files, step, seed, size, count):
    # Stateless per-step sampling makes resume independent of worker/RNG queues.
    rng = random.Random(seed * 1000003 + step)
    generator = torch.Generator().manual_seed(seed * 1000003 + step)
    patches = []
    for _ in range(count):
        x = gray(files[rng.randrange(len(files))])
        if min(x.shape[-2:]) < size:
            raise ValueError(f'Training image smaller than patch {size}; reduce patch size')
        top = rng.randrange(x.shape[-2] - size + 1)
        left = rng.randrange(x.shape[-1] - size + 1)
        x = x[:, top:top+size, left:left+size]
        x = torch.rot90(x, rng.randrange(4), (-2, -1))
        if rng.random() < .5:
            x = x.flip(-1)
        patches.append(x)
    clean = torch.stack(patches)
    sigma = torch.rand((count, 1, 1, 1), generator=generator) * (50 / 255)
    noisy = clean + torch.randn(clean.shape, generator=generator) * sigma
    return noisy, clean


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def save_checkpoint(path, payload):
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def train(args):
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    files = images(cfg['train_roots'])
    hashes = [digest(p) for p in files]
    if len(set(hashes)) != len(hashes):
        raise ValueError('Duplicate training image content detected')
    identity = dict(config=cfg, backbone=args.backbone, variant=args.variant,
                    seed=args.seed, image_hashes=hashes)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    state_file = out / 'latest.pt'
    if not args.resume and any(out.iterdir()):
        raise ValueError('Output must be empty for a fresh run; use --resume or another directory')
    model = build(args.backbone, args.variant, args.seed).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['steps'], eta_min=cfg.get('min_lr', 1e-6))
    start = 0
    if args.resume:
        state = torch.load(state_file, map_location='cpu', weights_only=True)
        if state['identity'] != identity:
            raise ValueError('Resume config/data/model differs from saved run')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start = state['step']
    (out / 'manifest.json').write_text(json.dumps(identity, indent=2), encoding='utf-8')
    model.train()
    end = min(cfg['steps'], args.stop_after or cfg['steps'])
    for step in range(start, end):
        noisy, clean = batch(files, step, args.seed, cfg['patch'], cfg['batch'])
        optimizer.zero_grad(set_to_none=True)
        prediction = model(noisy.to(args.device))
        loss = (prediction - clean.to(args.device)).abs().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at step {step+1}')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if (step + 1) % cfg.get('log_every', 50) == 0 or step == start:
            record = dict(step=step+1, loss=loss.item(), grad_norm=float(norm),
                          lr=optimizer.param_groups[0]['lr'])
            with (out / 'train.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record) + '\n')
            print(record, flush=True)
        if (step + 1) % cfg['save_every'] == 0 or step + 1 == end:
            payload = dict(identity=identity, step=step+1, model=model.state_dict(),
                           optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
            save_checkpoint(state_file, payload)
            save_checkpoint(out / f'iter_{step+1:06d}.pt', payload)


def evaluate(args):
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    ident = state['identity']
    model = build(ident['backbone'], ident['variant'], ident['seed']).to(args.device)
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    files = images([args.test_root])
    rows = []
    with torch.inference_mode():
        for path in files:
            sha = digest(path)
            if sha in ident['image_hashes']:
                raise ValueError(f'Train/test leakage: {path}')
            clean = gray(path).unsqueeze(0)
            for sigma in args.sigmas:
                seed = int(hashlib.sha256(f'{sha}:{sigma}:2026'.encode()).hexdigest()[:15], 16)
                noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(seed))
                pred = model((clean + noise * (sigma / 255)).to(args.device)).cpu().clamp(0, 1)
                if not torch.isfinite(pred).all():
                    raise FloatingPointError(f'Nonfinite prediction: {path}, sigma={sigma}')
                target, result = clean[0, 0].numpy(), pred[0, 0].numpy()
                mse = float(np.mean((target-result)**2))
                psnr = -10 * np.log10(max(mse, 1e-15))
                ssim = structural_similarity(target, result, data_range=1., gaussian_weights=True,
                                             sigma=1.5, use_sample_covariance=False)
                rows.append(dict(image=path.name, sha256=sha, sigma=sigma, psnr=psnr, ssim=ssim))
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / 'per_image.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {str(s): {k: float(np.mean([r[k] for r in rows if r['sigma']==s]))
                       for k in ('psnr', 'ssim')} for s in args.sigmas}
    report = dict(backbone=ident['backbone'], variant=ident['variant'], seed=ident['seed'],
                  step=state['step'], checkpoint_sha256=digest(Path(args.checkpoint)),
                  dataset=str(Path(args.test_root).resolve()), images=len(files), metrics=summary,
                  protocol='PIL L grayscale; fixed content-seeded AWGN; no input clipping; output clamp; full image PSNR; gaussian SSIM')
    (destination / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


def profile(args):
    report = []
    for backbone in ('scgn', 'ramit'):
        for variant in ('baseline', 'matched', 'explog'):
            model = build(backbone, variant).to(args.device).eval()
            x = torch.zeros(1, 1, args.size, args.size, device=args.device)
            conv_macs = [0]
            def hook(layer, inputs, output):
                conv_macs[0] += output.numel() * (layer.in_channels // layer.groups) * np.prod(layer.kernel_size)
            handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
            with torch.inference_mode():
                model(x)
                for h in handles:
                    h.remove()
                for _ in range(args.warmup):
                    model(x)
                sync(args.device)
                if args.device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(args.device)
                times = []
                for _ in range(args.repeats):
                    sync(args.device)
                    begin = time.perf_counter()
                    model(x)
                    sync(args.device)
                    times.append((time.perf_counter()-begin)*1000)
            report.append(dict(backbone=backbone, variant=variant,
                               parameters=sum(p.numel() for p in model.parameters()),
                               conv_macs_only=int(conv_macs[0]), mean_ms=float(np.mean(times)),
                               median_ms=float(np.median(times)),
                               peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device)
                               if args.device.type=='cuda' else None))
            del model, x
            if args.device.type == 'cuda':
                torch.cuda.empty_cache()
    payload = dict(input=[1,1,args.size,args.size], dtype='float32', device=str(args.device),
                   torch=torch.__version__, warmup=args.warmup, repeats=args.repeats,
                   note='Conv MACs only; excludes attention matmuls, FFT, exp/log, normalization and other elementwise ops. NOT total FLOPs; cross-backbone budgets not matched.', models=report)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cuda','cpu'])
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--config', default=str(ROOT/'configs/screen.json'))
    t.add_argument('--backbone', required=True, choices=['scgn','ramit'])
    t.add_argument('--variant', required=True, choices=['baseline','matched','explog'])
    t.add_argument('--seed', type=int, default=10)
    t.add_argument('--output', required=True)
    t.add_argument('--resume', action='store_true')
    t.add_argument('--stop-after', type=int, help='Stop at this step without changing schedule; useful for resume checks')
    e = sub.add_parser('evaluate')
    e.add_argument('--checkpoint', required=True)
    e.add_argument('--test-root', required=True)
    e.add_argument('--sigmas', type=int, nargs='+', default=[15,25,50])
    e.add_argument('--output', required=True)
    p = sub.add_parser('profile')
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--repeats', type=int, default=30)
    p.add_argument('--output', default='profile.json')
    args = parser.parse_args()
    args.device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    {'train':train, 'evaluate':evaluate, 'profile':profile}[args.command](args)


if __name__ == '__main__':
    main()
