"""Recompute PSNR and binary IoU from saved PNG results."""

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image


def first_channel(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path), dtype=np.float64)
    return image[..., 0] if image.ndim == 3 else image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=127.5)
    args = parser.parse_args()

    psnrs, ious = [], []
    for gt_path in sorted(args.gt.glob("*.png")):
        result_path = args.results / gt_path.name
        gt, result = first_channel(gt_path), first_channel(result_path)
        mse = np.mean(((gt - result) / 255.0) ** 2)
        psnrs.append(-10.0 * math.log10(mse))
        gt_mask, result_mask = gt > args.threshold, result > args.threshold
        union = np.logical_or(gt_mask, result_mask).sum()
        ious.append(0.0 if union == 0 else np.logical_and(gt_mask, result_mask).sum() / union)

    print(f"images={len(psnrs)} PSNR={np.mean(psnrs):.4f} IoU={np.mean(ious):.4f}")


if __name__ == "__main__":
    main()

