# SCGN ExpLog experiment reconstruction

Reconstructed experiment variants for the SCGN TEM denoising model. The baseline
comes from [HeasonLee/SCGN](https://github.com/HeasonLee/SCGN); ExpLog variants are
recovered from surviving source, checkpoints, result PNGs, a progress report, and
the original experiment conversation.

## Variants

```bash
python inspect_model.py baseline
python inspect_model.py exp5_spatial_main
python inspect_model.py exp7_frequency
python inspect_model.py exp8_full_sdgw
```

Available names are defined in `models/scgn_variants.py`. Detailed provenance and
historical metrics are in `EXPERIMENTS.md`.

## Re-evaluate saved results

```bash
python evaluate_results.py \
  --gt tem_test_data4/gt \
  --results convLast_std_tem_data4_exp8_result
```

The evaluator intentionally matches the surviving script's first-channel PSNR and
thresholded IoU behavior. SSIM should be evaluated with `pytorch-msssim`, matching
the original training script.

## Data and weights

Datasets, checkpoints, and generated PNGs are excluded from Git by default. Follow
the upstream SCGN repository for dataset setup, then keep local artifacts in the
historical directory names if checkpoint compatibility is required.

## Provenance

This is a research reconstruction, not an official SCGN release. Some early variant
source files were missing, so inferred implementations are labeled explicitly.

