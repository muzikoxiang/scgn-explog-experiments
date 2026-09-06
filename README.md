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

## Current follow-up: SPAN-ExpLog

The next-stage lightweight super-resolution experiment is prepared in
`SPAN-ExpLog/`. It transfers the best residual ExpLog idea to SPAN and includes
GN and parameter-matched convolution controls.

Current status (2026-09-05):

- six variants pass Mac CPU forward/backward, finite-value, and gradient smoke tests;
- all six Windows 1,000-iteration smoke experiments passed;
- D (single ExpLog at block 3) has the best PSNR: 28.0479;
- C (parameter-matched Conv) has the best SSIM: 0.7612;
- A/C/D 10k seed-10 screening and D learning-rate ablation are complete;
- D at 3.5e-4 reached 29.7528/0.8214; multi-seed validation is next;
- official pretrained-checkpoint reproduction and full training remain pending;
- Windows-validated progress is tracked in Git; datasets, checkpoints, and logs remain local.

See `SPAN-ExpLog/EXPERIMENT_STATUS.md` and `SPAN-ExpLog/WINDOWS_RUNBOOK.md`.
