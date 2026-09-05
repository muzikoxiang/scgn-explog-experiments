# Experiment status

Last synchronized: 2026-09-05 (Asia/Shanghai)

| Item | Status | Evidence |
|---|---|---|
| Official source snapshot | prepared | upstream `c77a5917759f09e66fbc7124220c5afc5ee221e5` |
| ExpLog architecture | prepared | static source review |
| Matched controls | prepared | static source review |
| Mac forward/backward | smoke-tested | six variants passed on CPU, PyTorch 2.10.0 |
| Windows CUDA pipeline | smoke-tested | A and D completed training, validation, save and reload |
| Official baseline reproduced | pending | needs datasets and checkpoint |
| Full ablations | pending | needs Windows GPU |

Do not report prepared or smoke-tested rows as experimental results.

## Windows smoke results (2026-09-05)

These are early-convergence diagnostics on DIV2K100 after only 1,000 iterations,
not final model comparisons.

| Experiment | PSNR | SSIM | Total time | Status |
|---|---:|---:|---:|---|
| A SPAN baseline (`net_g_1000`) | 27.2765 | 0.7294 | 16:02 | passed |
| B GN at SPAB 3 | pending | pending | pending | run next |
| C matched Conv at SPAB 3 | pending | pending | pending | run after B |
| D ExpLog at SPAB 3 | 28.0479 | 0.7581 | 20:40 | passed |
| E ExpLog at SPAB 2/4/6 | pending | pending | pending | defer |
| F ExpLog at all SPABs | pending | pending | pending | defer |

D is +0.7714 dB PSNR and +0.0287 SSIM over A at 1,000 iterations, but
took about 29% longer end-to-end. This supports trainability and a promising
early convergence trend only; it is not a final accuracy claim.

The initial A training validation reported 11.6924/0.0388 because it evaluated
the lagging EMA network (`ema_decay: 0.999`). Reloading and testing
`net_g_1000.pth` produced the valid 27.2765/0.7294 result. Generated smoke
configs now force `ema_decay: 0`; restore 0.999 only for full training.

## Prepared ablations

- A: untouched SPAN baseline
- B: normalization-only control at SPAB 3
- C: parameter-matched ordinary-convolution control at SPAB 3
- D: ExpLog at SPAB 3
- E: ExpLog at SPAB 2, 4, and 6
- F: ExpLog at all six SPABs

The generated configurations use x4 super-resolution and 1,000-iteration smoke
training. Full-run iteration counts must be restored only after the official
baseline and smoke runs pass on Windows.
