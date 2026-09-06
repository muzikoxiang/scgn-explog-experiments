# Experiment status

Last synchronized: 2026-09-06 (Asia/Shanghai)

| Item | Status | Evidence |
|---|---|---|
| Official source snapshot | prepared | upstream `c77a5917759f09e66fbc7124220c5afc5ee221e5` |
| ExpLog architecture | prepared | static source review |
| Matched controls | prepared | static source review |
| Mac forward/backward | smoke-tested | six variants passed on CPU, PyTorch 2.10.0 |
| Windows CUDA pipeline | smoke-tested | A--F completed training, validation, save and reload |
| Official baseline reproduced | pending | needs datasets and checkpoint |
| 1,000-iteration ablations | complete | all A--F smoke runs finished |
| 10,000-iteration screening | complete | A, C, and D completed on seed 10 |
| ExpLog learning-rate ablation | complete | 5e-4, 2.5e-4, and 3.5e-4 completed |
| Multi-seed validation | in progress | seed 10 complete; seeds 20 and 30 pending |
| Full training | pending | run only after 10k screening |

Do not report prepared or smoke-tested rows as experimental results.

## Windows smoke results (2026-09-05)

These are early-convergence diagnostics on DIV2K100 after only 1,000 iterations,
not final model comparisons.

| Experiment | PSNR | SSIM | Total time | Status |
|---|---:|---:|---:|---|
| A SPAN baseline (`net_g_1000`) | 27.2765 | 0.7294 | 16:02 | passed |
| B GN at SPAB 3 | 27.7591 | 0.7499 | 19:40 | passed |
| C matched Conv at SPAB 3 | 27.5982 | **0.7612** | 17:20 | passed; best SSIM |
| D ExpLog at SPAB 3 | 28.0479 | 0.7581 | 20:40 | passed |
| E ExpLog at SPAB 2/4/6 | 27.0701 | 0.7223 | 19:04 | passed |
| F ExpLog at all SPABs | 27.6457 | 0.7515 | 19:54 | passed |

D is +0.7714 dB PSNR and +0.0287 SSIM over A at 1,000 iterations, but
took about 29% longer end-to-end. This supports trainability and a promising
early convergence trend only; it is not a final accuracy claim.

Against B, D gains 0.2888 dB PSNR and 0.0082 SSIM. Against the
parameter-matched C control, D gains 0.4497 dB PSNR but trails by 0.0031
SSIM. Therefore the current evidence does not support claiming that ExpLog is
best on every metric.

The insertion-count experiment is non-monotonic: D (one insertion) is better
than E (three insertions) and F (six insertions), while F is better than E.
The defensible conclusion is that ExpLog is sensitive to insertion location and
count, with SPAB 3 currently the strongest placement.

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
training. All six have now passed on Windows. E and F remain useful
insertion-count ablations but do not need immediate long runs.

## 10,000-iteration screening and LR ablation (seed 10)

| Experiment | LR | 10k PSNR | 10k SSIM | Best PSNR |
|---|---:|---:|---:|---:|
| A SPAN baseline | 5e-4 | 29.1695 | 0.8214 | 29.6280 @ 8k |
| C matched Conv | 5e-4 | 29.6596 | **0.8220** | 29.7372 @ 8k |
| D ExpLog SPAB 3 | 5e-4 | 29.3086 | 0.8046 | 29.5293 @ 6k |
| D ExpLog SPAB 3 | 2.5e-4 | 29.6211 | 0.8190 | 29.6211 @ 10k |
| D ExpLog SPAB 3 | 3.5e-4 | **29.7528** | 0.8214 | **29.7528 @ 10k** |

At 5e-4, D peaks at 6k and degrades afterward. Lowering the learning rate
improves late-stage stability; 3.5e-4 is the selected D setting. Against C at
10k it gains 0.0932 dB PSNR and trails by 0.0006 SSIM. Comparing each run's
best PSNR, the advantage is only 0.0156 dB. This is too small for a final
superiority claim from one seed.

Next, keep A at 5e-4, C at 5e-4, and D at 3.5e-4 and add seeds 20 and 30.
Use the 10k final checkpoint as the primary comparison for every method and
report mean plus standard deviation. Do not mix one method's best checkpoint
with another method's final checkpoint.

The original A seed-10 run saved checkpoints every 5,000 iterations, whereas
C and D saved every 2,000. The committed stage configurations preserve that
historical detail. New seed-20/30 runs should save every 2,000 iterations.
