# Experiment reconstruction notes

This repository separates facts preserved by files/results from variants inferred
from the earlier experiment conversation. It does not claim that every historical
source file survived unchanged.

| Variant | Historical label | Blocks | Reconstructed change | PSNR | SSIM | IoU | Evidence |
|---|---|---:|---|---:|---:|---:|---|
| `baseline` | Original SCGN | 8 | Upstream architecture | 26.5507 | 0.9717 | 0.7330* | Upstream source + saved results |
| `exp1_raw` | exp1 | 8 | First raw Exp–Log local replacement | 26.4775 | — | 0.7313 | Result/weight survived; exact source inferred |
| `exp2_softplus` | exp2 | 8 | Single local Exp–Log, clamped exp and softplus log domain | 26.8479 | 0.9740 | 0.7404 | Chat code/result + saved result |
| `exp3_layernorm` | exp3 | 8 | exp2 + GroupNorm(1,C), used as CNN-style LayerNorm | 26.6060 | 0.9726 | 0.7337 | Chat code/result + saved result |
| `exp5_spatial_main` | ExpLog backbone | 4 | Two residual ExpLogBlocks, GN(4), original spectral/weight path | 26.9676 | 0.9746 | 0.7403 | Chat + report + saved result |
| `exp7_frequency` | Frequency ExpLog | 4 | exp5 + ExpLogLite in spectral pre-convolution | 24.7578 | 0.9488 | 0.6765 | Chat + saved result; historical source inferred |
| `exp8_full_sdgw` | SDGW full replacement | 4 | exp5 + ExpLogAttention after WindowStd | 26.8207 | 0.9735 | 0.7354 | Surviving source + saved result |

`*` The old summary reports baseline IoU 0.7317. Recomputing the current 100 saved
PNGs with the surviving threshold implementation gives 0.7330. Both values are
retained as a provenance warning until the original evaluation run is located.

The `exp4` and `exp6` identifiers are intentionally absent because no matching
checkpoint/result/source was present in the supplied folder.

## Important comparison caveat

The main variants change multiple factors together: block count (8 to 4), local
extractor, outer BN/GN, and sometimes spectral or attention components. Therefore
the existing table demonstrates performance of complete configurations; it does
not isolate ExpLog's causal contribution by itself.

