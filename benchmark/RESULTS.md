# Results log (GTX 1080 Ti, Windows 11, PyTorch cu126, CUDA 12.6, VS 2022 Build Tools)

## Correctness on GPU
- test_HF: logits max abs diff 3.86e-05, attention 5.96e-07, MLP 2.19e-05 (all pass, gate 1e-3)
- test_gate2: 500 tokens, cached == uncached token for token. Naive 59.34 s, cached 9.79 s

## Baseline throughput (benchmark.benchmark, FP32)
| Tokens | Naive tok/s | Cached tok/s | Speedup |
|---|---|---|---|
| 30 | 29.71 | 54.60 | 1.8x |
| 150 | 23.99 | 56.11 | 2.3x |
| 300 | 16.38 | 46.63 | 2.8x |

## FP32 vs current INT8 path (bench_int8_speed, 300 tokens cached)
- FP32 49.2 tok/s, INT8 (dequantize every call) 28.3 tok/s

## Profile (profile_decode, FP32 cached, 20 tokens)
- 1102 GPU operations per token
- GPU busy 9.53 ms per token vs 20.33 ms real: GPU idle ~half the time (CPU launch bound)
- Matrix-vector kernels ~2/3 of GPU time

## INT8 GEMV kernel (cuda/int8_gemv.cu, median of 9 x 200 launches, us)
| Version | gate/up 4864x896 | down 896x4864 | q/o 896x896 |
|---|---|---|---|
| v1 block per row | 73.37 | 37.81 | 14.16 |
| v2 warp per row + shuffle | 31.19 | 38.08 | 7.12 |
| v3 char4/float4 loads (final) | 18.76 | 17.50 | 6.21 |

Experiments on top of v3 (not kept):
- 16-byte loads: 23.10 / 18.85 / 6.92, slower (scattered x reads)
- Same, with x reads removed (ablation): 17.08 / 16.39 / 5.89, proves the x reads caused the slowdown
- x staged in shared memory: 18.03 / 18.91 / 5.67, mixed (copy + sync delays each block)
- #pragma unroll 4: 18.44 / 17.28 / 6.25, no change (compiler already unrolls)

## Kernel vs cuBLAS FP32 (same run)
| Shape | INT8 kernel | cuBLAS FP32 | Speedup |
|---|---|---|---|
| gate/up | 21.48 | 73.99 | 3.44x |
| down | 19.13 | 66.39 | 3.47x |
| q/o | 6.45 | 12.22 | 1.89x |
TODO: rerun 2-3 times on the desktop and record the median.