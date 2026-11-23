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

Rerun 3 times on the desktop (2026-11-20), median per shape, us:
| Shape | INT8 kernel | cuBLAS FP32 | Speedup |
|---|---|---|---|
| gate/up | 18.36 | 54.98 | 2.99x |
| down | 15.82 | 54.67 | 3.46x |
| q/o | 6.73 | 12.00 | 1.78x |
q/o is noisy (5.38-11.26 us across runs): at ~6 us it sits on the ~5 us launch floor.

## INT8 kernel in the model (int8_gemv_ext via torch cpp_extension, layers.linear)
- tests.test_int8_kernel: all 7 layer-0 projections max rel err <= 2.6e-07 vs dequantize; 200-token greedy decode identical with USE_FUSED on/off
- bench_int8_speed, 300 tokens cached, 3 runs: FP32 50.4 / 50.1 / 51.5 tok/s, INT8 46.8 / 46.7 / 47.4 tok/s
- Median INT8 46.8 tok/s vs 28.3 before (1.65x), still ~7% behind FP32: decode is CPU launch bound (see profile), so the ~3x faster GEMVs mostly hide behind launch overhead

## CUDA graph decode (generate.greedy_graph, model.decode_step)
Prefill runs on the normal path; each decode token is one replay of a captured graph instead of ~1100 launches.
Needs a fixed-size KV cache (written in place at pos, later slots masked) and RoPE tables indexed on the GPU.
- tests.test_graph: 300-token greedy decode identical to greedy_cached for FP32 and INT8 (fused kernel inside the graph)
- bench_int8_speed, 300 tokens, 3 runs, tok/s (includes prefill and graph capture):

| Weights | greedy_cached | greedy_graph | Speedup |
|---|---|---|---|
| FP32 | 46.9 / 51.6 / 52.4 (median 51.6) | 95.2 / 102.9 / 102.6 (median 102.6) | 2.0x |
| INT8 | 42.4 / 48.9 / 49.7 (median 48.9) | 147.8 / 144.6 / 150.9 (median 147.8) | 3.0x |

With launch overhead gone, INT8 is finally faster than FP32 end to end: 147.8 vs 102.6 tok/s (1.44x), where it was ~5% slower before.

## Chat app on the graph path (stream_sample -> stream_graph on CUDA)
App now loads weights on CUDA when available (was always CPU); greedy and sampling share stream_graph, token choice (argmax / sample) runs outside the graph.
Both endpoints share one threading.Lock (stream endpoint had none; concurrent graph captures would fail).
- tests.test_graph: seed 0, T=0.8, top_p=0.9, 100 tokens: graph sampling identical to cached sampling (FP32 and INT8); tests.test_sample Degenerate Pass
- bench_int8_speed after moving argmax outside the graph: FP32 graph 102.8 tok/s, INT8 graph 148.8 tok/s (no change)
- bench_app (100-token /generate, FP32 weights, GPU + graphs):

| Clients | p50 (s) | p95 (s) | Wall (s) | Req/s |
|---|---|---|---|---|
| 1 | 1.31 | 1.31 | 1.31 | 0.76 |
| 4 | 3.36 | 3.85 | 3.85 | 1.04 |
| 8 | 4.09 | 5.91 | 5.92 | 1.35 |

## Reruns for medians (2026-11-20)
- INT8 dequantize every call (layers.USE_FUSED = False), greedy_cached 300 tokens, 3 runs: 30.4 / 30.3 / 30.5 tok/s, median 30.4 (earlier single run: 28.3)
- bench_app, GPU, FP32, CUDA graphs, 3 more runs (p50 / p95 / wall / req/s):

| Clients | Run 1 | Run 2 | Run 3 | Median per column |
|---|---|---|---|---|
| 1 | 1.10 / 1.10 / 1.10 / 0.91 | 0.47 / 0.47 / 0.47 / 2.14 | 0.77 / 0.77 / 0.77 / 1.30 | 0.77 / 0.77 / 0.77 / 1.30 |
| 4 | 2.50 / 2.68 / 2.68 / 1.49 | 3.32 / 4.33 / 4.33 / 0.92 | 1.64 / 2.66 / 2.66 / 1.50 | 2.50 / 2.68 / 2.68 / 1.49 |
| 8 | 5.35 / 6.77 / 6.77 / 1.18 | 5.01 / 6.36 / 6.37 / 1.26 | 4.12 / 7.19 / 7.19 / 1.11 | 5.01 / 6.77 / 6.77 / 1.18 |

Large run to run spread: requests sample at temperature 0.7 and stop at EOS, so reply length (work per request) varies. max_tokens=100 is a cap, not a fixed length.

## Continuous batching (src/batching.py BatchEngine, 8 slots, max_len 2048, FP32)
- tests.test_batching: 6 greedy requests on 4 slots (2 queued, 2 stop early at EOS), every output identical to greedy_cached; over-long request rejected; 4 cancelled streams free their slots
- Bug found on the way: RoPE tables and positions were locals in __init__, freed after capture, and the graph then read reused memory (first decode logits off by 19.6). Fixed by keeping them on the engine.
- decode_step now batched and uses grouped GQA instead of repeat_interleave; test_graph still passes, greedy_graph FP32 99.3 / INT8 145.8 tok/s (1 run)
- Graph step cost (ms): 1 slot 11.23, 8 slots 16.80 at max_len 2048; 9.59 / 14.00 at max_len 256. So graphs are captured per batch size (1, 2, 4, 8).
- bench_batching, 100 greedy tokens per request, 3 runs, tok/s (one at a time / batched):

| Requests | Run 1 | Run 2 | Run 3 | Median speedup |
|---|---|---|---|---|
| 1 | 98.1 / 88.2 | 98.6 / 88.1 | 97.2 / 87.7 | 0.90x |
| 4 | 97.9 / 284.8 | 98.3 / 286.0 | 98.0 / 285.4 | 2.91x |
| 8 | 98.1 / 402.5 | 97.9 / 400.1 | 97.9 / 401.0 | 4.10x |

With a single 8 slot graph (before per-size graphs) a lone request ran at 58.0 tok/s.

- bench_app, batched app, 3 runs (p50 / p95 / wall / req/s):

| Clients | Run 1 | Run 2 | Run 3 | Median per column |
|---|---|---|---|---|
| 1 | 0.47 / 0.47 / 0.47 / 2.12 | 1.53 / 1.53 / 1.53 / 0.65 | 1.54 / 1.54 / 1.54 / 0.65 | 1.53 / 1.53 / 1.53 / 0.65 |
| 4 | 2.04 / 2.04 / 2.04 / 1.96 | 1.90 / 1.90 / 1.90 / 2.10 | 2.00 / 2.00 / 2.00 / 2.00 | 2.00 / 2.00 / 2.00 / 2.00 |
| 8 | 2.90 / 2.91 / 2.91 / 2.75 | 2.76 / 2.76 / 2.76 / 2.90 | 3.00 / 3.01 / 3.01 / 2.66 | 2.90 / 2.91 / 2.91 / 2.75 |


## Batched INT8 kernel and build fallback (2026-11-23)
- int8_ext.available() now tries the build once and returns False (with a warning) when it fails, instead of crashing on the first INT8 call. tests.test_ext_fallback passes; in a shell without ninja on PATH the app logged the warning and served FP32.
- cuda/int8_gemv_ext.cu takes x as (cols) or (batch, cols), batch 1 to 8, templated on batch so each warp loads a weight row once for all inputs. Batch 1 does the same arithmetic as v3.
- tests.test_int8_kernel: every layer-0 projection at batch 2 to 8 within 1e-5 rel of dequantize + matmul, each batched row bit-identical to running that row alone, batch 9 and cols not matching rejected, 200 token decode identical
- tests.test_batching: 6 greedy requests on 4 slots identical to greedy_cached for FP32 and INT8 engines. test_graph, test_sample, test_HF still pass.
- benchmark_quantized perplexity, first 300 tokens of the gutenberg.txt on this PC: fp32 5.4849, int8 full forward 5.4388, int8 token by token through the fused kernel 5.4388. This PC's gutenberg.txt is Frankenstein (ebook 84), whose first 300 tokens are the Project Gutenberg license header, not Pride and Prejudice (ebook 1342) as used for the README's 20.8487 / 20.8605, so the two are not comparable. The point here is that the kernel path matches the full forward on the same text.

## Perplexity re-measured on Pride and Prejudice (2026-11-23)
- gutenberg.txt replaced with pg1342.txt (772,386 bytes) from the official mirror gutenberg.pglaf.org (gutenberg.org was down for maintenance). Old Frankenstein file kept outside the repo.
- benchmark_quantized as is (first 300 tokens = license header): fp32 7.1541, int8 7.1433, int8 through kernel 7.1433 on GPU; CPU fp32 7.1540, int8 7.1428
- The README's July 20.8487 / 20.8605 could not be reproduced. First 300 tokens of other versions, fp32: 1342-0.txt 33.43, text after the START marker 34.58, from Chapter I heading 10.17, from the opening line 12.82, HTML 2.92 to 3.58, CRLF line endings 6.69. None is 20.85, so the July input is unknown.
- Story text from Chapter 1's opening line (fp32 / int8 / change): 300 tokens 12.8151 / 12.9054 / +0.71%, 1024 tokens 14.4113 / 14.5161 / +0.73%, 2048 tokens 14.8940 / 14.9574 / +0.43%. int8 through the kernel on the 300 token slice: 12.9054 (kernel confirmed built).
- bench_int8_batch, µs per call, 3 runs, median (kernel / dequantize + F.linear / FP32 cuBLAS):

| Projection | Batch 1 | Batch 2 | Batch 4 | Batch 8 |
|---|---|---|---|---|
| gate/up 4864x896 | 18.14 / 241.74 / 55.94 | 22.99 / 240.07 / 57.75 | 39.51 / 241.40 / 57.10 | 70.11 / 263.26 / 79.63 |
| down 896x4864 | 18.27 / 252.73 / 53.80 | 24.24 / 250.65 / 55.38 | 35.00 / 240.74 / 56.81 | 62.24 / 288.55 / 101.59 |
| q/o 896x896 | 11.16 / 54.44 / 13.28 | 10.65 / 54.86 / 13.44 | 10.96 / 55.52 / 14.20 | 13.45 / 64.59 / 22.81 |

Raw runs, kernel gate b1..b8: 19.92 27.98 42.41 70.11 | 18.14 22.99 39.51 75.93 | 17.30 20.90 34.32 66.50

- bench_batching, 100 greedy tokens per request, 3 runs, tok/s (one at a time / batched):

| Weights | Requests | Run 1 | Run 2 | Run 3 | Median speedup |
|---|---|---|---|---|---|
| FP32 | 1 | 103.2 / 92.5 | 100.8 / 94.9 | 99.2 / 88.9 | 0.90x |
| FP32 | 4 | 99.6 / 296.6 | 101.0 / 291.5 | 98.9 / 282.4 | 2.89x |
| FP32 | 8 | 96.5 / 418.4 | 101.3 / 389.8 | 95.3 / 398.6 | 4.18x |
| INT8 | 1 | 144.8 / 130.1 | 149.1 / 129.7 | 140.4 / 123.7 | 0.88x |
| INT8 | 4 | 149.9 / 343.3 | 145.9 / 335.9 | 138.7 / 315.7 | 2.29x |
| INT8 | 8 | 149.6 / 434.7 | 147.0 / 442.1 | 142.1 / 428.6 | 3.01x |

Before the batched kernel, the INT8 engine would fall back to dequantize for batch > 1 (about 240 µs per gate/up call), which is why the app served FP32.

- App now quantizes when the kernel is available. bench_app, INT8 batched app, 3 runs (p50 / p95 / wall / req/s):

| Clients | Run 1 | Run 2 | Run 3 | Median per column |
|---|---|---|---|---|
| 1 | 1.33 / 1.33 / 1.33 / 0.75 | 1.22 / 1.22 / 1.22 / 0.82 | 0.44 / 0.44 / 0.44 / 2.25 | 1.22 / 1.22 / 1.22 / 0.82 |
| 4 | 1.83 / 1.83 / 1.83 / 2.19 | 1.81 / 1.81 / 1.81 / 2.21 | 1.03 / 1.64 / 1.64 / 2.44 | 1.81 / 1.81 / 1.81 / 2.21 |
| 8 | 2.88 / 2.88 / 2.88 / 2.77 | 2.78 / 2.78 / 2.79 / 2.87 | 2.71 / 2.72 / 2.72 / 2.94 | 2.78 / 2.78 / 2.79 / 2.87 |
