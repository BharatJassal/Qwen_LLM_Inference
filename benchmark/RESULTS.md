# Results log (GTX 1080 Ti, Windows 11, PyTorch cu126, CUDA 12.6, VS 2022 Build Tools)

## Reproducing the nvprof timeline

nvprof ships with CUDA 12.6 and supports Pascal (Nsight Compute does not, and nsys and nvprof counters need admin rights here). It has to launch the real interpreter rather than the venv launcher, so the venv goes through `PYTHONPATH`:

```powershell
$env:PATH = "$env:CUDA_PATH\extras\CUPTI\lib64;$env:PATH"
$env:PYTHONPATH = "$PWD;$PWD\.venv\Lib\site-packages"
$py = (python -c "import sys; print(sys._base_executable)")
foreach ($m in "cached", "graph") { nvprof --profile-from-start off --print-gpu-trace --csv -u us --log-file benchmark/trace_$m.csv $py -m benchmark.profile_trace $m }
python -m benchmark.analyze_trace                          # timeline.svg
python -m benchmark.analyze_trace path/to/old/trace_graph.csv   # also fusion.svg, old vs current
```

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

## Reruns for medians
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


## Batched INT8 kernel and build fallback
- int8_ext.available() now tries the build once and returns False (with a warning) when it fails, instead of crashing on the first INT8 call. tests.test_ext_fallback passes; in a shell without ninja on PATH the app logged the warning and served FP32.
- cuda/int8_gemv_ext.cu takes x as (cols) or (batch, cols), batch 1 to 8, templated on batch so each warp loads a weight row once for all inputs. Batch 1 does the same arithmetic as v3.
- tests.test_int8_kernel: every layer-0 projection at batch 2 to 8 within 1e-5 rel of dequantize + matmul, each batched row bit-identical to running that row alone, batch 9 and cols not matching rejected, 200 token decode identical
- tests.test_batching: 6 greedy requests on 4 slots identical to greedy_cached for FP32 and INT8 engines. test_graph, test_sample, test_HF still pass.
- benchmark_quantized perplexity, first 300 tokens of the gutenberg.txt on this PC: fp32 5.4849, int8 full forward 5.4388, int8 token by token through the fused kernel 5.4388. This PC's gutenberg.txt is Frankenstein (ebook 84), whose first 300 tokens are the Project Gutenberg license header, not Pride and Prejudice (ebook 1342) as used for the README's 20.8487 / 20.8605, so the two are not comparable. The point here is that the kernel path matches the full forward on the same text.

## Perplexity remeasured on Pride and Prejudice
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

## INT8 output layer
- quantize_model now also quantizes model.embed_tokens.weight. It is tied, so one INT8 copy serves the embedding lookup (gather INT8 rows, multiply by their scales) and the output layer (linear, so decode goes through the kernel)
- tests: test_int8_kernel, test_batching (fp32 and int8), test_graph, test_ext_fallback, test_HF all pass
- benchmark_quantized: fp32 model 1884.59 MB, int8 model 473.09 MB (3.98x, was 862.00 MB / 2.19x). First 300 tokens: fp32 7.1541, int8 7.1538 (was 7.1433), int8 through kernel 7.1538
- Story text from Chapter 1 (fp32 / int8 / change): 300 tokens 12.8151 / 12.8968 / +0.64%, 1024 tokens 14.4113 / 14.5206 / +0.76%, 2048 tokens 14.8940 / 14.9675 / +0.49%. With the FP32 embedding it was +0.71 / +0.73 / +0.43%
- bench_int8_speed, 300 tokens, tok/s. Before the change (1 run): FP32 cached 56.9, graph 108.1, INT8 cached 54.1, graph 161.5. After (3 runs):

| Run | FP32 cached | FP32 graph | INT8 cached | INT8 graph |
|---|---|---|---|---|
| 1 | 58.0 | 108.4 | 59.1 | 195.9 |
| 2 | 57.8 | 108.4 | 60.0 | 195.7 |
| 3 | 58.2 | 108.1 | 60.3 | 195.3 |

- INT8 graph 161.5 to 195.7 tok/s (1.21x). Weight bytes per token went from ~904 MB to ~496 MB (1.82x), so the remaining time is not weight reads: 195.7 x 496 MB = ~97 GB/s, ~20% of the 484 GB/s peak
- bench_batching, 100 greedy tokens per request, 3 runs, tok/s (one at a time / batched):

| Weights | Requests | Run 1 | Run 2 | Run 3 | Median speedup |
|---|---|---|---|---|---|
| FP32 | 1 | 106.0 / 96.9 | 105.8 / 96.9 | 105.9 / 96.9 | 0.92x |
| FP32 | 4 | 105.4 / 317.2 | 105.8 / 317.3 | 106.0 / 316.7 | 3.00x |
| FP32 | 8 | 105.8 / 440.4 | 105.9 / 444.4 | 105.7 / 442.2 | 4.19x |
| INT8 | 1 | 185.2 / 151.7 | 185.8 / 151.1 | 185.2 / 149.0 | 0.81x |
| INT8 | 4 | 178.7 / 360.4 | 179.3 / 360.8 | 177.4 / 358.8 | 2.02x |
| INT8 | 8 | 177.6 / 466.1 | 176.8 / 466.0 | 177.0 / 462.5 | 2.62x |

- FP32 code did not change but measured ~10% faster than the earlier session (batched 8: 442.2 vs 398.6), so only compare numbers from the same session. bench_app was not re-run after this change

## Bandwidth and time breakdown per decode step (2026-11-23)
- bench_bandwidth: one decode step (batch 1, 512 token KV cache) captured as a CUDA graph, replayed 200 times and timed with CUDA events, then 20 replays under torch.profiler with kernels grouped by name
- Device to device copy of 1 GiB: 355 / 353 / 354 GB/s, 73% of the 484 GB/s spec. Used as the practical ceiling
- Bytes per step = every weight tensor (the tied embedding counts once, it is read in full as the output layer) + the whole K and V cache, since decode_step attends over all of it with a mask

| Run | FP32 ms/step | FP32 GB/s | INT8 ms/step | INT8 GB/s |
|---|---|---|---|---|
| 1 | 8.988 | 221 | 4.947 | 103 |
| 2 | 8.967 | 222 | 4.953 | 103 |
| 3 | 8.964 | 222 | 4.946 | 103 |

- Profiled kernel time per step (same in all 3 runs to within 0.01 ms):

| Group | FP32 kernels | FP32 ms | INT8 kernels | INT8 ms |
|---|---|---|---|---|
| Weight matmuls | 169 | 6.159 (321 GB/s, 91% of copy) | 169 | 1.915 (259 GB/s, 73% of copy) |
| Small ops | 756 | 2.214 | 831 | 2.219 |
| Attention | 72 | 0.742 | 72 | 0.740 |

- INT8 has 75 more kernels than FP32: 72 separate bias adds after the kernel for q/k/v (F.linear fuses the bias into cuBLAS) and 3 for the INT8 embedding lookup
- 202 decode steps/s agrees with greedy_graph's 195.7 tok/s, which also includes prefill, graph capture and picking each token on the host

## Kernel timeline with nvprof (2026-11-23)
- Nsight Systems 2024.4.2 starts the target suspended and exits right after "ReflexStatsTraceLoggingProvider provider requires administrator privileges", also with --sample=none --cpuctxsw=none and with the base interpreter, so it needs an admin terminal on this machine. Nsight Compute 2024.3 does not support Pascal (sm_61). nvprof hardware counters fail with "Internal profiling error 4211:27" without admin (GPU performance counters restricted to admins)
- nvprof tracing works without admin once extras/CUPTI/lib64 is on PATH (otherwise it exits with 0xC0000135, DLL not found) and it launches the base interpreter with the venv on PYTHONPATH (the venv python.exe is a launcher, the real process is its child)
- profile_trace: 5 token warm-up, then 20 INT8 tokens between cudaProfilerStart/Stop. analyze_trace splits steps at the output layer kernel (grid 18992), drops the first step

| Mode | Steps | Median step | GPU busy | Idle | Kernels per step |
|---|---|---|---|---|---|
| greedy_cached | 18 | 21.58 ms | 4.30 ms | 80% | 1119 |
| greedy_graph | 19 | 5.47 ms | 4.86 ms | 11% | 1076 |

- Cached runs at ~16.6 ms per token untraced, so tracing adds ~5 ms of CPU time per step there. Graph step 5.47 ms vs 4.95 ms back to back replays in bench_bandwidth: ~0.5 ms per token between replays (argmax, .item() sync, replay launch)
- One layer (layer 12 of the median step): cached 838 us, 81% idle, graph 201 us, 9% idle
- INT8 kernel per projection inside the graph, median us over 19 steps x 24 layers, GB/s counts W + scales + x:

| Projection | Shape | Blocks | us | GB/s |
|---|---|---|---|---|
| q | 896x896 | 112 | 5.22 | 155 |
| k | 128x896 | 16 | 3.46 | 34 |
| v | 128x896 | 16 | 3.42 | 35 |
| o | 896x896 | 112 | 5.09 | 159 |
| gate | 4864x896 | 608 | 17.12 | 256 |
| up | 4864x896 | 608 | 16.61 | 264 |
| down | 896x4864 | 112 | 15.68 | 279 |
| output | 151936x896 | 18992 | 386.45 | 354 |

- Sum 24 x 66.6 + 386 = 1.98 ms, matches bench_bandwidth's 1.92 ms of weight matmuls. int8_gemv<1> uses 31 registers per thread, 256 threads per block
- bench_bandwidth re-run after moving category() into analyze_trace: same kernel counts per group, INT8 4.983 ms per step, copy 350 GB/s

## Kernel fusion (2026-11-23)
- merge_projections (loader): q/k/v stacked into qkv_proj 1152x896 (+ concatenated bias), gate/up into gate_up_proj 9728x896. int8_gemv takes an optional bias. cuda/fused_ops.cu: rmsnorm (optionally adds a residual and writes the sum), rope_ (in place on the first heads of each token, _rn intrinsics), silu_mul. Bindings moved to cuda/bindings.cpp, loader renamed src/int8_ext.py -> src/kernels.py, extension name qwen_kernels
- tests: test_fused_ops (rmsnorm max rel err 7.9e-08 to 1.2e-07, add_rmsnorm sum identical, rope bit-identical and v untouched, silu_mul identical, 5 bad inputs rejected), test_int8_kernel (merged shapes, bias, 200 token decode with all kernels identical to plain PyTorch), test_HF (logits 3.62e-05, attention 6.56e-07, mlp 2.29e-05), test_graph, test_batching, test_ext_fallback, test_sample all pass
- perplexity: benchmark_quantized fp32 7.1541, int8 7.1538, via kernel 7.1538. Story text fp32 / int8: 300 tokens 12.8150 / 12.8968, 1024 tokens 14.4112 / 14.5206, 2048 tokens 14.8940 / 14.9675 (fp32 differs from before in the 4th decimal at most, RMSNorm sums in a different order)
- First attempt at timing was with osu! and Wallpaper Engine running (GPU at 22 to 30% utilization, SM clock ~1330 MHz): before 6.17 to 6.80 ms, after 3.74 to 4.04 ms. Discarded, re-run with both closed (3% utilization)
- A = previous commit fe611b6 in a git worktree, B = current code with only the INT8 kernel enabled (scratch runner wrapping layers.use_kernels so it is True only inside linear), C = current code. Run A, B, C, A, B, C, A, B, C. bench_bandwidth, ms per step:

| Round | Copy GB/s (A/B/C) | A FP32 | A INT8 | B FP32 | B INT8 | C FP32 | C INT8 |
|---|---|---|---|---|---|---|---|
| 1 | 338 / 340 / 336 | 9.409 | 5.152 | 8.949 | 4.755 | 7.607 | 3.362 |
| 2 | 339 / 338 / 337 | 9.446 | 5.177 | 8.967 | 4.744 | 7.607 | 3.360 |
| 3 | 339 / 338 / 337 | 9.431 | 5.182 | 8.977 | 4.747 | 7.600 | 3.359 |

- Kernels per step: FP32 997 / 875 / 390, INT8 1072 / 878 / 393. INT8 groups, median (A -> C): weight matmuls 169 kernels 2.047 ms 72% of copy -> 97 kernels 1.790 ms 82% of copy; small ops 831 kernels 2.258 ms -> 224 kernels 0.728 ms; attention 72 kernels 0.764 -> 0.794 ms
- bench_int8_speed, alternating A and C, tok/s:

| Round | A fp32 cached | A fp32 graph | A int8 cached | A int8 graph | C fp32 cached | C fp32 graph | C int8 cached | C int8 graph |
|---|---|---|---|---|---|---|---|---|
| 1 | 53.8 | 103.8 | 52.9 | 189.8 | 68.7 | 136.7 | 80.0 | 293.4 |
| 2 | 54.1 | 108.8 | 53.8 | 190.7 | 67.9 | 136.0 | 80.1 | 293.4 |
| 3 | 53.7 | 108.9 | 53.3 | 188.8 | 68.3 | 135.9 | 78.3 | 283.6 |

- bench_batching, alternating A and C, one at a time / batched tok/s:

| Weights | Requests | A run 1 | A run 2 | A run 3 | C run 1 | C run 2 | C run 3 |
|---|---|---|---|---|---|---|---|
| FP32 | 1 | 105.5 / 98.1 | 103.5 / 96.6 | 107.6 / 98.4 | 134.9 / 115.7 | 135.6 / 116.3 | 137.3 / 117.6 |
| FP32 | 4 | 106.6 / 301.5 | 107.0 / 318.4 | 107.1 / 275.2 | 134.9 / 367.0 | 136.9 / 369.2 | 137.1 / 373.0 |
| FP32 | 8 | 108.5 / 423.1 | 107.8 / 430.2 | 95.2 / 383.2 | 134.7 / 475.2 | 124.2 / 474.3 | 136.9 / 485.6 |
| INT8 | 1 | 180.1 / 148.2 | 181.9 / 150.6 | 168.2 / 138.4 | 277.0 / 195.5 | 281.5 / 197.2 | 283.1 / 194.9 |
| INT8 | 4 | 174.3 / 345.5 | 177.0 / 352.7 | 171.2 / 346.0 | 263.2 / 405.2 | 267.5 / 417.0 | 267.1 / 416.3 |
| INT8 | 8 | 173.6 / 444.0 | 163.7 / 416.6 | 175.0 / 453.4 | 260.2 / 460.5 | 263.9 / 521.5 | 264.0 / 520.5 |

- A lone request in the batch engine is now 31% slower than the single request path (195.5 vs 281.5), up from 18%: its attention over the full 2048 token slot is a fixed cost
- bench_int8_batch on merged shapes, 3 runs, µs (kernel / dequantize / cuBLAS), median: gate_up b1 28.71 / 436.24 / 97.57, b8 112.96 / 493.58 / 156.30; down b1 15.91 / 231.00 / 52.55, b8 51.98 / 282.59 / 103.99; qkv b1 9.31 / 64.15 / 15.80, b8 20.53 / 74.49 / 24.97. Raw kernel gate_up b1: 30.03 28.68 28.71
- nvprof trace after fusion: without graph 15.09 ms per step, 84% idle, 490 kernels; with graph 3.16 ms, 11% idle, 397 kernels. Layer 12: 586 us / 109 us. Before fusion (graph trace of A): 5.61 ms, 1076 kernels, layer 12 205 us with 44 kernels; after: layer 12 109 us with 16 kernels
- Per projection after merging, us / GB/s: qkv 5.50 / 189, o 4.96 / 163, gate_up 28.00 / 313, down 14.58 / 301, output 359.18 / 381. The output layer beats the 338 GB/s measured copy: a copy reads and writes, the matmul mostly reads, so the copy is a reference, not a ceiling

## Baselines vs Hugging Face and llama.cpp (2026-11-24)

bench_baselines: "Write a story about the ocean." chat prompt, 300 new tokens, greedy, one request, median of 3 runs per round. Hugging Face transformers `generate` with `min_new_tokens=300`, default attention. llama.cpp b12686 official Windows CUDA 12.4 build, `llama-bench -ngl 99 -p 0 -n 300 -fa 1`, Qwen's own qwen2.5-0.5b-instruct-q8_0.gguf (638.74 MiB). Other apps held the GPU at ~9% while idle.

| tok/s | round 1 | round 2 |
|---|---|---|
| HF FP32 | 68.0 | 71.2 |
| HF FP16 | 72.2 | 72.2 |
| this engine FP32 (graph) | 131.0 | 130.4 |
| this engine INT8 (graph) | 286.0 | 285.1 |
| llama.cpp Q8_0 | 197.0 | 198.4 |

- Earlier standalone runs while the GPU was busier: llama.cpp tg300 160.6 / 168.3 with flash attention off, 169.2 / 170.6 on; engine INT8 256.6 / 264.9 / 286.1. The chart uses round 2.
- HF FP16 is no faster than FP32: at ~14 ms per token the GPU mostly waits on Python, so halving the bytes changes nothing. Pascal also runs FP16 math at 1/64 rate.
- llama-bench with 32 tokens instead of 300 gives 204 tok/s: attention over a longer context costs more per token.
- llama.cpp was not profiled, so the gap is measured, not explained. Q8_0 stores an FP16 scale per 32 weights (8.5 bits per weight) and an untied output matrix; this engine uses one scale per row.
