# Qwen LLM Inference

Built an inference engine for **Qwen2.5-0.5B-Instruct** from scratch, developed in pure PyTorch by loading raw `safetensors` weights directly, without using `transformers`. Implemented the Llama architecture (RoPE, GQA, SwiGLU, RMSNorm), autoregressive decoding with sampling, and KV caching, verifying each component against HuggingFace. On NVIDIA GPUs, decoding runs custom CUDA kernels (an INT8 matrix-vector kernel plus fused RMSNorm, RoPE and SiLU) and replays each token from a CUDA graph: **293 tok/s INT8 on a GTX 1080 Ti, 2.2x FP32**. Profiling drove each step, and the analysis is below. The server batches concurrent requests continuously on INT8 weights, reaching 521 tok/s combined across 8 requests.


## Purpose

Modern LLM stacks such as vLLM, llama.cpp and more hide the complexity involved in running a model. This project rebuilds that pipeline from the ground up, to better understand how it works. From mapping weights to computations, understanding how KV caching speeds up generation by avoiding redundant computations.

## Results

| Check | Method | Result |
|---|---|---|
| End to end logits | Max abs. deviation vs HuggingFace reference, fp32 | 6.08e-05, gate: < 1e-3 |
| Layer components | Forward hooks on reference model, layer 0 | RMSNorm, attention, MLP all < 1e-4 |
| KV cache | Greedy decode, cached vs uncached, 300 token generation | Token-for-token identical |
| INT8 CUDA kernel | Every layer-0 projection, batch 1 to 8, with and without bias, vs dequantize + matmul | Max rel. error 2.4e-07 |
| Fused RMSNorm, RoPE, SiLU kernels | vs the PyTorch versions on random inputs | RMSNorm max rel. error 1.2e-07, RoPE and SiLU bit-identical |
| All custom kernels together | 200 token greedy decode vs plain PyTorch, INT8 | Token-for-token identical |
| CUDA graph decode | 300 token greedy and 100 token seeded sampling, graph vs cached, FP32 and INT8 | Token-for-token identical |
| Continuous batching | 6 greedy requests on 4 slots vs `greedy_cached`, FP32 and INT8 | Token-for-token identical |

### Throughput (KV-cache impact)

Measured on M4 MacBook Air.

| Generated tokens | No cache (tok/s) | KV cache (tok/s) | Speedup |
|---|---|---|---|
| 30  | 7.87 | 23.4 | 3.0x |
| 150 | 6.21 | 27.4 | 4.4x |
| 300 | 4.07 | 26.3 | 6.4x |

Without KV caching, every new token requires a full forward pass over the entire sequence, making total generation O(n²). With KV caching, the model stores each layer's key and value tensors, therefore a new token only computes its own QKV and attends against cached history, O(n) total. Cached throughput stays roughly flat while uncached throughput falls, so the speedup factor grows with generation length: 3.0× at 30 tokens, 6.4× at 300.

- **Grouped-query attention:** 2 KV heads serve 14 query heads; each KV head is broadcast across its query group before scaled dot product.
- **RoPE:** rotate half convention, precomputed cos/sin tables from `rope_theta`; during cached decoding, the new token's rotation uses its **absolute** position.
- **Qwen-specific details:** bias terms on QKV projections, tied input/output embeddings

### INT8 Quantization

Linear layer weights and the embedding table are quantized using per channel symmetric INT8: each row gets one FP32 scale, `scale = max|row| / 127`, and stores `round(w / scale)`. Layer norms and biases stay in FP32.

| Measurement | fp32 | int8 | Change |
|---|---|---|---|
| Quantized tensors | 1884.31 MB | 472.82 MB | 3.99x reduction |
| Whole model | 1884.59 MB | 473.09 MB | 3.98x reduction |
| Perplexity, story text, 300 tokens | 12.8151 | 12.8968 | +0.64% |
| Perplexity, story text, 1024 tokens | 14.4113 | 14.5206 | +0.76% |
| Perplexity, story text, 2048 tokens | 14.8940 | 14.9675 | +0.49% |

Measured on Project Gutenberg ebook 1342 (*Pride and Prejudice*), starting at Chapter 1. INT8 costs under 1% perplexity. `benchmark_quantized` itself reads the first 300 tokens of the file, which are the Gutenberg license header, so it reports a lower 7.1541 FP32 / 7.1538 INT8. On a GPU it also runs the text one token at a time through the fused kernel and gets the same INT8 perplexity to 4 decimals.

The embedding table (151936 x 896, ~519 MB in FP32) is over a quarter of the model, and because Qwen ties its embeddings it doubles as the output layer that turns the final hidden state into a score for every token. Quantizing it was the last FP32 matrix left, so it takes the whole model from 2.19x to 3.98x smaller. A lookup reads the token's INT8 row and multiplies by its scale; the output layer runs through the same kernel as every other projection. Since one tensor feeds both the input and the logits, its rounding error shows up twice, but perplexity moved less than 0.1 points against the previous INT8 numbers (+0.71/+0.73/+0.43% with an FP32 embedding).

On CPU (and for prompt processing on GPU), weights dequantize to FP32 right before `torch.matmul`, which keeps the full RAM savings in plain PyTorch but costs speed. On an NVIDIA GPU, decode (up to 8 tokens per call, one per request in a batch) instead calls a fused INT8 kernel that never dequantizes, described below.

## GPU: custom CUDA kernels and CUDA graphs

Measured on a GTX 1080 Ti (484 GB/s peak bandwidth), Windows 11, PyTorch cu126, CUDA 12.6. Full log in [`benchmark/RESULTS.md`](benchmark/RESULTS.md).

### Fused INT8 GEMV kernel

Decoding one token is a matrix-vector product per projection, so it is limited by how fast weights stream from memory, not by compute. The kernel ([`cuda/int8_gemv_ext.cu`](cuda/int8_gemv_ext.cu)) reads the INT8 weights directly and applies the per-row scale once at the end: `y[i] = scale[i] * sum_j W[i][j] * x[j]`. That moves 4x fewer weight bytes than FP32.

| Version (µs per call) | gate/up 4864x896 | down 896x4864 | q/o 896x896 |
|---|---|---|---|
| v1: one block per row | 73.37 | 37.81 | 14.16 |
| v2: one warp per row, shuffle reduction | 31.19 | 38.08 | 7.12 |
| v3: `char4`/`float4` vectorized loads (final) | 18.76 | 17.50 | 6.21 |

Final kernel vs cuBLAS FP32 GEMV, median of 3 runs:

| Shape | INT8 kernel (µs) | cuBLAS FP32 (µs) | Speedup |
|---|---|---|---|
| gate/up | 18.36 | 54.98 | 2.99x |
| down | 15.82 | 54.67 | 3.46x |
| q/o | 6.73 | 12.00 | 1.78x |

q/o gains least because at ~6 µs it is close to the ~5 µs cost of launching an empty kernel. Tried and dropped on top of v3: 16-byte weight loads (slower, because the matching `x` reads scatter), staging `x` in shared memory (mixed), and `#pragma unroll` (no change).

#### Batched kernel

With batched kernel, each step multiplies every weight matrix up to 8 vectors, one per active request. Running the kernel 8 times reads the weights 8 times, the kernel takes a batch: each warp loads a weight row once and accumulates it against every input, with the batch size as a template parameter so the 8 partial sums stay in registers. Each row's arithmetic is identical to a batch of 1, so a request gets bit-identical results whether it runs alone or in a batch. The kernel also adds an optional bias after the scale.

µs per call on the merged matrices (see [kernel fusion](#kernel-fusion)), median of 3 runs ([`benchmark/bench_int8_batch.py`](benchmark/bench_int8_batch.py)):

| Shape | Batch | INT8 kernel | Dequantize + matmul | cuBLAS FP32 |
|---|---|---|---|---|
| gate_up 9728x896 | 1 | 28.71 | 436.24 | 97.57 |
| gate_up | 8 | 112.96 | 493.58 | 156.30 |
| down 896x4864 | 1 | 15.91 | 231.00 | 52.55 |
| down | 8 | 51.98 | 282.59 | 103.99 |
| qkv 1152x896 | 1 | 9.31 | 64.15 | 15.80 |
| qkv | 8 | 20.53 | 74.49 | 24.97 |

The INT8 advantage shrinks as the batch grows: at batch 8 the kernel is doing 8x the math on each weight byte and stops being purely memory bound. It still beats FP32 at every size, and is about 4x faster than dequantizing, which is what batched INT8 used before.

### CUDA graphs

A 3x faster kernel barely changed end-to-end speed at first: INT8 decode was still slower than FP32. Profiling showed why. Each token launches ~1100 small GPU operations, and the GPU sat idle about half the time waiting for Python to launch the next one.

[`greedy_graph` / `stream_graph`](src/generate.py) process the prompt normally, then record one decode step ([`decode_step`](src/model.py)) as a CUDA graph and replay it once per token, one launch instead of ~1100. A graph needs fixed shapes and memory addresses, so the decode step:

- uses a preallocated KV cache, writing each new token's K/V in place at its position and masking slots not yet written (the regular path grows the cache with `torch.cat`);
- builds the RoPE tables once on the GPU and indexes them with a position tensor that also lives on the GPU;
- leaves choosing the next token (`argmax` or top-p `sample`) outside the graph, so sampling behaves exactly as on the regular path.

### End-to-end decode (300 tokens, median of 3 runs)

Benchmarked in alternating runs with the current code so both see the same GPU conditions.

| Weights | Cached decode (tok/s) | Cached + CUDA graph (tok/s) | Graph speedup |
|---|---|---|---|
| FP32, before fusion | 53.8 | 108.8 | 2.0x |
| FP32 | 68.3 | 136.0 | 2.0x |
| INT8, before fusion | 53.3 | 189.8 | 3.6x |
| INT8 | 80.0 | **293.4** | 3.7x |

Timings include prompt processing and graph capture. INT8 is 2.2x faster than FP32. Dequantizing on every call instead of using the kernel ran at 30 tok/s.

Took three rounds of measuring. Quantizing the output layer cut the weight bytes read per token from ~904 MB to ~496 MB (1.82x), but speed rose only 1.21x (161.5 tok/s to 195.7 tok/s), so weight bytes were no longer the main cost. The next two sections find where the time went, and [kernel fusion](#kernel-fusion) fixes it.

### Where the time went (before fusion)

A decode step has to read every weight once, so the fastest it could possibly run is by the bytes it reads divided by the memory bandwidth. Comparing the real step time against that limit shows how much is left to gain. The spec sheet for my 1080 TI says 484 GB/s, but a large device copy only reaches about 340 GB/s 70%, so this copy is used as a reference. A copy reads and writes every byte, while a matmul mostly reads, and the output layer kernel later beats it. [`bench_bandwidth`](benchmark/bench_bandwidth.py) replays one decode step's CUDA graph 200 times, then profiles it and groups every kernel by what it does. Median of 3 runs, 512 token KV cache:

| | FP32 | INT8 |
|---|---|---|
| Bytes read per step | 1989 MB | 509 MB |
| Time per step | 8.97 ms | 4.95 ms |
| Achieved bandwidth | 222 GB/s (63% of copy) | 103 GB/s (29% of copy) |
| Step time if it ran at copy speed | 5.62 ms | 1.44 ms |
| Kernels per step | 997 | 1072 |

Where an INT8 step's 4.9 ms goes:

| Group | Kernels | Time | Share | Bandwidth |
|---|---|---|---|---|
| Small ops (RMSNorm, RoPE, residual adds, SiLU, bias) | 831 | 2.22 ms | 46% | |
| Weight matmuls (fused INT8 kernel) | 169 | 1.92 ms | 39% | 259 GB/s, 73% of copy |
| Attention (QK, softmax, AV) | 72 | 0.74 ms | 15% | |

- **Small ops are the biggest cost.** Each of the 831 kernels does little work, about 2.7 µs on average, which is close to the fixed cost of any kernel. They take the same 2.2 ms in FP32 and INT8, which is why INT8's lead shrank as its matmuls got faster. RMSNorm alone is ~294 of them (square, mean, add eps, sqrt, divide, scale, for each of 49 norms) and RoPE ~240.
- **The INT8 kernel reaches 73% of copy speed**, while cuBLAS's FP32 matmuls reach 91% (321 GB/s). Matching cuBLAS would save about 0.4 ms per step.
- **Attention** runs through general cuBLAS batched matmuls plus a separate softmax kernel for every layer, over the whole preallocated cache. A single fused decode attention kernel would replace these.

If every step only cost its weight reads at copy speed, INT8 would run near 700 tok/s instead of 202.

### Kernel timeline (before fusion)

[`profile_trace`](benchmark/profile_trace.py) decodes 20 INT8 tokens under nvprof, recording only the measured run, and [`analyze_trace`](benchmark/analyze_trace.py) splits the trace into decode steps (each ends with the 18,992 block output layer kernel), prints per step and per projection numbers and draws the charts below.

| Median over 18 to 19 steps | Step time | GPU busy | GPU idle | Kernels |
|---|---|---|---|---|
| Without CUDA graph | 21.58 ms | 4.30 ms | 80% | 1119 |
| With CUDA graph | 5.47 ms | 4.86 ms | 11% | 1076 |

Without the graph, the GPU spends 80% of each step waiting for Python to launch the next kernel (tracing adds some CPU cost, so this run is slower than the untraced 16.6 ms). With the graph, the kernels run back to back. The remaining 11% (~0.5 ms) is between steps: picking the token, copying it to the CPU with `.item()`, and launching the next replay. Back to back replays in `bench_bandwidth` take 4.95 ms.

Per projection, INT8 kernel inside the graph:

| Projection | Rows x cols | Blocks | µs | GB/s |
|---|---|---|---|---|
| q, o | 896 x 896 | 112 | 5.2 | 155 to 159 |
| k, v | 128 x 896 | 16 | 3.4 | 34 to 35 |
| gate, up | 4864 x 896 | 608 | 16.6 to 17.1 | 256 to 264 |
| down | 896 x 4864 | 112 | 15.7 | 279 |
| output layer | 151936 x 896 | 18992 | 386.5 | 354 |

The output layer runs at the measured copy speed, so the kernel saturates memory once a matrix is big enough. The small attention projections don't: k and v launch 16 blocks on a GPU with 28 SMs and take about the same ~3.4 µs any kernel costs. q, k, v and o together take 26% of the matmul time for 4% of the bytes.

Nsight Compute, which reads per-kernel hardware counters, only supports Volta and newer GPUs, and on Windows nvprof's counters need administrator rights, so these numbers come from timing and known byte counts.

### Kernel fusion

The profile pointed at two fixes: fewer, bigger matmuls, and far fewer small kernels.

- **Merged projections.** [`merge_projections`](src/loader.py) stacks q, k and v into one 1152 x 896 matrix and gate and up into one 9728 x 896 matrix at load time. Quantization is per row, so the INT8 weights are unchanged. Each layer runs 4 matmuls instead of 7, and the q/k/v bias is added inside the INT8 kernel instead of by 3 separate kernels.
- **Fused RMSNorm and residual add.** One block per row sums the squares with warp shuffles, then writes the normalized row, and with a residual also writes `x + residual` for the next layer. The layer loop in [`model.py`](src/model.py) carries the sum `x` and its normalized copy `h`, so every residual add joins the norm after it. One kernel replaces up to 8.
- **Fused RoPE.** After merging, q and k sit next to each other in the qkv output, so one kernel rotates both in place. It uses `__fmul_rn`/`__fadd_rn` so the compiler can't fuse multiply and add into an FMA, which keeps it bit-identical to the PyTorch version.
- **Fused SiLU x up.** One kernel reads both halves of the gate_up output.

The kernels are in [`cuda/fused_ops.cu`](cuda/fused_ops.cu) and build into the same extension as the INT8 kernel. Each op in [`layers.py`](src/layers.py) uses its kernel on the GPU and the original PyTorch code otherwise (CPU, or if the build fails), and the PyTorch versions are what [`test_fused_ops`](tests/test_fused_ops.py) checks the kernels against. Perplexity is unchanged to 4 decimals.

Each stage measured with `bench_bandwidth`, all three versions run in alternating rounds, median of 3:

| INT8 decode step | Kernels | Time per step | Steps/s |
|---|---|---|---|
| Before | 1072 | 5.177 ms | 193 |
| Merged projections, bias in the kernel | 878 | 4.747 ms | 211 |
| Plus fused RMSNorm, RoPE, SiLU | **393** | **3.360 ms** | **298** |

FP32 goes from 9.431 to 7.607 ms per step with the same changes. Where an INT8 step's time goes now:

| Group | Kernels | Before | After |
|---|---|---|---|
| Weight matmuls | 169 → 97 | 2.05 ms (72% of copy speed) | 1.79 ms (82% of copy speed) |
| Small ops | 831 → 224 | 2.26 ms | 0.73 ms |
| Attention | 72 → 72 | 0.76 ms | 0.79 ms |

![Decode step before and after kernel fusion](benchmark/fusion.svg)

In the trace, one transformer layer drops from 44 kernels in 205 µs to 16 kernels in 109 µs. Per projection after merging:

| Projection | Rows x cols | Blocks | µs | GB/s |
|---|---|---|---|---|
| qkv | 1152 x 896 | 144 | 5.50 | 189 |
| o | 896 x 896 | 112 | 4.96 | 163 |
| gate_up | 9728 x 896 | 1216 | 28.00 | 313 |
| down | 896 x 4864 | 112 | 14.58 | 301 |
| output layer | 151936 x 896 | 18992 | 359.18 | 381 |

q, k and v now take 5.5 µs together instead of 12.1 µs, and gate_up takes 28.0 µs instead of 33.7 µs for gate and up separately. The weight matmuls are now over half of each step, which is where a memory-bound model should spend its time. What is left:

- **Attention** (0.79 ms, 24%) still runs as cuBLAS batched matmuls and a separate softmax for every layer, over the whole preallocated cache. A fused decode attention kernel is the next target. In the batch engine, every request attends over its full 2048 token slot, which is why a lone request there now runs 31% slower than the single request path.
- **The INT8 kernel** reaches 82% of copy speed on average, but o and down use only 112 blocks on 28 SMs and run at 163 to 301 GB/s.
- **The gap between steps** (~0.3 to 0.5 ms per token): picking the token on the CPU with `.item()` before the next replay.

The same step with and without the CUDA graph, after fusion:

![INT8 decode timeline](benchmark/timeline.svg)

## Serving

The engine is served through a FastAPI app with token level streaming and a browser chat UI. On a GPU, requests go through a continuous batching engine, with INT8 weights when the kernel builds and FP32 otherwise. On CPU, a lock runs them one at a time on FP32 weights.

```bash
uvicorn src.app:app --port 8000

# or with Docker
docker build -t qwen-inference .
docker run -p 8000:8000 qwen-inference
```

### Continuous batching

Before batching, the GPU served one request at a time, so 8 clients meant the last one waited behind 7 others. [`BatchEngine`](src/batching.py) runs a background thread that owns the GPU and decodes every active request together:

- Each request gets a slot in a preallocated KV cache (8 slots, 2048 tokens each). A new request is prefilled on its own, then joins the batch at the next step. Finished or disconnected requests free their slot right away, so queued requests don't wait for the whole batch.
- Each step is one CUDA graph replay over all active slots, using the same batched `decode_step` as the single request path. Graphs are captured once at startup for batch sizes 1, 2, 4 and 8, and each step uses the smallest one that covers the active slots. A single request doesn't pay for 8.
- Each request keeps its own temperature and top-p, since tokens are picked outside the graph.

Aggregate throughput, 100 greedy tokens per request, median of 3 runs ([`benchmark/bench_batching.py`](benchmark/bench_batching.py)):

| Weights | Concurrent requests | One at a time (tok/s) | Batched (tok/s) | Speedup |
|---|---|---|---|---|
| FP32 | 1 | 135.6 | 116.3 | 0.86x |
| FP32 | 4 | 136.9 | 369.2 | 2.72x |
| FP32 | 8 | 134.7 | 475.2 | 3.55x |
| INT8 | 1 | 281.5 | 195.5 | 0.70x |
| INT8 | 4 | 267.1 | 416.3 | 1.56x |
| INT8 | 8 | 263.9 | **520.5** | 1.98x |

Decoding is limited by reading the weights from memory, and a batch reads them once for all its requests, so throughput grows with batch size. INT8 gains less from batching because each step already reads 4x fewer weight bytes, but it stays ahead of FP32 at every load (1.68x for one request, 1.10x for eight) with a quarter of the weight memory. A lone request is about 31% slower than the dedicated single request path because it attends over the full 2048 token cache slot, a fixed cost that weighs more as the rest of the step gets faster.

### Load test

All requests fired simultaneously via `asyncio.gather` ([`benchmark/bench_app.py`](benchmark/bench_app.py)). GTX 1080 Ti, FP32 weights, up to 100 tokens per request, median of 3 runs per column:

| Clients | One at a time p50 (s) | One at a time p95 (s) | One at a time req/s | Batched p50 (s) | Batched p95 (s) | Batched req/s |
|---|---|---|---|---|---|---|
| 1 | 0.77 | 0.77 | 1.30 | 1.53 | 1.53 | 0.65 |
| 4 | 2.50 | 2.68 | 1.49 | 2.00 | 2.00 | 2.00 |
| 8 | 5.01 | 6.77 | 1.18 | 2.90 | 2.91 | 2.75 |

With batching, p95 at 8 clients drops from 6.77 s to 2.91 s and p50 matches p95, since nobody waits in line. Serving INT8 before the output layer was quantized brings that to 2.78 s, and 1 and 4 clients to 1.22 s and 1.81 s p50 (2.87 req/s at 8 clients). Single client numbers swing a lot between runs (0.47 to 1.54 s) because requests sample at temperature 0.7 and stop at end of sequence, so reply length changes from run to run. The token based benchmark above is the cleaner comparison.

CPU, 50-token generations:

| Clients | p50 (s) | p95 (s) | Wall clock (s) | Req/s |
|---|---|---|---|---|
| 1 | 10.28 | 10.28 | 10.28 | 0.10 |
| 4 | 58.13 | 77.89 | 77.89 | 0.05 |
| 8 | 66.44 | 118.90 | 118.90 | 0.07 |

On CPU, latency degrades roughly linearly with concurrency because a lock runs one request at a time, so at 8 clients the last request waits behind the other 7, pushing p95 to roughly 2x p50.

## Usage

```bash
git clone https://github.com/BharatJassal/Qwen_LLM_Inference.git
cd Qwen_LLM_Inference
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
# NVIDIA GPU only: install the CUDA build of PyTorch first
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python download_model.py
```

```bash
python -m tests.test_HF
python -m tests.test_generate
python -m tests.test_gate2
python -m tests.test_sample
python -m benchmark.benchmark
```

`test_HF` diffs RMSNorm/attention/MLP/logits against the HF reference. `test_generate` exercises naive greedy generation end to end and prints the decoded output.
`test_gate2` checks that cached and uncached decoding produce token for token identical output.
`test_sample` checks sampling degenerates to greedy decoding at `temperature=0`. The benchmark times 30, 150, 300 token requests, cached and uncached, for the prompt set by `PROMPT` in `benchmark.py`.

### GPU kernel and CUDA graphs (NVIDIA only)

The CUDA extension (`cuda/bindings.cpp`, `int8_gemv_ext.cu`, `fused_ops.cu`) compiles on first use, so it needs the CUDA Toolkit 12.6, C++ compiler (Windows: Visual Studio 2022 Build Tools) and `ninja` (in `requirements.txt`). If the build fails, it prints a warning instead of crashing: INT8 falls back to dequantize, the fused ops to PyTorch, and the app to FP32. On Windows, run from a VS developer shell with the venv activated:

```powershell
& "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
.venv\Scripts\Activate.ps1
python -m tests.test_int8_kernel
python -m tests.test_fused_ops
python -m tests.test_graph
python -m tests.test_batching
python -m tests.test_ext_fallback
python -m benchmark.bench_int8_speed
python -m benchmark.bench_int8_batch
python -m benchmark.bench_batching
python -m benchmark.bench_bandwidth
```

Kernel timeline with nvprof (ships with CUDA 12.6, supports Pascal). On Windows, nvprof has to launch the real interpreter rather than the venv's `python.exe` launcher, so the venv is passed through `PYTHONPATH`:

```powershell
$env:PATH = "$env:CUDA_PATH\extras\CUPTI\lib64;$env:PATH"
$env:PYTHONPATH = "$PWD;$PWD\.venv\Lib\site-packages"
$py = (python -c "import sys; print(sys._base_executable)")
foreach ($m in "cached", "graph") { nvprof --profile-from-start off --print-gpu-trace --csv -u us --log-file benchmark/trace_$m.csv $py -m benchmark.profile_trace $m }
python -m benchmark.analyze_trace
```

Passing a graph trace recorded on an older commit, `python -m benchmark.analyze_trace path/to/trace_graph.csv`, also draws `benchmark/fusion.svg` comparing it with the current one.

On Volta or newer, or from an administrator terminal, Nsight Systems records the same run with `nsys profile -t cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi -o benchmark/decode python -m benchmark.profile_trace graph` and opens in `nsys-ui` (not tested here, nsys needs admin rights on this machine).

`test_int8_kernel` checks the kernel against dequantize + matmul on every projection shape at batch 1 to 8 and with a bias, that a batched row matches the same row run alone bit for bit, and that decoding with all custom kernels gives the same tokens as plain PyTorch. `test_fused_ops` checks the RMSNorm, RoPE and SiLU kernels against their PyTorch versions and that bad shapes are rejected. `test_ext_fallback` fakes a failed build and checks INT8 layers and the fused ops still work through PyTorch. `test_graph` checks graph decoding (greedy and seeded sampling, FP32 and INT8) is token for token identical to the cached path. `test_batching` runs 6 greedy requests through a 4 slot engine, FP32 and INT8, and checks each one matches `greedy_cached`, then checks over-long requests are rejected and cancelled requests free their slots. `bench_int8_speed` times FP32 and INT8 decoding, with and without CUDA graphs. `bench_int8_batch` times the kernel at batch 1 to 8 against dequantize and cuBLAS. `bench_batching` compares batched and one at a time throughput, FP32 and INT8. `bench_bandwidth` measures achieved memory bandwidth per decode step against a device copy and breaks the step's time down by kernel type. `analyze_trace` turns the nvprof traces into the step and per-projection tables and `benchmark/timeline.svg`.

The standalone kernel harness in [`cuda/int8_gemv.cu`](cuda/int8_gemv.cu) checks the kernel against a CPU reference and times it against cuBLAS (`-arch=sm_61` targets the GTX 1080 Ti; change it for your GPU):

```powershell
cd cuda
nvcc -O3 -arch=sm_61 -lcublas -o int8_gemv.exe int8_gemv.cu
.\int8_gemv.exe
```

```python
from transformers import AutoTokenizer
from src.loader import load_weights
from src.config import ModelConfig
from src.generate import greedy_cached

tok = AutoTokenizer.from_pretrained("model")
weights = load_weights("model/model.safetensors")
cfg = ModelConfig.from_json("model/config.json")

messages = [{"role": "user", "content": "Write a story about the ocean."}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").input_ids

out = greedy_cached(ids, weights, cfg, max_new_tokens=50)
print(tok.decode(out[0]))
```

On an NVIDIA GPU, load the weights and prompt onto CUDA, optionally quantize, and use the graph decoder:

```python
from src.quantize import quantize_model
from src.generate import greedy_graph

weights = quantize_model(load_weights("model/model.safetensors", "cuda"))
out = greedy_graph(ids.to("cuda"), weights, cfg, max_new_tokens=50)
print(tok.decode(out[0]))
```

## References
- Qwen2.5 technical report / model card
- Tokenizer is reused from HuggingFace `AutoTokenizer`. This project implements the model.
- `transformers` appears in requirements as the verification reference and tokenizer.

## License

MIT License © 2026 Bharat Jassal
Contact: jassal.bharat18@gmail.com