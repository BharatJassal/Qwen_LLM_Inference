# Qwen LLM Inference

Built an inference engine for **Qwen2.5-0.5B-Instruct** from scratch, developed in pure PyTorch by loading raw `safetensors` weights directly, without using `transformers`. Implemented the Llama architecture (RoPE, GQA, SwiGLU, RMSNorm), autoregressive decoding with sampling, and KV caching, verifying each component against HuggingFace. On NVIDIA GPUs, decoding runs a custom fused INT8 GEMV CUDA kernel and replays each token from a CUDA graph: **195.7 tok/s INT8 on a GTX 1080 Ti, 1.8x FP32**. The server batches concurrent requests continuously on INT8 weights, reaching 466 tok/s combined across 8 requests.


## Purpose

Modern LLM stacks such as vLLM, llama.cpp and more hide the complexity involved in running a model. This project rebuilds that pipeline from the ground up, to better understand how it works. From mapping weights to computations, understanding how KV caching speeds up generation by avoiding redundant computations.

## Results

| Check | Method | Result |
|---|---|---|
| End to end logits | Max abs. deviation vs HuggingFace reference, fp32 | 6.08e-05, gate: < 1e-3 |
| Layer components | Forward hooks on reference model, layer 0 | RMSNorm, attention, MLP all < 1e-4 |
| KV cache | Greedy decode, cached vs uncached, 300 token generation | Token-for-token identical |
| INT8 CUDA kernel | Every layer-0 projection, batch 1 to 8, vs dequantize + matmul; 200 token greedy decode | Max rel. error 4.1e-07; token-for-token identical |
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

## GPU: fused INT8 kernel and CUDA graphs

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

With continuous batching, each step multiplies every weight matrix by up to 8 vectors, one per active request. Running the kernel 8 times would read the weights 8 times, so the kernel takes a batch: each warp loads a weight row once and accumulates it against every input, with the batch size as a template parameter so the 8 partial sums stay in registers. Each row's arithmetic is identical to a batch of 1, so a request gets bit-identical results whether it runs alone or in a batch.

µs per call, median of 3 runs ([`benchmark/bench_int8_batch.py`](benchmark/bench_int8_batch.py)):

| Shape | Batch | INT8 kernel | Dequantize + matmul | cuBLAS FP32 |
|---|---|---|---|---|
| gate/up | 1 | 18.14 | 241.74 | 55.94 |
| gate/up | 8 | 70.11 | 263.26 | 79.63 |
| down | 1 | 18.27 | 252.73 | 53.80 |
| down | 8 | 62.24 | 288.55 | 101.59 |
| q/o | 8 | 13.45 | 64.59 | 22.81 |

The INT8 advantage shrinks as the batch grows: at batch 8 the kernel is doing 8x the math on each weight byte and stops being purely memory bound. It still beats FP32 at every size, and is about 4x faster than dequantizing, which is what batched INT8 used before.

### CUDA graphs

A 3x faster kernel barely changed end-to-end speed at first: INT8 decode was still slower than FP32. Profiling showed why. Each token launches ~1100 small GPU operations, and the GPU sat idle about half the time waiting for Python to launch the next one.

[`greedy_graph` / `stream_graph`](src/generate.py) process the prompt normally, then record one decode step ([`decode_step`](src/model.py)) as a CUDA graph and replay it once per token, one launch instead of ~1100. A graph needs fixed shapes and memory addresses, so the decode step:

- uses a preallocated KV cache, writing each new token's K/V in place at its position and masking slots not yet written (the regular path grows the cache with `torch.cat`);
- builds the RoPE tables once on the GPU and indexes them with a position tensor that also lives on the GPU;
- leaves choosing the next token (`argmax` or top-p `sample`) outside the graph, so sampling behaves exactly as on the regular path.

### End-to-end decode (300 tokens, median of 3 runs)

| Weights | Cached decode (tok/s) | Cached + CUDA graph (tok/s) | Graph speedup |
|---|---|---|---|
| FP32 | 58.0 | 108.4 | 1.9x |
| INT8 kernel, FP32 output layer (1 run) | 54.1 | 161.5 | 3.0x |
| INT8 kernel, INT8 output layer | 60.0 | **195.7** | 3.3x |

Timings include prompt processing and graph capture. With launch overhead gone, the kernel's savings show end to end: INT8 is 1.8x faster than FP32. Dequantizing on every call instead of using the kernel ran at about 30 tok/s.

Quantizing the output layer cut the weight bytes read per token from ~904 MB to ~496 MB (1.82x), but speed rose only 1.21x. At 195.7 tok/s the GPU streams ~97 GB/s of weights, about 20% of its 484 GB/s peak, so weight bytes are no longer the main cost per token.

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
| FP32 | 1 | 105.9 | 96.9 | 0.92x |
| FP32 | 4 | 105.8 | 317.2 | 3.00x |
| FP32 | 8 | 105.8 | 442.2 | 4.19x |
| INT8 | 1 | 185.2 | 151.1 | 0.81x |
| INT8 | 4 | 178.7 | 360.4 | 2.02x |
| INT8 | 8 | 177.0 | **466.0** | 2.62x |

Decoding is limited by reading the weights from memory, and a batch reads them once for all its requests, so throughput grows with batch size. INT8 gains less from batching because each step already reads 4x fewer weight bytes, but it stays ahead of FP32 at every load (1.56x for one request, 1.05x for eight) with a quarter of the weight memory. A lone request is about 18% slower than the dedicated single request path because it attends over the full 2048 token cache slot, a fixed cost that weighs more now that the rest of the step is faster.

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

The INT8 extension compiles on first use, so it needs the CUDA Toolkit 12.6, C++ compiler (Windows: Visual Studio 2022 Build Tools) and `ninja` (in `requirements.txt`). If the build fails, it prints a warning and INT8 falls back to dequantize (and the app to FP32) instead of crashing. On Windows, run from a VS developer shell with the venv activated:

```powershell
& "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
.venv\Scripts\Activate.ps1
python -m tests.test_int8_kernel
python -m tests.test_graph
python -m tests.test_batching
python -m tests.test_ext_fallback
python -m benchmark.bench_int8_speed
python -m benchmark.bench_int8_batch
python -m benchmark.bench_batching
```

`test_int8_kernel` checks the kernel against dequantize + matmul on every projection shape at batch 1 to 8, that a batched row matches the same row run alone bit for bit, and that decoding with and without it gives identical tokens. `test_ext_fallback` fakes a failed build and checks INT8 layers still work through dequantize. `test_graph` checks graph decoding (greedy and seeded sampling, FP32 and INT8) is token for token identical to the cached path. `test_batching` runs 6 greedy requests through a 4 slot engine, FP32 and INT8, and checks each one matches `greedy_cached`, then checks over-long requests are rejected and cancelled requests free their slots. `bench_int8_speed` times FP32 and INT8 decoding, with and without CUDA graphs. `bench_int8_batch` times the kernel at batch 1 to 8 against dequantize and cuBLAS. `bench_batching` compares batched and one at a time throughput, FP32 and INT8.

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