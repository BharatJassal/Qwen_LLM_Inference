# Qwen LLM Inference

Built an inference engine for **Qwen2.5-0.5B-Instruct** from scratch, developed in pure PyTorch by loading raw `safetensors` weights directly, without using `transformers`. Implemented the Llama architecture (RoPE, GQA, SwiGLU, RMSNorm), autoregressive decoding with sampling, and KV caching, verifying each component against HuggingFace.


## Purpose

Modern LLM stacks such as vLLM, llama.cpp and more hide the complexity involved in running a model. This project rebuilds that pipeline from the ground up, to better understand how it works. From mapping weights to computations, understanding how KV caching speeds up generation by avoiding redundant computations.

## Results

| Check | Method | Result |
|---|---|---|
| End to end logits | Max abs. deviation vs HuggingFace reference, fp32 | 6.08e-05, gate: < 1e-3 |
| Layer components | Forward hooks on reference model, layer 0 | RMSNorm, attention, MLP all < 1e-5 |
| KV cache | Greedy decode, cached vs uncached, 300 token generation | Token-for-token identical |

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

Linear layer weights are quantized using per channel symmetric INT8 and dequantized back to FP32 right before each matrix multiplication. Layer norms, biases, and embedding tables stay in FP32.

| Measurement | fp32 | int8 | Change |
|---|---|---|---|
| Quantized tensors | 1365.0 MB | 342.41 MB | 3.99x reduction |
| Whole model | 1884.59 MB | 862.00 MB | 2.19x reduction |
| Perplexity | 20.8487 | 20.8605 | +0.06% |

Measured on Project Gutenberg ebook 1342 (*Pride and Prejudice*).

The reason why the whole model reduction is only 2.19x instead of 4x is because `embed_tokens.weight` matrix 
(151936 x 896, ~519 MB in fp32) is left unquantized. Because the model uses tied embeddings, any quantization error here degrades the input token representation and the final logit prediction. At this model size the embedding is a large fraction of total parameters, but in larger models it is negligible and the whole model int8 approaches the full 4x.

Production engines use custom fused INT8 GEMM kernels to accelerate compute on hardware. Here, weights dequantize to FP32 right before `torch.matmul` giving you the full RAM savings while keeping the Pytorch implementation clean and dependency free.

## Serving

The engine is served through a FastAPI app with token level SSE streaming, an asyncio request queue, and a browser chat UI served at the root route.

```bash
uvicorn src.app:app --port 8000

# or with Docker
docker build -t qwen-inference .
docker run -p 8000:8000 qwen-inference
```

### Load test

50-token generations on CPU, all requests fired simultaneously via `asyncio.gather`.

| Clients | p50 (s) | p95 (s) | Wall clock (s) | Req/s |
|---|---|---|---|---|
| 1 | 10.28 | 10.28 | 10.28 | 0.10 |
| 4 | 58.13 | 77.89 | 77.89 | 0.05 |
| 8 | 66.44 | 118.90 | 118.90 | 0.07 |

Latency degrades roughly linearly with concurrency because an `asyncio.Lock` serializes generation. Only one request generates at a time, so at 8 clients the last request in line waits behind the other 7, pushing p95 to roughly 2x p50. Production engines solve this with continuous batching, where multiple sequences share each forward pass instead of queuing for the model.

## Usage

```bash
git clone https://github.com/BharatJassal/Qwen_LLM_Inference.git
cd Qwen_LLM_Inference
python -m venv .venv
source .venv/bin/activate
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

`test_HF` diffs RMSNorm/attention/MLP/logits against the HF reference.`test_generate` exercises naive greedy generation end to end and prints the decoded output.
`test_gate2` checks that cached and uncached decoding produce token for token identical output.
`test_sample` checks sampling degenerates to greedy decoding at `temperature=0`. The benchmark times 30, 150, 300 token requests, cached and uncached, for the prompt set by `PROMPT` in `benchmark.py`.

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

## References
- Qwen2.5 technical report / model card
- Tokenizer is reused from HuggingFace `AutoTokenizer`. This project implements the model.
- `transformers` appears in requirements as the verification reference and tokenizer.

## License

MIT License © 2026 Bharat Jassal
Contact: jassal.bharat18@gmail.com