# Qwen LLM Inference

Built a inference engine for **Qwen2.5-0.5B-Instruct** from scratch, developed in pure PyTorch by loading raw `safetensors` weights directly, without using `transformers`. Implemented the Llama architecture (RoPE, GQA, SwiGLU, RMSNorm), autoregressive decoding with sampling, and KV caching, verifying each component against HuggingFace.


## Purpose

Modern LLM stacks such as vLLM, llama.cpp and more hide the complexity involved in running a model. This project rebuilds that pipeline from the ground, to better understand how it works. From mapping weights to computations, understanding how KV caching speeds up generation by avoiding redundant computations.

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

## Usage

```bash
pip install -r requirements.txt
python download_model.py

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

## Notes/References
- Qwen2.5 technical report / model card
- Tokenizer is reused from HuggingFace `AutoTokenizer`. This project implements the model.
- `transformers` appears in requirements as the verification reference and tokenizer.

## License

MIT License © 2026 Bharat Jassal
Contact: jassal.bharat18@gmail.com