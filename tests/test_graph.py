import torch
from transformers import AutoTokenizer
from src.loader import load_weights
from src.config import ModelConfig
from src.generate import greedy_cached, greedy_graph, sample, stream_sample
from src.model import forward
from src.quantize import quantize_model

assert torch.cuda.is_available(), "CUDA graphs need a GPU"
device = "cuda"
tok = AutoTokenizer.from_pretrained("model")
cfg = ModelConfig.from_json("model/config.json")
fp32 = load_weights("model/model.safetensors", device)
int8 = quantize_model(fp32)

msgs = [{"role": "user", "content": "Write a long story multiple paragraphs about a robot who learns love"}]
ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
          return_tensors="pt").input_ids.to(device)
N = 300

for name, w in [("fp32", fp32), ("int8", int8)]:
    ref = greedy_cached(ids, w, cfg, max_new_tokens=N)
    out = greedy_graph(ids, w, cfg, max_new_tokens=N)
    assert torch.equal(ref, out), f"{name}: graph decode diverges from greedy_cached"
    print(f"{name}: graph decode matches greedy_cached for {out.shape[1] - ids.shape[1]} tokens")

# with the same seed, graph sampling should pick the same tokens as the cached loop
def sample_cached(ids, w, n, temperature, top_p):
    logits, cache = forward(ids, w, cfg)
    out = []
    t = sample(logits[0, -1, :], temperature, top_p)
    for _ in range(n):
        if t.item() == cfg.eos_token_id:
            break
        out.append(t.item())
        logits, cache = forward(t.view(1, 1), w, cfg, cache)
        t = sample(logits[0, -1, :], temperature, top_p)
    return out

for name, w in [("fp32", fp32), ("int8", int8)]:
    torch.manual_seed(0)
    ref = sample_cached(ids, w, 100, 0.8, 0.9)
    torch.manual_seed(0)
    got = list(stream_sample(ids, w, cfg, 100, 0.8, 0.9))
    assert got == ref, f"{name}: graph sampling diverges from cached sampling"
    print(f"{name}: graph sampling matches cached sampling for {len(got)} tokens (seed 0, T=0.8, top_p=0.9)")

# zero new tokens returns the prompt unchanged
assert torch.equal(greedy_graph(ids, fp32, cfg, max_new_tokens=0), ids)
print("Graph gate Pass.")
