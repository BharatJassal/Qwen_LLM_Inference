import torch
from torch.profiler import profile, ProfilerActivity
from torch.autograd import DeviceType
from transformers import AutoTokenizer
from src.loader import load_weights
from src.config import ModelConfig
from src.generate import greedy_cached

N = 20
tok = AutoTokenizer.from_pretrained("model")
cfg = ModelConfig.from_json("model/config.json")
weights = load_weights("model/model.safetensors", "cuda")

msgs = [{"role": "user", "content": "Write a story about the ocean."}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").input_ids.to("cuda")

greedy_cached(ids, weights, cfg, max_new_tokens=5)
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    out = greedy_cached(ids, weights, cfg, max_new_tokens=N)
    torch.cuda.synchronize()

gpu_events = [e for e in prof.events() if e.device_type == DeviceType.CUDA]
gpu_ms = sum(e.device_time_total for e in gpu_events) / 1000
n = out.shape[1] - ids.shape[1]

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=12))
print(f"GPU operations per token: {len(gpu_events) / n:.0f}")
print(f"GPU busy time per token:  {gpu_ms / n:.2f} ms")
print(f"Real time per token:      {1000 / 49.2:.2f} ms (from your benchmark)")
prof.export_chrome_trace("trace.json")