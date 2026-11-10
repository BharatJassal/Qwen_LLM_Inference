import time, torch
from transformers import AutoTokenizer
from src.loader import load_weights
from src.config import ModelConfig
from src.generate import greedy_cached
from src.quantize import quantize_model

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("model")
cfg = ModelConfig.from_json("model/config.json")
fp32 = load_weights("model/model.safetensors", device)
int8 = quantize_model(fp32)

msgs = [{"role": "user", "content": "Write a story about the ocean."}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").input_ids.to(device)

for name, w in [("fp32", fp32), ("int8", int8)]:
    greedy_cached(ids, w, cfg, max_new_tokens=5)  # warm-up
    torch.cuda.synchronize()
    start = time.time()
    out = greedy_cached(ids, w, cfg, max_new_tokens=300)
    torch.cuda.synchronize()
    n = out.shape[1] - ids.shape[1]
    print(f"{name}: {n / (time.time() - start):.1f} tok/s")