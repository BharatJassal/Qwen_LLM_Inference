import time
import torch
from transformers import AutoTokenizer
from src.loader import load_weights
from src.config import ModelConfig
from src.generate import greedy, greedy_cached

MODEL_DIR = "model"
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
weights = load_weights("model/model.safetensors")
cfg = ModelConfig.from_json("model/config.json")

messages = [{"role": "user", "content": "Write a long story multiple paragraphs about a robot who learns love"}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").input_ids
N = 500

start_naive = time.time()
out_naive  = greedy(ids, weights, cfg, max_new_tokens=N)
end_naive = time.time()
print(f"Naive time: {end_naive - start_naive:.2f} seconds\n")

start_cached = time.time()
out_cached = greedy_cached(ids, weights, cfg, max_new_tokens=N)
end_cached = time.time()
print(f"Cached time: {end_cached - start_cached:.2f} seconds\n")

print("naive shape: ", out_naive.shape)
print("cached shape:", out_cached.shape)

assert torch.equal(out_naive, out_cached), "Gate 2 Fail sequences diverge"
print("Gate 2 Pass.")
print(tok.decode(out_cached[0, -N:]))