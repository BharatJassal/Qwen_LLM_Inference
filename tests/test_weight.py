import torch
from transformers import AutoTokenizer
from src.loader import load_weights
from src.config import ModelConfig
from src.generate import greedy_cached
from src.quantize import quantize_model

MODEL_DIR = "model"
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
weights = load_weights("model/model.safetensors", device)
cfg = ModelConfig.from_json("model/config.json")

q_weights = quantize_model(weights)

messages = [{"role": "user", "content": "Write a story about the ocean."}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").input_ids.to(device)

out_fp32 = greedy_cached(ids, weights, cfg, max_new_tokens=50)
out_int8 = greedy_cached(ids, q_weights, cfg, max_new_tokens=50)

print("fp32:", tok.decode(out_fp32[0][ids.shape[1]:]))
print("int8:", tok.decode(out_int8[0][ids.shape[1]:]))