import torch
from transformers import AutoTokenizer
from src.loader import load_weights
from src.config import ModelConfig
from src.generate import greedy_cached, generate_sample

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "model"
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
weights = load_weights("model/model.safetensors", device)
cfg = ModelConfig.from_json("model/config.json")

messages = [{"role": "user", "content": "Write a long story multiple paragraphs about a robot who learns love"}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").input_ids.to(device)
N = 30

for i in range(3):
    out = generate_sample(ids, weights, cfg, max_new_tokens=N, temperature=0.8, top_p=0.9)
    print(tok.decode(out[0, ids.shape[1]:]))

out_greedy = greedy_cached(ids, weights, cfg, max_new_tokens=N)
out_sampled_zero = generate_sample(ids, weights, cfg, max_new_tokens=N, temperature=0.0, top_p=0.9)

print("Greedy Output:")
print(tok.decode(out_greedy[0, ids.shape[1]:]))

if torch.equal(out_greedy, out_sampled_zero):
    print("Degenerate Pass.")
else:
    print("Degenerate Fail.")