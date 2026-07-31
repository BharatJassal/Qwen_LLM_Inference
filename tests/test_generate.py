import time
import torch
from transformers import AutoTokenizer
from src.loader import load_weights
from src.config import ModelConfig
from src.generate import greedy

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "model"
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
weights = load_weights("model/model.safetensors", device)
cfg = ModelConfig.from_json("model/config.json")

messages = [{"role": "user", "content": "Write a long story multiple paragraphs about a robot who learns love"}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").input_ids.to(device)
print(f"Prompt text:\n{text}\n")

start_time = time.time()
out_ids = greedy(ids, weights, cfg, max_new_tokens=500)
end_time = time.time()

final_text = tok.decode(out_ids[0])
print(final_text)

num_generated = out_ids.shape[1] - ids.shape[1]
duration = end_time - start_time
print(f"\nGenerated {num_generated} tokens in {duration:.2f} seconds.")
print(f"Throughput: {num_generated / duration:.2f} tokens/second.")