import sys
import torch
from transformers import AutoTokenizer
from src.loader import load_weights
from src.config import ModelConfig
from src.generate import greedy_cached, greedy_graph
from src.quantize import quantize_model

# python -m benchmark.profile_trace cached|graph, run under nvprof or nsys (see README)
decode = {"cached": greedy_cached, "graph": greedy_graph}[sys.argv[1]]
tok = AutoTokenizer.from_pretrained("model")
cfg = ModelConfig.from_json("model/config.json")
weights = quantize_model(load_weights("model/model.safetensors", "cuda"))

msgs = [{"role": "user", "content": "Write a story about the ocean."}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").input_ids.to("cuda")

decode(ids, weights, cfg, max_new_tokens=5)
torch.cuda.synchronize()
# only the measured run is recorded, not loading or warm-up
torch.cuda.profiler.start()
decode(ids, weights, cfg, max_new_tokens=20)
torch.cuda.synchronize()
torch.cuda.profiler.stop()
