import time
import threading
import torch
from transformers import AutoTokenizer
from src.loader import load_weights
from src.config import ModelConfig
from src.generate import greedy_cached
from src.batching import BatchEngine
from src.quantize import quantize_model

assert torch.cuda.is_available(), "batching needs a GPU"
tok = AutoTokenizer.from_pretrained("model")
cfg = ModelConfig.from_json("model/config.json")
fp32 = load_weights("model/model.safetensors", "cuda")
int8 = quantize_model(fp32)

prompts = [
    "Write a story about the ocean.",
    "Explain how a CPU cache works.",
    "List five fruits.",
    "Write a long story multiple paragraphs about a robot who learns love",
    "What is the capital of France?",
    "Describe a sunset in detail.",
]
ids = [tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                   add_generation_prompt=True), return_tensors="pt").input_ids.cuda()
       for p in prompts]
N = 80

# 6 requests on 4 slots, so two have to wait and join mid-batch
for name, weights in [("fp32", fp32), ("int8", int8)]:
    engine = BatchEngine(weights, cfg, slots=4)
    results = [None] * len(ids)
    def run(i):
        results[i] = list(engine.submit(ids[i], N, 0.0, 1.0))
    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(ids))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for i, x in enumerate(ids):
        ref = greedy_cached(x, weights, cfg, max_new_tokens=N)[0, x.shape[1]:].tolist()
        assert results[i] == ref, f"{name} prompt {i}: batched output differs from greedy_cached"
    print(f"{name}: {len(ids)} batched requests match greedy_cached")

try:
    engine.submit(ids[0], 5000, 0.0, 1.0)
    raise AssertionError("over-long request was accepted")
except ValueError:
    print("over-long request rejected")

streams = [engine.submit(ids[0], 1500, 0.8, 0.9) for _ in range(4)]
for s in streams:
    next(s)
    s.close()
deadline = time.time() + 5
while any(engine.slots) and time.time() < deadline:
    time.sleep(0.05)
assert not any(engine.slots), "cancelled requests still hold slots"
print("cancelled requests freed their slots")
print("Batching Pass.")
