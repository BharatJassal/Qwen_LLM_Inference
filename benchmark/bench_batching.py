import time
import threading
import torch
from transformers import AutoTokenizer
from src.loader import load_weights
from src.config import ModelConfig
from src.generate import greedy_graph
from src.batching import BatchEngine
from src.quantize import quantize_model

N = 100
tok = AutoTokenizer.from_pretrained("model")
cfg = ModelConfig.from_json("model/config.json")
fp32 = load_weights("model/model.safetensors", "cuda")
int8 = quantize_model(fp32)

msgs = [{"role": "user", "content": "Write a long story multiple paragraphs about a robot who learns love"}]
ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
          return_tensors="pt").input_ids.cuda()


def one_at_a_time(weights, n):
    start = time.time()
    tokens = sum(greedy_graph(ids, weights, cfg, N).shape[1] - ids.shape[1] for _ in range(n))
    return tokens, time.time() - start


def batched(engine, n):
    counts = [0] * n
    def run(i):
        counts[i] = len(list(engine.submit(ids, N, 0.0, 1.0)))
    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return sum(counts), time.time() - start


print(f"{'weights':>7} | {'requests':>8} | {'one at a time tok/s':>19} | {'batched tok/s':>13} | {'speedup':>7}")
for name, weights in [("fp32", fp32), ("int8", int8)]:
    engine = BatchEngine(weights, cfg, slots=8)
    one_at_a_time(weights, 1)
    batched(engine, 1)
    for n in [1, 4, 8]:
        t1, s1 = one_at_a_time(weights, n)
        t2, s2 = batched(engine, n)
        print(f"{name:>7} | {n:>8} | {t1 / s1:>19.1f} | {t2 / s2:>13.1f} | {(t2 / s2) / (t1 / s1):>6.2f}x")
