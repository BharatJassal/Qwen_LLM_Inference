import time
from transformers import AutoTokenizer
from src.loader import load_weights
from src.config import ModelConfig
from src.generate import greedy, greedy_cached

MODEL_DIR = "model"
LENGTHS = [30, 150, 300]
PROMPT = "Write a story about the ocean."
RUN_NAIVE = True


def build_ids(tok):
    messages = [{"role": "user", "content": PROMPT}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt").input_ids


def time_generation(fn, ids, weights, config, n):
    start = time.time()
    out = fn(ids, weights, config, max_new_tokens=n)
    elapsed = time.time() - start
    generated = out.shape[1] - ids.shape[1]
    return elapsed, generated


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    weights = load_weights("model/model.safetensors")
    cfg = ModelConfig.from_json("model/config.json")
    ids = build_ids(tok)

    rows = []
    for n in LENGTHS:
        if RUN_NAIVE:
            naive_s, naive_n = time_generation(greedy, ids, weights, cfg, n)
            naive_tps = naive_n / naive_s
        else:
            naive_s, naive_tps = float("nan"), float("nan")

        cached_s, cached_n = time_generation(greedy_cached, ids, weights, cfg, n)
        cached_tps = cached_n / cached_s

        speedup = naive_s / cached_s if RUN_NAIVE else float("nan")
        rows.append((cached_n, naive_s, cached_s, naive_tps, cached_tps, speedup))

    header = f"{'tokens':>7} | {'naive s':>9} | {'cached s':>9} | {'naive t/s':>10} | {'cached t/s':>11} | {'speedup':>8}"
    print(header)
    for n, ns, cs, ntps, ctps, sp in rows:
        print(f"{n:>7} | {ns:>9.2f} | {cs:>9.2f} | {ntps:>10.2f} | {ctps:>11.2f} | {sp:>7.1f}x")

    print("| Generated tokens | Naive (s) | Cached (s) | Naive tok/s | Cached tok/s | Speedup |")
    print("|---|---|---|---|---|---|")
    for n, ns, cs, ntps, ctps, sp in rows:
        print(f"| {n} | {ns:.2f} | {cs:.2f} | {ntps:.2f} | {ctps:.1f} | {sp:.1f}x |")


if __name__ == "__main__":
    main()