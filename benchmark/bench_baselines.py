import json
import statistics
import subprocess
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.loader import load_weights
from src.config import ModelConfig
from src.generate import greedy_graph
from src.quantize import quantize_model

# same prompt and length as bench_int8_speed, one request, greedy
MODEL_DIR = "model"
NEW_TOKENS = 300
RUNS = 3
ENGINE, OTHER = "#2a78d6", "#8a8983"


def timed(fn):
    fn(5)  # warm-up
    rates = []
    for _ in range(RUNS):
        torch.cuda.synchronize()
        start = time.time()
        n = fn(NEW_TOKENS)
        torch.cuda.synchronize()
        rates.append(n / (time.time() - start))
    return statistics.median(rates)


def llama_cpp(bench_exe, gguf):
    out = subprocess.run([bench_exe, "-m", gguf, "-ngl", "99", "-p", "0", "-n", str(NEW_TOKENS),
                          "-r", str(RUNS), "-fa", "1", "-o", "json"], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)[0]["avg_ts"]


def chart(results, out):
    rows = sorted(results.items(), key=lambda r: -r[1])
    x0, width, top, row = 250, 420, 56, 38
    scale = width / rows[0][1]
    height = top + row * len(rows) + 30
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 {height}" font-family="sans-serif" font-size="13">',
             f'<rect width="760" height="{height}" fill="#fcfcfb"/>',
             '<text x="16" y="26" font-size="15" font-weight="bold" fill="#0b0b0b">Decode speed, Qwen2.5-0.5B on a GTX 1080 Ti</text>',
             f'<text x="16" y="44" fill="#52514e">tok/s, one request, {NEW_TOKENS} generated tokens</text>']
    for i, (name, rate) in enumerate(rows):
        y = top + i * row
        w = rate * scale
        color = ENGINE if name.startswith("this engine") else OTHER
        parts.append(f'<text x="{x0 - 10}" y="{y + 19}" text-anchor="end" fill="#0b0b0b">{name}</text>')
        # rounded data end, square end on the baseline
        parts.append(f'<path d="M{x0},{y + 4} h{w - 4:.1f} a4,4 0 0 1 4,4 v12 a4,4 0 0 1 -4,4 h{4 - w:.1f} z" fill="{color}"/>')
        parts.append(f'<text x="{x0 + w + 8:.1f}" y="{y + 19}" fill="#0b0b0b" font-weight="bold">{rate:.0f}</text>')
    parts.append(f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{top + row * len(rows) - 10}" stroke="#52514e"/>')
    parts.append(f'<text x="16" y="{height - 12}" font-size="11" fill="#52514e">Median of {RUNS} runs. '
                 'llama.cpp: llama-bench, official CUDA 12.4 build, Q8_0 GGUF, flash attention on.</text>')
    parts.append("</svg>")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    cfg = ModelConfig.from_json("model/config.json")
    msgs = [{"role": "user", "content": "Write a story about the ocean."}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.to("cuda")

    results = {}
    for dtype in (torch.float32, torch.float16):
        hf = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=dtype).to("cuda").eval()

        def run(n):
            out = hf.generate(ids, max_new_tokens=n, min_new_tokens=n, do_sample=False,
                              attention_mask=torch.ones_like(ids), pad_token_id=tok.eos_token_id)
            return out.shape[1] - ids.shape[1]
        with torch.no_grad():
            results[f"Hugging Face transformers, {'FP32' if dtype == torch.float32 else 'FP16'}"] = timed(run)
        del hf
        torch.cuda.empty_cache()

    fp32 = load_weights("model/model.safetensors", "cuda")
    for name, w in [("this engine, FP32", fp32), ("this engine, INT8", quantize_model(fp32))]:
        results[name] = timed(lambda n: greedy_graph(ids, w, cfg, max_new_tokens=n).shape[1] - ids.shape[1])

    # optional: python -m benchmark.bench_baselines path/to/llama-bench.exe path/to/model-q8_0.gguf
    if len(sys.argv) == 3:
        results["llama.cpp, Q8_0"] = llama_cpp(*sys.argv[1:])

    for name, rate in results.items():
        print(f"{name:>32}: {rate:6.1f} tok/s")
    chart(results, "benchmark/baselines.svg")


if __name__ == "__main__":
    main()
