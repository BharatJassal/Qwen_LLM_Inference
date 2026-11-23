import torch
from transformers import AutoTokenizer
from src.loader import load_weights
from src.config import ModelConfig
from src.generate import greedy_cached
from src.quantize import quantize_model, get_weight
from src import int8_ext, layers

assert int8_ext.available(), "needs CUDA (extension builds from cuda/int8_gemv_ext.cu)"
device = "cuda"
cfg = ModelConfig.from_json("model/config.json")
weights = quantize_model(load_weights("model/model.safetensors", device))

# kernel vs dequantize + matmul on each projection in layer 0
torch.manual_seed(0)
for name in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
    key = f"model.layers.0.{name}.weight"
    w_int8, scales = weights[key]
    x = torch.randn(w_int8.shape[1], device=device)
    ref = get_weight(weights, key) @ x
    got = int8_ext.int8_gemv(w_int8, scales, x)
    rel = ((got - ref).abs().max() / ref.abs().max()).item()
    print(f"{name:18s} {tuple(w_int8.shape)}  max rel err {rel:.2e}")
    assert rel < 1e-5, f"{name} kernel mismatch"

    for b in range(2, int8_ext.MAX_BATCH + 1):
        xb = torch.randn(b, w_int8.shape[1], device=device)
        got = int8_ext.int8_gemv(w_int8, scales, xb)
        ref = xb @ get_weight(weights, key).T
        assert ((got - ref).abs().max() / ref.abs().max()).item() < 1e-5, f"{name} batch {b} mismatch"
        # each row should match running it alone exactly
        alone = torch.stack([int8_ext.int8_gemv(w_int8, scales, r.contiguous()) for r in xb])
        assert torch.equal(got, alone), f"{name} batch {b} rows differ from single row results"
print(f"batched kernel matches dequantize for batch 2 to {int8_ext.MAX_BATCH}")

for bad in [torch.randn(int8_ext.MAX_BATCH + 1, 896, device=device), torch.randn(3, 900, device=device)]:
    try:
        int8_ext.int8_gemv(*weights["model.layers.0.self_attn.q_proj.weight"], bad)
        raise AssertionError(f"kernel accepted x of shape {tuple(bad.shape)}")
    except RuntimeError:
        pass
print("bad batch shapes rejected")

# decoding with and without the kernel should give the same tokens
tok = AutoTokenizer.from_pretrained("model")
msgs = [{"role": "user", "content": "Write a story about the ocean."}]
ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
          return_tensors="pt").input_ids.to(device)
N = 200
out_kernel = greedy_cached(ids, weights, cfg, max_new_tokens=N)
layers.USE_FUSED = False
out_dequant = greedy_cached(ids, weights, cfg, max_new_tokens=N)
layers.USE_FUSED = True

assert torch.equal(out_kernel, out_dequant), "kernel and dequantize decodes diverge"
print(f"INT8 kernel decode matches dequantize path for {out_kernel.shape[1] - ids.shape[1]} tokens. Pass.")
