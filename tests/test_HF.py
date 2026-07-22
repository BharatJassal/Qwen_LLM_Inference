import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.loader import load_weights
from src.config import ModelConfig
from src.layers import embedding, rmsnorm, rope_tables, apply_rope, attention, mlp
from src.model import forward

MODEL_DIR = "model"

ref = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
ref.eval()
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
weights = load_weights("model/model.safetensors")

cfg = ModelConfig.from_json("model/config.json")
ids = tok("The capital of Canada is", return_tensors="pt").input_ids
emb = embedding(ids, weights["model.embed_tokens.weight"])

mine_logits, _ = forward(ids, weights, cfg)
with torch.no_grad():
    ref_logits = ref(ids).logits

diff = torch.max(torch.abs(mine_logits - ref_logits))
print("max abs diff:", diff.item())
assert diff < 1e-3, "logits mismatch"
print("logits match")

attn_input = rmsnorm(
    emb,
    weights["model.layers.0.input_layernorm.weight"],
    eps=cfg.rms_norm_eps
)
mine_attn, _ = attention(attn_input, weights, 0, cfg)
capture = {}

def hook(module, input, output):
    capture["attn"] = output[0] if isinstance(output, tuple) else output

h = ref.model.layers[0].self_attn.register_forward_hook(hook)
with torch.no_grad():
    ref(ids)
h.remove()

diff = torch.max(torch.abs(mine_attn - capture["attn"]))
print("attention max abs diff:", diff.item())
assert diff < 1e-3, "attention mismatch"
print("attention matches")

h = emb + mine_attn
mlp_input = rmsnorm(
    h,
    weights["model.layers.0.post_attention_layernorm.weight"],
    eps=cfg.rms_norm_eps
)
mine_mlp = mlp(mlp_input, weights, 0)

captured_mlp = {}
def hook_mlp(module, inp, out):
    captured_mlp["mlp"] = out[0] if isinstance(out, tuple) else out

h_handle = ref.model.layers[0].mlp.register_forward_hook(hook_mlp)
with torch.no_grad():
    ref(ids)
h_handle.remove()
diff = torch.max(torch.abs(mine_mlp - captured_mlp["mlp"]))
print("mlp max abs diff:", diff.item())
assert diff < 1e-3, "mlp mismatch"
print("mlp matches")