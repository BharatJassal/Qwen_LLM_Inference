import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.loader import load_weights
from src.config import ModelConfig
from src.layers import embedding, rmsnorm, rope_tables, apply_rope, attention

MODEL_DIR = "model"

ref = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
ref.eval()
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
weights = load_weights("model/model.safetensors")

cfg = ModelConfig.from_json("model/config.json")
ids = tok("The capital of Canada is", return_tensors="pt").input_ids
emb = embedding(ids, weights["model.embed_tokens.weight"])

attn_input = rmsnorm(
    emb,
    weights["model.layers.0.input_layernorm.weight"],
    eps=cfg.rms_norm_eps
)
mine_attn = attention(attn_input, weights, 0, cfg)
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

cos, sin = rope_tables(seq=5, head_dim=64, theta=1_000_000.0)
print("cos/sin shape:", cos.shape) 
q = torch.randn(1, 14, 5, 64)
cos_b = cos[None, None, :, :]
sin_b = sin[None, None, :, :]
q_rot = apply_rope(q, cos_b, sin_b)
print("q_rot shape:", q_rot.shape)     
print("pos0 unchanged:", torch.allclose(q_rot[:, :, 0, :], q[:, :, 0, :], atol=1e-6))
