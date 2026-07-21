import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.loader import load_weights
from src.layers import embedding, rmsnorm, rope_tables, apply_rope

MODEL_DIR = "model"

ref = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
ref.eval()
tok = AutoTokenizer.from_pretrained(MODEL_DIR)

weights = load_weights("model/model.safetensors")

cos, sin = rope_tables(seq=5, head_dim=64, theta=1_000_000.0)
print("cos/sin shape:", cos.shape) 
q = torch.randn(1, 14, 5, 64)
cos_b = cos[None, None, :, :]
sin_b = sin[None, None, :, :]
q_rot = apply_rope(q, cos_b, sin_b)
print("q_rot shape:", q_rot.shape)     
print("pos0 unchanged:", torch.allclose(q_rot[:, :, 0, :], q[:, :, 0, :], atol=1e-6))

ids = tok("The capital of Canada is", return_tensors="pt").input_ids

emb = embedding(ids, weights["model.embed_tokens.weight"])

mine = rmsnorm(
    emb, 
    weights["model.layers.0.input_layernorm.weight"],
    eps = 1e-6
)

with torch.no_grad():
    ref_norm = ref.model.layers[0].input_layernorm(emb)

diff = torch.max(torch.abs(mine - ref_norm))
print("rmsnorm max abs diff:", diff.item())
assert diff < 1e-6, "rmsnorm mismatch"
print("rmsnorm matches")