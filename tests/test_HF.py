import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.loader import load_weights
from src.layers import embedding, rmsnorm

MODEL_DIR = "model"

ref = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
ref.eval()
tok = AutoTokenizer.from_pretrained(MODEL_DIR)


weights = load_weights("model/model.safetensors")

ids = tok("The capital of France is", return_tensors="pt").input_ids

emb = embedding(ids, weights["model.embed_tokens.weight"])

mine = rmsnorm(
    emb, 
    weights["model.layers.0.input_layernorm.weight"],
    eps = 1e-6
)

with torch.no_grad():
    ref_norm = ref.model.layers[0].input_layernorm(emb)

# Compare
diff = torch.max(torch.abs(mine - ref_norm))
print("rmsnorm max abs diff:", diff.item())
assert diff < 1e-6, "rmsnorm mis`match"
print("rmsnorm matches")