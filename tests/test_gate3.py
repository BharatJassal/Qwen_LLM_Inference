from src.quantize import quantize_model, compare_memory
from src.loader import load_weights
weights = load_weights("model/model.safetensors")
q_weights = quantize_model(weights)
compare_memory(weights, q_weights)