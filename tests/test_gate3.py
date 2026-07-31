from src.quantize import quantize_model, compare_memory
from src.loader import load_weights
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
weights = load_weights("model/model.safetensors", device)
q_weights = quantize_model(weights)
compare_memory(weights, q_weights)