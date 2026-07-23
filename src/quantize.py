import torch

def quantize_weight(w):
    high = w.abs().amax(dim=1, keepdim=True)
    scales = high / 127.0
    scales = torch.clamp(scales, min=1e-8) 
    w_scaled = w / scales
    w_quantized = torch.clamp(torch.round(w_scaled), min=-127, max=127).to(torch.int8)
    return w_quantized, scales

def dequantize_weight(w_int, scales):
    w_int = w_int.to(torch.float32)
    w_dequantized = w_int * scales
    return w_dequantized

def quantize_model(weights):
    quantized = {}
    for key, value in weights.items():
        if key.endswith("_proj.weight"):
            quantized[key] = quantize_weight(value)
        else:
            quantized[key] = value
    return quantized

def get_weight(weights, key):
    value = weights[key]
    if isinstance(value, tuple):
        return dequantize_weight(value[0], value[1])
    return value