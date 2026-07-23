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

if __name__ == "__main__":
    from src.loader import load_weights
    weights = load_weights("model/model.safetensors")

    w = weights["model.layers.0.self_attn.q_proj.weight"]
    w_int8, scales = quantize_weight(w)
    w_recovered = dequantize_weight(w_int8, scales)

    print("shapes:", w.shape, w_int8.shape, scales.shape)
    print("dtypes:", w.dtype, "->", w_int8.dtype)
    print("original bytes: ", w.numel() * 4)
    print("quantized bytes:", w_int8.numel() * 1 + scales.numel() * 4)
    print("max abs error: ", torch.max(torch.abs(w - w_recovered)).item())
    print("mean abs error:", torch.mean(torch.abs(w - w_recovered)).item())