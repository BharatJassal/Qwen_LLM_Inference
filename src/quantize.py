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

def memory(weights):
    '''
    Total bytes of a weights dict.
    input: (int8_tensor, fp32_scales) tuple.
    returns: (total_bytes, quantized_bytes, unquantized_bytes)
    '''
    quantized_bytes = 0
    unquantized_bytes = 0
    for key, value in weights.items():
        if isinstance(value, tuple):
            w_int8, scales = value
            quantized_bytes += (w_int8.numel() * w_int8.element_size()) + (scales.numel() * scales.element_size())
        else:
            unquantized_bytes += value.numel() * value.element_size()
    total = quantized_bytes + unquantized_bytes
    return total, quantized_bytes, unquantized_bytes

def compare_memory(fp32_weights, int8_weights):
    fp32_total, _, _ = memory(fp32_weights)
    int8_total, int8_q, int8_unq = memory(int8_weights)

    fp32_of_quantized = sum(v[0].numel() * 4 for v in int8_weights.values() if isinstance(v, tuple))

    mb = lambda b: b / (1024 ** 2)

    print(f"fp32 model total:        {mb(fp32_total):.2f} MB")
    print(f"int8 model total:        {mb(int8_total):.2f} MB")
    print(f"whole-model reduction:   {fp32_total / int8_total:.2f}x")
    print(f"quantized tensors, fp32: {mb(fp32_of_quantized):.2f} MB")
    print(f"quantized tensors, int8: {mb(int8_q):.2f} MB")
    print(f"quantized-only reduction: {fp32_of_quantized / int8_q:.2f}x")