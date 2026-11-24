import torch
import torch.nn.functional as F
from src.loader import load_weights
from src.quantize import quantize_model, get_weight
from src import kernels

fp32 = load_weights("model/model.safetensors", "cuda")
int8 = quantize_model(fp32)


def time_us(fn, reps=200):
    for _ in range(10):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / reps


print(f"{'projection':>10} | {'batch':>5} | {'int8 kernel (us)':>16} | {'dequantize (us)':>15} | {'fp32 cuBLAS (us)':>16}")
for name in ["gate_up_proj", "down_proj", "qkv_proj"]:
    part = "self_attn" if name == "qkv_proj" else "mlp"
    key = f"model.layers.0.{part}.{name}.weight"
    w_int8, scales = int8[key]
    for b in [1, 2, 4, 8]:
        x = torch.randn(b, w_int8.shape[1], device="cuda")
        kernel = time_us(lambda: kernels.int8_gemv(w_int8, scales, x))
        dequant = time_us(lambda: F.linear(x, get_weight(int8, key)))
        cublas = time_us(lambda: F.linear(x, fp32[key]))
        print(f"{name:>10} | {b:>5} | {kernel:>16.2f} | {dequant:>15.2f} | {cublas:>16.2f}")
