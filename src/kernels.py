import os
import warnings
import torch
from torch.utils.cpp_extension import load

MAX_BATCH = 8
SOURCES = ["bindings.cpp", "int8_gemv_ext.cu", "fused_ops.cu"]
_ext = None
_failed = False

def _load():
    global _ext, _failed
    if _ext is None and not _failed:
        here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cuda")
        try:
            _ext = load(name="qwen_kernels", sources=[os.path.join(here, s) for s in SOURCES], verbose=True)
        except Exception as e:
            _failed = True
            warnings.warn(f"CUDA kernels failed to build, falling back to PyTorch: {e}")
    return _ext

def available():
    return torch.cuda.is_available() and _load() is not None

def int8_gemv(w_int8, scales, x, bias=None):
    return _load().int8_gemv(w_int8, scales.reshape(-1), x, bias)

def rmsnorm(x, weight, eps):
    return _load().rmsnorm(x.contiguous(), weight, eps)[0]

def add_rmsnorm(x, residual, weight, eps):
    return tuple(_load().rmsnorm(x.contiguous(), weight, eps, residual.contiguous()))

def rope_(x, cos_table, sin_table, pos, heads, head_dim):
    _load().rope_(x, cos_table, sin_table, pos, heads, head_dim)

def silu_mul(gate_up):
    return _load().silu_mul(gate_up.contiguous())
