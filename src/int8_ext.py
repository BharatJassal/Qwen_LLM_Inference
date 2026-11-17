import os
import torch
from torch.utils.cpp_extension import load

_ext = None  # filled in the first time it's needed


def _load():
    global _ext
    if _ext is None:
        here = os.path.dirname(os.path.abspath(__file__))
        source = os.path.join(here, "..", "cuda", "int8_gemv_ext.cu")
        _ext = load(name="int8_gemv_ext", sources=[source], verbose=True)
    return _ext


def available():
    return torch.cuda.is_available()


def int8_gemv(w_int8, scales, x):
    return _load().int8_gemv(w_int8, scales.reshape(-1), x)