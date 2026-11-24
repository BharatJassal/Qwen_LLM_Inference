import os
import warnings
import torch
from torch.utils.cpp_extension import load

MAX_BATCH = 8
_ext = None
_failed = False


def _load():
    global _ext, _failed
    if _ext is None and not _failed:
        here = os.path.dirname(os.path.abspath(__file__))
        source = os.path.join(here, "..", "cuda", "int8_gemv_ext.cu")
        try:
            _ext = load(name="int8_gemv_ext", sources=[source], verbose=True)
        except Exception as e:
            _failed = True
            warnings.warn(f"INT8 extension failed to build, falling back to dequantize: {e}")
    return _ext


def available():
    return torch.cuda.is_available() and _load() is not None


def int8_gemv(w_int8, scales, x):
    return _load().int8_gemv(w_int8, scales.reshape(-1), x)
