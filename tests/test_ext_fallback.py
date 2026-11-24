import torch
from src import kernels, layers
from src.quantize import quantize_weight, get_weight

# pretend the build fails, e.g. no ninja or no compiler
def broken_load(*args, **kwargs):
    raise RuntimeError("Ninja is required to load C++ extensions")

kernels._ext, kernels._failed = None, False
kernels.load = broken_load
assert not kernels.available(), "available() should be False when the build fails"
assert kernels._failed, "a failed build should be remembered so it isn't retried"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
weights = {"w": quantize_weight(torch.randn(64, 32, device=device))}
x = torch.randn(1, 1, 32, device=device)
y = layers.linear(x, weights, "w")
assert torch.allclose(y, x @ get_weight(weights, "w").T), "fallback output is wrong"

# the fused ops fall back too
w = torch.ones(32, device=device)
s, h = layers.add_rmsnorm(x, x, w, 1e-6)
assert torch.allclose(h, layers.rmsnorm(2 * x, w, 1e-6)), "add_rmsnorm fallback is wrong"
assert layers.silu_mul(torch.randn(1, 1, 64, device=device)).shape == (1, 1, 32)
print("Extension fallback Pass.")
