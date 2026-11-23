import torch
from src import int8_ext, layers
from src.quantize import quantize_weight, get_weight

# pretend the build fails, e.g. no ninja or no compiler
def broken_load(*args, **kwargs):
    raise RuntimeError("Ninja is required to load C++ extensions")

int8_ext._ext, int8_ext._failed = None, False
int8_ext.load = broken_load
assert not int8_ext.available(), "available() should be False when the build fails"
assert int8_ext._failed, "a failed build should be remembered so it isn't retried"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
weights = {"w": quantize_weight(torch.randn(64, 32, device=device))}
x = torch.randn(1, 1, 32, device=device)
y = layers.linear(x, weights, "w")
assert torch.allclose(y, x @ get_weight(weights, "w").T), "fallback output is wrong"
print("Extension fallback Pass.")
