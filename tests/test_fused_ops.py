import torch
from src import kernels, layers

assert kernels.available(), "needs CUDA (extension builds from the files in cuda/)"
torch.manual_seed(0)
device = "cuda"


def both(fn, *args):
    """Runs fn with the CUDA kernels and with the PyTorch fallback."""
    layers.USE_FUSED = True
    fused = fn(*[a.clone() for a in args])
    layers.USE_FUSED = False
    ref = fn(*[a.clone() for a in args])
    layers.USE_FUSED = True
    return fused, ref


def close(a, b, name):
    err = ((a - b).abs().max() / b.abs().max()).item()
    assert err < 1e-6, f"{name}: max rel err {err:.2e}"
    return err


w = torch.randn(896, device=device)
for shape in [(1, 1, 896), (8, 1, 896), (2, 37, 896)]:
    x, r = torch.randn(shape, device=device), torch.randn(shape, device=device)
    err = close(*both(lambda x: layers.rmsnorm(x, w, 1e-6), x), "rmsnorm")
    (s1, h1), (s2, h2) = both(lambda x, r: layers.add_rmsnorm(x, r, w, 1e-6), x, r)
    assert torch.equal(s1, s2), "add_rmsnorm sum differs"
    close(h1, h2, "add_rmsnorm")
    print(f"rmsnorm and add_rmsnorm {shape} match, max rel err {err:.1e}")

cos, sin = layers.rope_tables(64, 64, 1000000.0)
cos, sin = cos.to(device), sin.to(device)
for tokens in [1, 8, 37]:
    qkv = torch.randn(tokens, 1, 1152, device=device)
    pos = torch.randint(0, 64, (tokens,), device=device)

    def rotate(qkv):
        layers.rope_(qkv, cos, sin, pos, 16, 64)
        return qkv
    fused, ref = both(rotate, qkv)
    assert torch.equal(fused, ref), f"rope differs for {tokens} tokens"
    assert torch.equal(fused[..., 1024:], qkv[..., 1024:]), "rope touched v"
print("rope matches the PyTorch version bit for bit and leaves v alone")

for shape in [(1, 1, 9728), (5, 3, 9728)]:
    err = close(*both(layers.silu_mul, torch.randn(shape, device=device)), "silu_mul")
    print(f"silu_mul {shape} matches, max rel err {err:.1e}")

bad = [
    lambda: kernels.rmsnorm(torch.randn(2, 896, device=device), torch.randn(895, device=device), 1e-6),
    lambda: kernels.add_rmsnorm(torch.randn(2, 896, device=device), torch.randn(3, 896, device=device), w, 1e-6),
    lambda: kernels.rope_(torch.randn(3, 1152, device=device), cos, sin, pos[:2].contiguous(), 16, 64),
    lambda: kernels.rope_(torch.randn(3, 1000, device=device), cos, sin, torch.zeros(3, dtype=torch.long, device=device), 16, 64),
    lambda: kernels.silu_mul(torch.randn(2, 9727, device=device)),
]
for i, call in enumerate(bad):
    try:
        call()
        raise AssertionError(f"bad input {i} was accepted")
    except RuntimeError:
        pass
print("bad shapes rejected. Fused ops Pass.")
