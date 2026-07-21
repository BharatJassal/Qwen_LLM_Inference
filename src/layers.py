import torch

def embedding(token_ids, embed_weight):
    embed_tokens = embed_weight[token_ids]
    return embed_tokens

def rmsnorm(x, weight, eps):
    squared = x * x
    mean_squared = squared.mean(dim=-1, keepdim=True)
    rms = torch.sqrt(mean_squared + eps)
    normalized = x / rms
    return normalized * weight

def rope_tables(seq, head_dim, theta):
    i = torch.arange(0, head_dim, 2).float()
    inv_freq = 1.0 / (theta ** (i / head_dim))
    pos = torch.arange(seq).float()
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin

def rotate_half(x):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)