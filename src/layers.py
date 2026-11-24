import torch
import torch.nn.functional as F
from src.quantize import get_weight

from src import kernels

USE_FUSED = True


def use_kernels(x):
    return USE_FUSED and x.is_cuda and kernels.available()

def linear(x, weights, key, bias=None):
    w = weights[key]
    n = x.shape[0] * x.shape[1]

    if isinstance(w, tuple) and n <= kernels.MAX_BATCH and use_kernels(x):
        y = kernels.int8_gemv(w[0], w[1], x.reshape(n, -1).contiguous(), bias)
        return y.view(x.shape[0], x.shape[1], -1)

    return F.linear(x, get_weight(weights, key), bias)

def embedding(token_ids, embed_weight):
    if isinstance(embed_weight, tuple):
        w_int8, scales = embed_weight
        return w_int8[token_ids].float() * scales[token_ids]
    return embed_weight[token_ids]

def rmsnorm(x, weight, eps):
    if use_kernels(x):
        return kernels.rmsnorm(x, weight, eps)
    squared = x * x
    mean_squared = squared.mean(dim=-1, keepdim=True)
    rms = torch.sqrt(mean_squared + eps)
    normalized = x / rms
    return normalized * weight

def add_rmsnorm(x, residual, weight, eps):
    """Returns x + residual and its RMSNorm, one kernel on the GPU instead of eight."""
    if use_kernels(x):
        return kernels.add_rmsnorm(x, residual, weight, eps)
    x = x + residual
    return x, rmsnorm(x, weight, eps)

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

def rope_(qkv, cos_table, sin_table, pos, heads, head_dim):
    """Rotates the first `heads` heads of every token in place, which are q and k in a merged qkv."""
    if use_kernels(qkv):
        return kernels.rope_(qkv, cos_table, sin_table, pos, heads, head_dim)
    x = qkv.view(-1, qkv.shape[-1])[:, :heads * head_dim].view(-1, heads, head_dim)
    x.copy_(apply_rope(x, cos_table[pos][:, None], sin_table[pos][:, None]))

def silu_mul(gate_up):
    if use_kernels(gate_up):
        return kernels.silu_mul(gate_up)
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up

def attention(x, weights, layer_idx, config, layer_cache=None):
    num_heads = config.num_attention_heads
    head_dim = config.head_dim
    num_kv_heads = config.num_key_value_heads
    batch_size, seq_len, hidden_size = x.size()


    if layer_cache is None:
        past_len = 0
    else:
        past_len = layer_cache[0].shape[2] 

    qkv = linear(
        x,
        weights,
        f"model.layers.{layer_idx}.self_attn.qkv_proj.weight",
        weights[f"model.layers.{layer_idx}.self_attn.qkv_proj.bias"]
    )
    cos, sin = (t.to(x.device) for t in rope_tables(past_len + seq_len, head_dim, config.rope_theta))
    pos = torch.arange(past_len, past_len + seq_len, device=x.device).repeat(batch_size)
    rope_(qkv, cos, sin, pos, num_heads + num_kv_heads, head_dim)
    q, k, v = qkv.split([num_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim], dim=-1)
    q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    if layer_cache is not None:
        k = torch.cat([layer_cache[0], k], dim=2)
        v = torch.cat([layer_cache[1], v], dim=2)
    
    new_layer_cache = (k, v)

    n_rep = num_heads // num_kv_heads
    k = k.repeat_interleave(n_rep, dim=1)
    v = v.repeat_interleave(n_rep, dim=1)

    scores = q @ k.transpose(-2, -1) / (head_dim ** 0.5)
    if seq_len > 1:
        mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=x.device),
            diagonal=1,
        )
        scores = scores + mask

    weights_attn = F.softmax(scores, dim=-1)
    output = weights_attn @ v
    output = output.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)
    out = linear(
        output,
        weights,
        f"model.layers.{layer_idx}.self_attn.o_proj.weight"
    )

    return out, new_layer_cache

def mlp(x, weights, layer_idx):
    hidden = silu_mul(linear(x, weights, f"model.layers.{layer_idx}.mlp.gate_up_proj.weight"))
    out = linear(hidden, weights, f"model.layers.{layer_idx}.mlp.down_proj.weight")
    return out