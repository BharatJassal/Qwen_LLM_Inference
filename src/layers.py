import torch
import torch.nn.functional as F
from src.quantize import get_weight

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

def attention(x, weights, layer_idx, config, layer_cache=None):
    num_heads = config.num_attention_heads
    head_dim = config.head_dim
    num_kv_heads = config.num_key_value_heads
    batch_size, seq_len, hidden_size = x.size()


    if layer_cache is None:
        past_len = 0
    else:
        past_len = layer_cache[0].shape[2] 

    q = F.linear(
        x,
        get_weight(weights, f"model.layers.{layer_idx}.self_attn.q_proj.weight"),
        weights[f"model.layers.{layer_idx}.self_attn.q_proj.bias"]
        )
    k = F.linear(
        x,
        get_weight(weights, f"model.layers.{layer_idx}.self_attn.k_proj.weight"),
        weights[f"model.layers.{layer_idx}.self_attn.k_proj.bias"]
        )
    v = F.linear(
        x,
        get_weight(weights, f"model.layers.{layer_idx}.self_attn.v_proj.weight"),
        weights[f"model.layers.{layer_idx}.self_attn.v_proj.bias"]
        )
    q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    cos, sin = rope_tables(past_len + seq_len, head_dim, config.rope_theta)
    cos = cos[past_len:]
    sin = sin[past_len:]
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

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
            torch.full((seq_len, seq_len), float('-inf')),
            diagonal=1,
        )
        scores = scores + mask

    weights_attn = F.softmax(scores, dim=-1)
    output = weights_attn @ v
    output = output.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)
    out = F.linear(
        output,
        get_weight(weights, f"model.layers.{layer_idx}.self_attn.o_proj.weight"),
    )

    return out, new_layer_cache

def mlp(x, weights, layer_idx):
    gate = F.linear(x, get_weight(weights, f"model.layers.{layer_idx}.mlp.gate_proj.weight"))
    up = F.linear(x, get_weight(weights, f"model.layers.{layer_idx}.mlp.up_proj.weight"))
    hidden = F.silu(gate) * up
    out = F.linear(hidden, get_weight(weights, f"model.layers.{layer_idx}.mlp.down_proj.weight"))
    return out