import torch
import torch.nn.functional as F
from src.layers import embedding, rmsnorm, attention, mlp, linear, apply_rope
from src.config import ModelConfig


def forward(token_ids, weights, config, cache=None):
    if cache is None:
        cache = [None] * config.num_hidden_layers

    x = embedding(token_ids, weights["model.embed_tokens.weight"])
    for i in range(config.num_hidden_layers):
        norm_x = rmsnorm(
            x,
            weights[f"model.layers.{i}.input_layernorm.weight"],
            config.rms_norm_eps
        )
        attn_out, cache[i] = attention(norm_x, weights, i, config, cache[i])
        x = x + attn_out
        
        norm_x = rmsnorm(
            x,
            weights[f"model.layers.{i}.post_attention_layernorm.weight"],
            config.rms_norm_eps
        )
        x = x + mlp(norm_x, weights, i)
    
    x = rmsnorm(
        x,
        weights["model.norm.weight"],
        config.rms_norm_eps
    )
    logits = linear(x, weights, "model.embed_tokens.weight")

    return logits, cache


def decode_step(tok, pos, weights, config, k_cache, v_cache, cos_table, sin_table, positions):
    """Decode one token for each of S sequences using a fixed size KV cache, so it can run inside a CUDA graph."""
    S = tok.shape[0]
    H, KV, D = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    rows = torch.arange(S, device=tok.device)
    cos = cos_table.index_select(0, pos)[:, None, :]
    sin = sin_table.index_select(0, pos)[:, None, :]
    future = (positions[None, :] > pos[:, None])[:, None, None, :]

    x = embedding(tok, weights["model.embed_tokens.weight"])
    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}."
        h = rmsnorm(x, weights[p + "input_layernorm.weight"], config.rms_norm_eps)
        # group query heads by their KV head
        q = linear(h, weights, p + "self_attn.q_proj.weight", weights[p + "self_attn.q_proj.bias"]).view(S, KV, H // KV, D)
        k = linear(h, weights, p + "self_attn.k_proj.weight", weights[p + "self_attn.k_proj.bias"]).view(S, KV, D)
        v = linear(h, weights, p + "self_attn.v_proj.weight", weights[p + "self_attn.v_proj.bias"]).view(S, KV, D)
        q = apply_rope(q, cos[:, None], sin[:, None])
        k = apply_rope(k, cos, sin)
        k_cache[i][rows, :, pos] = k
        v_cache[i][rows, :, pos] = v

        scores = (q @ k_cache[i].transpose(-2, -1) / (D ** 0.5)).masked_fill(future, float("-inf"))
        attn = (F.softmax(scores, dim=-1) @ v_cache[i]).reshape(S, 1, H * D)
        x = x + linear(attn, weights, p + "self_attn.o_proj.weight")

        h = rmsnorm(x, weights[p + "post_attention_layernorm.weight"], config.rms_norm_eps)
        x = x + mlp(h, weights, i)

    x = rmsnorm(x, weights["model.norm.weight"], config.rms_norm_eps)
    return linear(x, weights, "model.embed_tokens.weight")
