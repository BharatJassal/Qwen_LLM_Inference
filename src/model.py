import torch
import torch.nn.functional as F
from src.layers import embedding, rmsnorm, add_rmsnorm, attention, mlp, linear, rope_
from src.config import ModelConfig


def forward(token_ids, weights, config, cache=None):
    if cache is None:
        cache = [None] * config.num_hidden_layers

    eps = config.rms_norm_eps
    x = embedding(token_ids, weights["model.embed_tokens.weight"])
    # h is the normalized input to the next block, x the running residual sum
    h = rmsnorm(x, weights["model.layers.0.input_layernorm.weight"], eps)
    for i in range(config.num_hidden_layers):
        attn_out, cache[i] = attention(h, weights, i, config, cache[i])
        x, h = add_rmsnorm(x, attn_out, weights[f"model.layers.{i}.post_attention_layernorm.weight"], eps)
        x, h = add_rmsnorm(x, mlp(h, weights, i), weights[next_norm(i, config)], eps)

    logits = linear(h, weights, "model.embed_tokens.weight")
    return logits, cache


def next_norm(i, config):
    if i + 1 < config.num_hidden_layers:
        return f"model.layers.{i + 1}.input_layernorm.weight"
    return "model.norm.weight"


def decode_step(tok, pos, weights, config, k_cache, v_cache, cos_table, sin_table, positions):
    """Decode one token for each of S sequences using a fixed size KV cache, so it can run inside a CUDA graph."""
    S = tok.shape[0]
    H, KV, D = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    eps = config.rms_norm_eps
    rows = torch.arange(S, device=tok.device)
    future = (positions[None, :] > pos[:, None])[:, None, None, :]

    x = embedding(tok, weights["model.embed_tokens.weight"])
    h = rmsnorm(x, weights["model.layers.0.input_layernorm.weight"], eps)
    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}."
        qkv = linear(h, weights, p + "self_attn.qkv_proj.weight", weights[p + "self_attn.qkv_proj.bias"])
        rope_(qkv, cos_table, sin_table, pos, H + KV, D)
        q, k, v = qkv.split([H * D, KV * D, KV * D], dim=-1)
        # group query heads by their KV head
        q, k, v = q.reshape(S, KV, H // KV, D), k.reshape(S, KV, D), v.reshape(S, KV, D)
        k_cache[i][rows, :, pos] = k
        v_cache[i][rows, :, pos] = v

        scores = (q @ k_cache[i].transpose(-2, -1) / (D ** 0.5)).masked_fill(future, float("-inf"))
        attn = (F.softmax(scores, dim=-1) @ v_cache[i]).reshape(S, 1, H * D)
        x, h = add_rmsnorm(x, linear(attn, weights, p + "self_attn.o_proj.weight"),
                           weights[p + "post_attention_layernorm.weight"], eps)
        x, h = add_rmsnorm(x, mlp(h, weights, i), weights[next_norm(i, config)], eps)

    return linear(h, weights, "model.embed_tokens.weight")
