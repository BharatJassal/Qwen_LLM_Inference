import torch
import torch.nn.functional as F
from src.layers import embedding, rmsnorm, attention, mlp
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
    logits = F.linear(x, weights["model.embed_tokens.weight"])

    return logits, cache