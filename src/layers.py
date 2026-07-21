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

