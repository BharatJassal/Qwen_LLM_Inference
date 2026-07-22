import torch
from src.model import forward

@torch.no_grad()
def greedy(token_ids, weights, config, max_new_tokens):
    for _ in range(max_new_tokens):
        logits = forward(token_ids, weights, config)
        next_token_logits = logits[0, -1, :]
        next_token = torch.argmax(next_token_logits)
        if next_token.item() == config.eos_token_id:
            break
        next_token_reshape = next_token.unsqueeze(0).unsqueeze(0)
        token_ids = torch.cat([token_ids, next_token_reshape], dim=1)
    return token_ids
