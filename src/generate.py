import torch
from src.model import forward

@torch.no_grad()
def greedy(token_ids, weights, config, max_new_tokens):
    for _ in range(max_new_tokens):
        logits, _ = forward(token_ids, weights, config)
        next_token_logits = logits[0, -1, :]
        next_token = torch.argmax(next_token_logits)
        if next_token.item() == config.eos_token_id:
            break
        next_token_reshape = next_token.unsqueeze(0).unsqueeze(0)
        token_ids = torch.cat([token_ids, next_token_reshape], dim=1)
    return token_ids

@torch.no_grad()
def greedy_cached(token_ids, weights, config, max_new_tokens):
    generated = []

    logits, cache = forward(token_ids, weights, config)
    next_token = torch.argmax(logits[0, -1, :])

    for _ in range(max_new_tokens):
        if next_token.item() == config.eos_token_id:
            break
        generated.append(next_token.item())
        next_input = next_token.view(1, 1)
        logits, cache = forward(next_input, weights, config, cache)
        next_token = torch.argmax(logits[0, -1, :])

    gen_tensor = torch.tensor([generated], dtype=token_ids.dtype)
    return torch.cat([token_ids, gen_tensor], dim=1)

@torch.no_grad()
def sample(logits, temperature, top_p):
    if temperature <= 0:
        return torch.argmax(logits)
    
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cum_sum = torch.cumsum(sorted_probs, dim=-1)

    remove = cum_sum > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_probs[remove] = 0
    sorted_probs = sorted_probs / sorted_probs.sum()

    choice = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_indices[choice]

@torch.no_grad()
def generate_sample(token_ids, weights, config, max_new_tokens, temperature, top_p):
    generated = []

    logits, cache = forward(token_ids, weights, config)
    next_token = sample(logits[0, -1, :], temperature, top_p)

    for _ in range(max_new_tokens):
        if next_token.item() == config.eos_token_id:
            break
        generated.append(next_token.item())
        next_input = next_token.view(1, 1)
        logits, cache = forward(next_input, weights, config, cache)
        next_token = sample(logits[0, -1, :], temperature, top_p)

    gen_tensor = torch.tensor([generated], dtype=token_ids.dtype)
    return torch.cat([token_ids, gen_tensor], dim=1)