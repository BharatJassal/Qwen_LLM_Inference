import torch
from src.model import forward, decode_step
from src.layers import rope_tables

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

    gen_tensor = torch.tensor([generated], dtype=token_ids.dtype, device=token_ids.device)
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
    generated = list(stream_sample(token_ids, weights, config, max_new_tokens, temperature, top_p))
    gen_tensor = torch.tensor([generated], dtype=token_ids.dtype, device=token_ids.device)
    return torch.cat([token_ids, gen_tensor], dim=1)

@torch.no_grad()
def stream_sample(token_ids, weights, config, max_new_tokens, temperature, top_p):
    if token_ids.is_cuda:
        yield from stream_graph(token_ids, weights, config, max_new_tokens,
                                lambda logits: sample(logits, temperature, top_p))
        return

    logits, cache = forward(token_ids, weights, config)
    next_token = sample(logits[0, -1, :], temperature, top_p)

    for _ in range(max_new_tokens):
        if next_token.item() == config.eos_token_id:
            break
        yield next_token.item()

        next_input = next_token.view(1, 1)
        logits, cache = forward(next_input, weights, config, cache)
        next_token = sample(logits[0, -1, :], temperature, top_p)


@torch.no_grad()
def greedy_graph(token_ids, weights, config, max_new_tokens):
    generated = list(stream_graph(token_ids, weights, config, max_new_tokens, torch.argmax))
    gen_tensor = torch.tensor([generated], dtype=token_ids.dtype, device=token_ids.device)
    return torch.cat([token_ids, gen_tensor], dim=1)


@torch.no_grad()
def stream_graph(token_ids, weights, config, max_new_tokens, pick):
    """Like stream_sample, but each decode step is a single CUDA graph replay. pick chooses the next token."""
    device = token_ids.device
    prompt_len = token_ids.shape[1]
    max_len = prompt_len + max_new_tokens + 1  # +1 for the warm-up step

    logits, cache = forward(token_ids, weights, config)
    shape = (config.num_hidden_layers, 1, config.num_key_value_heads, max_len, config.head_dim)
    k_cache = torch.zeros(shape, device=device)
    v_cache = torch.zeros(shape, device=device)
    for i, (k, v) in enumerate(cache):
        k_cache[i, :, :, :prompt_len] = k
        v_cache[i, :, :, :prompt_len] = v
    cos_table, sin_table = (t.to(device) for t in rope_tables(max_len, config.head_dim, config.rope_theta))
    positions = torch.arange(max_len, device=device)

    tok = pick(logits[0, -1, :]).view(1, 1).clone()
    pos = torch.tensor([prompt_len], device=device)

    def step():
        out = decode_step(tok, pos, weights, config, k_cache, v_cache, cos_table, sin_table, positions)
        pos.add_(1)
        return out[0, -1, :]

    # torch.cuda.graph needs a warm-up run on a side stream first
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        next_logits = step()
    pos.fill_(prompt_len)

    for _ in range(max_new_tokens):
        t = tok.item()
        if t == config.eos_token_id:
            break
        yield t
        graph.replay()
        tok.copy_(pick(next_logits).view(1, 1))
