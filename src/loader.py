from safetensors import safe_open
import torch


def load_weights(path, device="cpu"):
    weight = {}
    with safe_open(path, framework="pt",) as f: 
        for key in f.keys():
            weight[key] = f.get_tensor(key).to(device=device, dtype=torch.float32)
    return merge_projections(weight)


def merge_projections(weights):
    # one larger matmul fills the GPU better than three (q, k, v) or two (gate, up) small ones
    i = 0
    while f"model.layers.{i}.self_attn.q_proj.weight" in weights:
        p = f"model.layers.{i}."
        for kind in ("weight", "bias"):
            weights[f"{p}self_attn.qkv_proj.{kind}"] = torch.cat(
                [weights.pop(f"{p}self_attn.{n}_proj.{kind}") for n in "qkv"])
        weights[f"{p}mlp.gate_up_proj.weight"] = torch.cat(
            [weights.pop(f"{p}mlp.{n}_proj.weight") for n in ("gate", "up")])
        i += 1
    return weights


if __name__ == "__main__":
    weights = load_weights("../model/model.safetensors")
    for key, value in weights.items():
        print(key, value.shape)