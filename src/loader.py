from safetensors import safe_open
import torch


def load_weights(path, device="cpu"):
    weight = {}
    with safe_open(path, framework="pt",) as f: 
        for key in f.keys():
            weight[key] = f.get_tensor(key).to(device=device, dtype=torch.float32)
    return weight


if __name__ == "__main__":
    weights = load_weights("../model/model.safetensors")
    for key, value in weights.items():
        print(key, value.shape)