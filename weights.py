from safetensors import safe_open

with safe_open("model/model.safetensors", framework="pt",) as f:
    for key in f.keys():
        print(key, f.get_tensor(key).shape)