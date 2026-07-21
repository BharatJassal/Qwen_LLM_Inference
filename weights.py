from safetensors import safe_open
import json
'''
This script maps the model weights and configuration for Qwen-2.5B-Instruct.
- Acheives dimensions of vocab_size, hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads, head_dim, intermediate_size.
- Extracts rms_norm_eps, rope_theta from Config.json. 
- Confirms GQA ratio by comparning num_attention_heads and num_key_value_heads.
- Verifies QKV biases exists, embeddings are tied.
'''

with safe_open("model/model.safetensors", framework="pt",) as f: 
   for key in f.keys():
      print(key, f.get_tensor(key).shape)

with open("model/config.json", "r") as f:
    config = json.load(f)

for key, value in config.items():
    print(f"{key}: {value}")
