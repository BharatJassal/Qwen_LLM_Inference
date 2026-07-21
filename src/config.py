from dataclasses import dataclass
import json
'''
This script builds a dataclass ModelConfig to store the model configuration for Qwen-2.5B-Instruct.
- It reads from a JSON file and initializes the ModelConfig dataclass with the relevant parameters.

'''
@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    eos_token_id: int

    @classmethod
    def from_json(cls, json_file: str):
        with open(json_file, "r") as f:
            config = json.load(f)
        return cls(
            vocab_size = config["vocab_size"],
            hidden_size = config["hidden_size"],
            num_hidden_layers = config["num_hidden_layers"],
            num_attention_heads = config["num_attention_heads"],
            num_key_value_heads = config["num_key_value_heads"],
            head_dim = config["hidden_size"] // config["num_attention_heads"],
            intermediate_size = config["intermediate_size"],
            rms_norm_eps = config["rms_norm_eps"],
            rope_theta = config["rope_theta"],
            tie_word_embeddings = config["tie_word_embeddings"],
            eos_token_id = config["eos_token_id"]
        )

if __name__ == "__main__":
    config = ModelConfig.from_json("../model/config.json")
    print(config)