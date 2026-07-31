import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from src.loader import load_weights
from src.config import ModelConfig
from src.model import forward
from src.quantize import quantize_model, compare_memory

MODEL_DIR = "model"
TEXT_PATH = test.txt # Put text file here. Ex "gutenberg.txt"
MAX_TOKENS = 300


@torch.no_grad()
def perplexity(ids, weights, config):
    logits, _ = forward(ids, weights, config)

    predict = logits[:, :-1, :]
    target = ids[:, 1:]
    predict = predict.reshape(-1, predict.shape[-1])
    target = target.reshape(-1)

    loss = F.cross_entropy(predict, target)
    return torch.exp(loss).item()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    cfg = ModelConfig.from_json("model/config.json")
    weights = load_weights("model/model.safetensors", device)

    with open(TEXT_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    ids = tok(text, return_tensors="pt").input_ids[:, :MAX_TOKENS].to(device)

    q_weights = quantize_model(weights)
    compare_memory(weights, q_weights)

    ppl_fp32 = perplexity(ids, weights, cfg)
    ppl_int8 = perplexity(ids, q_weights, cfg)

    print(f"fp32 perplexity: {ppl_fp32:.4f}")
    print(f"int8 perplexity: {ppl_int8:.4f}")


if __name__ == "__main__":
    main()