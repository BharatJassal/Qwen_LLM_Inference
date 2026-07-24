from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
from src.loader import load_weights
from src.config import ModelConfig
from src.generate import generate_sample

MODEL_DIR = "model"

print("Loading model...")
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
weights = load_weights(f"{MODEL_DIR}/model.safetensors")
cfg = ModelConfig.from_json(f"{MODEL_DIR}/config.json")
print("Model loaded.")
app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.7
    top_p: float = 0.9

@app.get("/")
def root():
    return {"status": "alive"}

@app.post("/generate")
def generate(req: GenerateRequest):
    messages = [{"role": "user", "content": req.prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids

    out = generate_sample(
        ids, weights, cfg,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )

    completion = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    return {"text": completion}