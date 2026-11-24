from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
from src.loader import load_weights
from src.config import ModelConfig
from src.generate import stream_sample
from src.batching import BatchEngine
from src.quantize import quantize_model
from src import kernels
import asyncio
import threading
from fastapi.middleware.cors import CORSMiddleware

MODEL_DIR = "model"
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
weights = load_weights(f"{MODEL_DIR}/model.safetensors", device)
# without the kernel INT8 is slower than FP32
use_int8 = kernels.available()
if use_int8:
    weights = quantize_model(weights)
cfg = ModelConfig.from_json(f"{MODEL_DIR}/config.json")
engine = BatchEngine(weights, cfg) if device == "cuda" else None
cpu_lock = threading.Lock()
print(f"Model loaded on {device}, {'INT8' if use_int8 else 'FP32'} weights.")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.7
    top_p: float = 0.9

@app.get("/")
def root():
    return {"status": "alive"}

def cpu_stream(ids, req):
    with cpu_lock:
        yield from stream_sample(ids, weights, cfg, req.max_tokens, req.temperature, req.top_p)

def token_ids(req):
    messages = [{"role": "user", "content": req.prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    if engine is None:
        return cpu_stream(ids, req)
    try:
        return engine.submit(ids, req.max_tokens, req.temperature, req.top_p)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/generate")
async def generate(req: GenerateRequest):
    stream = token_ids(req)
    out = await asyncio.to_thread(list, stream)
    return {"text": tok.decode(out, skip_special_tokens=True)}

@app.post("/generate/stream")
def generate_stream(req: GenerateRequest):
    stream = token_ids(req)

    def token_stream():
        for token_id in stream:
            yield tok.decode([token_id], skip_special_tokens=True)

    return StreamingResponse(token_stream(), media_type="text/plain")
