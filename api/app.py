import os
from fastapi import FastAPI, HTTPException
from api.schemas import SummarizeRequest, SummarizeResponse
from src.inference import T5Summarizer

MODEL_PATH = os.environ.get('MODEL_PATH', 'runs/t5-small-cnn')

app = FastAPI(title="Document Summarization API", version="1.0")

try:
    summarizer = T5Summarizer(MODEL_PATH)
except Exception as e:
    summarizer = None
    print(f"Model load failed: {e}")

@app.post('/summarize', response_model=SummarizeResponse)
async def summarize(req: SummarizeRequest):
    if summarizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Set MODEL_PATH to a valid checkpoint.")
    if len(req.text.split()) < 30:
        raise HTTPException(status_code=400, detail="Input is too short; provide a longer article.")
    summary = summarizer.summarize(req.text, max_length=req.max_length, min_length=req.min_length)
    return SummarizeResponse(summary=summary)

@app.get('/health')
async def health():
    return {"ok": summarizer is not None}
