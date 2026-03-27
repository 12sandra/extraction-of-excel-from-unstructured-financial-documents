# main.py — FastAPI ML Service Entry Point
# FIXED: PaddleOCR v3.x compatibility + PADDLE connectivity check disabled

import os

# ── Set BEFORE any paddle import ─────────────────────────────────────────────
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import extract, train
import uvicorn
import logging
from dotenv import load_dotenv

load_dotenv("../../.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("../../logs/fastapi.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("gst2_fastapi")

app = FastAPI(
    title="GST2 Converter ML Service",
    description="OCR + Table Extraction + LLM Validation for GST 2A Statements",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(extract.router, prefix="/api/extract", tags=["Extraction"])
app.include_router(train.router,   prefix="/api/train",   tags=["Training"])


@app.get("/health")
async def health_check():
    import torch
    return {
        "status": "running",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "service": "GST2 ML Service v2.0"
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True, log_level="info")