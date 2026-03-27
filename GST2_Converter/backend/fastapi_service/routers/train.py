import os
import logging
from fastapi import APIRouter, HTTPException
from pathlib import Path
from typing import Optional
import asyncio

router = APIRouter()
logger = logging.getLogger("gst2_fastapi.train")

TRAINING_DATA_PATH = os.getenv("TRAINING_DATA_PATH", "./training_data")
LAYOUTLMV3_MODEL_PATH = os.getenv("LAYOUTLMV3_MODEL_PATH", "./ml_models/layoutlmv3")

training_job_status = {}


@router.post("/start")
async def start_training(
    background_tasks=None,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    annotated_only: bool = True
):
    """
    Start fine-tuning LayoutLMv3 on accumulated training data.
    
    Args:
        epochs: Number of training epochs (start with 3)
        batch_size: Batch size (keep at 4 for 6GB VRAM)
        learning_rate: Learning rate for fine-tuning
        annotated_only: If True, only train on user-corrected data
    """
    import uuid
    
    # Check if we have training data
    data_dir = Path(TRAINING_DATA_PATH)
    if annotated_only:
        data_dir = data_dir / "annotated"
    else:
        data_dir = data_dir / "raw"
    
    json_files = list(data_dir.glob("*.json")) if data_dir.exists() else []
    
    if len(json_files) < 5:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough training data. Found {len(json_files)} files. Need at least 5. "
                   f"Keep uploading and correcting documents to build the dataset."
        )
    
    training_id = str(uuid.uuid4())[:8]
    training_job_status[training_id] = {
        "status": "queued",
        "data_files": len(json_files),
        "epochs": epochs,
        "progress": 0
    }
    
    # Run in background
    asyncio.create_task(
        run_training_job(training_id, data_dir, epochs, batch_size, learning_rate)
    )
    
    return {
        "training_id": training_id,
        "status": "started",
        "data_files": len(json_files),
        "epochs": epochs,
        "message": f"Training started with {len(json_files)} documents"
    }


async def run_training_job(training_id: str, data_dir: Path, epochs: int, batch_size: int, lr: float):
    """Background training job."""
    try:
        training_job_status[training_id]["status"] = "loading_data"
        
        # Import training utilities
        from training.layoutlm_trainer import train_layoutlmv3
        
        training_job_status[training_id]["status"] = "training"
        
        result = await train_layoutlmv3(
            data_dir=str(data_dir),
            output_dir=LAYOUTLMV3_MODEL_PATH,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            progress_callback=lambda p: training_job_status[training_id].update({"progress": p})
        )
        
        training_job_status[training_id].update({
            "status": "complete",
            "final_loss": result.get("final_loss"),
            "accuracy": result.get("accuracy"),
            "message": "Model updated! New extractions will use improved model."
        })
        
        # Reload the model singleton
        from services.layout_lm import _layoutlm_instance
        if _layoutlm_instance:
            _layoutlm_instance._loaded = False  # Force reload on next request
        
        logger.info(f"Training complete! Model saved to {LAYOUTLMV3_MODEL_PATH}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        training_job_status[training_id]["status"] = "failed"
        training_job_status[training_id]["error"] = str(e)


@router.get("/status/{training_id}")
async def get_training_status(training_id: str):
    """Get training job status."""
    if training_id not in training_job_status:
        raise HTTPException(status_code=404, detail="Training job not found")
    return training_job_status[training_id]


@router.get("/dataset/stats")
async def get_dataset_stats():
    """Get statistics about available training data."""
    raw_dir = Path(TRAINING_DATA_PATH) / "raw"
    annotated_dir = Path(TRAINING_DATA_PATH) / "annotated"
    
    raw_count = len(list(raw_dir.glob("*.json"))) if raw_dir.exists() else 0
    annotated_count = len(list(annotated_dir.glob("*.json"))) if annotated_dir.exists() else 0
    
    return {
        "raw_documents": raw_count,
        "annotated_documents": annotated_count,
        "total_documents": raw_count + annotated_count,
        "ready_for_training": annotated_count >= 5,
        "recommendation": "Annotated data gives better results than raw data for fine-tuning."
    }