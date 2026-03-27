import logging
import torch
from typing import Dict, Any

logger = logging.getLogger("gst2_fastapi.got_ocr")

CONFIDENCE_THRESHOLD = 0.80  # Use fallback if PaddleOCR avg confidence < 80%

class GOTOCRService:
    """
    GOT-OCR 2.0 (General OCR Theory) — used as fallback for low-confidence cases.
    Runs on GPU. Better at complex layouts and low-quality scans.
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._loaded = False
        
    def load(self):
        """Lazy load — only load model when actually needed."""
        if self._loaded:
            return
            
        logger.info("Loading GOT-OCR 2.0 model (GPU)...")
        try:
            from transformers import AutoModel, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                "stepfun-ai/GOT-OCR2_0",
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                "stepfun-ai/GOT-OCR2_0",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="cuda",
                use_safetensors=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            self.model = self.model.eval().cuda()
            self._loaded = True
            logger.info("GOT-OCR 2.0 loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load GOT-OCR 2.0: {e}")
            logger.warning("Fallback OCR not available. Will use PaddleOCR results even if low confidence.")
    
    def extract_text(self, image_path: str) -> Dict[str, Any]:
        """Extract text using GOT-OCR 2.0."""
        if not self._loaded:
            self.load()
            
        if not self._loaded:
            return {"full_text": "", "avg_confidence": 0.0, "lines": [], "source": "got_ocr_failed"}
        
        try:
            result = self.model.chat(
                self.tokenizer,
                image_path,
                ocr_type="ocr"  # Plain OCR mode
            )
            
            # GOT-OCR returns plain text — split into lines
            lines_text = [line.strip() for line in result.split("\n") if line.strip()]
            lines = [{"text": line, "confidence": 0.95, "bbox": None} for line in lines_text]
            
            return {
                "lines": lines,
                "full_text": result,
                "avg_confidence": 0.95,  # GOT-OCR doesn't provide per-token confidence
                "line_count": len(lines),
                "source": "got_ocr"
            }
        except Exception as e:
            logger.error(f"GOT-OCR extraction failed: {e}")
            return {"lines": [], "full_text": "", "avg_confidence": 0.0, "line_count": 0, "source": "got_ocr_failed"}


# Singleton
_got_ocr_instance = None

def get_got_ocr() -> GOTOCRService:
    global _got_ocr_instance
    if _got_ocr_instance is None:
        _got_ocr_instance = GOTOCRService()
    return _got_ocr_instance


def should_use_fallback(paddle_result: Dict) -> bool:
    """Decide if GOT-OCR fallback is needed."""
    return paddle_result.get("avg_confidence", 1.0) < CONFIDENCE_THRESHOLD
