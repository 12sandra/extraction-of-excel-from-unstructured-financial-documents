# paddle_ocr.py — PaddleOCR on GPU
# FIXED for PaddleOCR v3.x: use_gpu/gpu_id removed → replaced with device="gpu:0"

import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import logging
import numpy as np
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger("gst2_fastapi.paddle_ocr")


class PaddleOCRService:
    """PaddleOCR wrapper — compatible with both v2.x and v3.x."""

    def __init__(self, use_gpu: bool = True, gpu_id: int = 0):
        logger.info(f"Loading PaddleOCR (GPU={use_gpu}, device={gpu_id})...")
        try:
            from paddleocr import PaddleOCR
            device = f"gpu:{gpu_id}" if use_gpu else "cpu"
            # v3.x API: use device= parameter
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                device=device,
                show_log=False,
            )
            logger.info("PaddleOCR loaded successfully (v3.x API)!")
        except TypeError as e:
            if "device" in str(e):
                # Fallback to v2.x API
                logger.warning("Falling back to PaddleOCR v2.x API...")
                from paddleocr import PaddleOCR
                self.ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang="en",
                    use_gpu=use_gpu,
                    gpu_id=gpu_id,
                    show_log=False,
                )
                logger.info("PaddleOCR loaded (v2.x API fallback)!")
            else:
                logger.error(f"Failed to load PaddleOCR: {e}")
                raise
        except Exception as e:
            logger.error(f"Failed to load PaddleOCR: {e}")
            raise

    def extract_text(self, image_path: str) -> Dict[str, Any]:
        """Extract text from image. Handles both v2.x and v3.x result formats."""
        try:
            result = self.ocr.ocr(image_path, cls=True)
            lines = []
            all_confidences = []

            if result:
                for page_result in result:
                    if page_result is None:
                        continue
                    if isinstance(page_result, list):
                        for line in page_result:
                            if line is None:
                                continue
                            try:
                                bbox       = line[0]
                                text       = line[1][0]
                                confidence = float(line[1][1])
                                if text and text.strip():
                                    lines.append({"text": text.strip(), "confidence": confidence, "bbox": bbox})
                                    all_confidences.append(confidence)
                            except (IndexError, TypeError, ValueError):
                                continue
                    else:
                        # v3.x object-style
                        try:
                            for item in page_result:
                                try:
                                    if hasattr(item, "rec_res"):
                                        text       = item.rec_res[0]
                                        confidence = float(item.rec_res[1])
                                        bbox       = item.det_res.tolist() if hasattr(item, "det_res") else None
                                    else:
                                        bbox       = item[0]
                                        text       = item[1][0]
                                        confidence = float(item[1][1])
                                    if text and text.strip():
                                        lines.append({"text": text.strip(), "confidence": confidence, "bbox": bbox})
                                        all_confidences.append(confidence)
                                except (IndexError, TypeError, AttributeError, ValueError):
                                    continue
                        except TypeError:
                            logger.warning(f"Unexpected OCR result type: {type(page_result)}")

            avg_confidence = float(np.mean(all_confidences)) if all_confidences else 0.0
            full_text      = " ".join(line["text"] for line in lines)

            logger.info(f"Extracted {len(lines)} lines, avg confidence: {avg_confidence:.2f}")
            return {"lines": lines, "full_text": full_text, "avg_confidence": avg_confidence, "line_count": len(lines)}

        except Exception as e:
            logger.error(f"OCR extraction failed for {image_path}: {e}")
            return {"lines": [], "full_text": "", "avg_confidence": 0.0, "line_count": 0}

    def extract_from_multiple_images(self, image_paths: List[str]) -> List[Dict]:
        results = []
        for i, path in enumerate(image_paths):
            logger.info(f"Processing page {i+1}/{len(image_paths)}: {Path(path).name}")
            result = self.extract_text(path)
            result["page_number"] = i + 1
            result["image_path"]  = path
            results.append(result)
        return results


_paddle_ocr_instance = None

def get_paddle_ocr(use_gpu: bool = True) -> PaddleOCRService:
    global _paddle_ocr_instance
    if _paddle_ocr_instance is None:
        _paddle_ocr_instance = PaddleOCRService(use_gpu=use_gpu)
    return _paddle_ocr_instance