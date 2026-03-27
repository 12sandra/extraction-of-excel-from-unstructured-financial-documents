import logging
import torch
import os
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger("gst2_fastapi.layout_lm")

# GST 2A field labels that LayoutLM will classify
GST2A_LABELS = [
    "O",                  # Other (not a GST field)
    "B-GSTIN",           # GSTIN number start
    "I-GSTIN",           # GSTIN number continuation
    "B-TRADE_NAME",      # Supplier trade name
    "I-TRADE_NAME",
    "B-INVOICE_NO",      # Invoice number
    "I-INVOICE_NO",
    "B-INVOICE_DATE",    # Invoice date
    "I-INVOICE_DATE",
    "B-INVOICE_VALUE",   # Total invoice value
    "I-INVOICE_VALUE",
    "B-TAXABLE_VALUE",   # Taxable amount
    "I-TAXABLE_VALUE",
    "B-IGST",            # IGST amount
    "I-IGST",
    "B-CGST",            # CGST amount
    "I-CGST",
    "B-SGST",            # SGST/UTGST amount
    "I-SGST",
    "B-CESS",            # Cess amount
    "I-CESS",
    "B-PLACE_OF_SUPPLY", # Place of supply
    "I-PLACE_OF_SUPPLY",
    "B-RETURN_PERIOD",   # Return period (month-year)
    "I-RETURN_PERIOD",
]

LABEL2ID = {label: i for i, label in enumerate(GST2A_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(GST2A_LABELS)}


class LayoutLMv3Service:
    """
    Uses LayoutLMv3 to understand document structure.
    Takes OCR output (text + bounding boxes) and classifies each text token
    into its GST field type.
    """
    
    def __init__(self, model_path: str, use_gpu: bool = True):
        self.model_path = model_path
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._loaded = False
        
    def load(self):
        """Load LayoutLMv3 model and processor."""
        if self._loaded:
            return
            
        logger.info(f"Loading LayoutLMv3 from {self.model_path} on {self.device}...")
        
        try:
            from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
            
            # If fine-tuned model exists locally, use it. Otherwise use pretrained.
            if Path(self.model_path).exists() and any(Path(self.model_path).iterdir()):
                logger.info("Loading fine-tuned local model...")
                self.processor = LayoutLMv3Processor.from_pretrained(
                    self.model_path,
                    apply_ocr=False  # We provide our own OCR
                )
                self.model = LayoutLMv3ForTokenClassification.from_pretrained(
                    self.model_path,
                    num_labels=len(GST2A_LABELS),
                    id2label=ID2LABEL,
                    label2id=LABEL2ID
                )
            else:
                logger.info("No fine-tuned model found. Loading pretrained microsoft/layoutlmv3-base...")
                logger.info("This will download ~500MB on first run.")
                self.processor = LayoutLMv3Processor.from_pretrained(
                    "microsoft/layoutlmv3-base",
                    apply_ocr=False
                )
                self.model = LayoutLMv3ForTokenClassification.from_pretrained(
                    "microsoft/layoutlmv3-base",
                    num_labels=len(GST2A_LABELS),
                    id2label=ID2LABEL,
                    label2id=LABEL2ID,
                    ignore_mismatched_sizes=True
                )
            
            self.model = self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            logger.info(f"LayoutLMv3 loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load LayoutLMv3: {e}")
            raise
    
    def normalize_bbox(self, bbox, width: int, height: int) -> List[int]:
        """Normalize bounding box to 0-1000 range (LayoutLM requirement)."""
        # bbox from PaddleOCR: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        # We need [left, top, right, bottom] normalized to 0-1000
        x_coords = [pt[0] for pt in bbox]
        y_coords = [pt[1] for pt in bbox]
        
        left = min(x_coords)
        top = min(y_coords)
        right = max(x_coords)
        bottom = max(y_coords)
        
        return [
            int(1000 * left / width),
            int(1000 * top / height),
            int(1000 * right / width),
            int(1000 * bottom / height)
        ]
    
    def classify_fields(self, ocr_result: Dict, image_path: str) -> Dict[str, Any]:
        """
        Takes OCR output and classifies each text into GST field types.
        
        Returns dict of field_name → [values]
        """
        if not self._loaded:
            self.load()
        
        from PIL import Image
        
        try:
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            
            # Prepare inputs for LayoutLMv3
            words = []
            boxes = []
            
            for line in ocr_result.get("lines", []):
                if line.get("bbox") and line.get("text"):
                    word = line["text"]
                    bbox_normalized = self.normalize_bbox(line["bbox"], width, height)
                    words.append(word)
                    boxes.append(bbox_normalized)
            
            if not words:
                logger.warning("No text with bounding boxes available for LayoutLM")
                return self._fallback_extract(ocr_result)
            
            # Process with LayoutLMv3
            encoding = self.processor(
                image,
                words,
                boxes=boxes,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )
            
            # Move to GPU
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**encoding)
            
            # Get predicted labels
            predictions = outputs.logits.argmax(-1).squeeze().tolist()
            token_boxes = encoding["bbox"].squeeze().tolist()
            
            # Map predictions back to words
            classified_fields = {}
            current_field = None
            current_value = []
            
            word_ids = encoding.word_ids() if hasattr(encoding, 'word_ids') else None
            
            for idx, (pred, box) in enumerate(zip(predictions, token_boxes)):
                label = ID2LABEL.get(pred, "O")
                
                if label.startswith("B-"):  # Beginning of a new field
                    # Save previous field
                    if current_field and current_value:
                        field_name = current_field.replace("B-", "").replace("I-", "")
                        if field_name not in classified_fields:
                            classified_fields[field_name] = []
                        classified_fields[field_name].append(" ".join(current_value))
                    
                    current_field = label
                    # Find the corresponding word
                    current_value = [words[min(idx, len(words)-1)]] if idx < len(words) else []
                    
                elif label.startswith("I-") and current_field:  # Continuation
                    if idx < len(words):
                        current_value.append(words[idx])
            
            # Save last field
            if current_field and current_value:
                field_name = current_field.replace("B-", "").replace("I-", "")
                if field_name not in classified_fields:
                    classified_fields[field_name] = []
                classified_fields[field_name].append(" ".join(current_value))
            
            logger.info(f"LayoutLMv3 classified {len(classified_fields)} field types")
            return classified_fields
            
        except Exception as e:
            logger.error(f"LayoutLMv3 classification failed: {e}")
            return self._fallback_extract(ocr_result)
    
    def _fallback_extract(self, ocr_result: Dict) -> Dict:
        """
        Rule-based fallback extraction when LayoutLMv3 fails.
        Uses regex patterns for common GST fields.
        """
        import re
        
        full_text = ocr_result.get("full_text", "")
        fields = {}
        
        patterns = {
            "GSTIN": r"\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}\b",
            "INVOICE_NO": r"(?:Invoice\s*No\.?|Inv\.?\s*No\.?)\s*:?\s*([A-Z0-9/-]+)",
            "INVOICE_DATE": r"\b\d{2}[/-]\d{2}[/-]\d{4}\b",
            "TAXABLE_VALUE": r"(?:Taxable\s*Value|Taxable\s*Amt)\.?\s*:?\s*([\d,]+\.?\d*)",
            "IGST": r"(?:IGST)\s*:?\s*([\d,]+\.?\d*)",
            "CGST": r"(?:CGST)\s*:?\s*([\d,]+\.?\d*)",
            "SGST": r"(?:SGST|UTGST)\s*:?\s*([\d,]+\.?\d*)",
        }
        
        for field, pattern in patterns.items():
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            if matches:
                fields[field] = matches
        
        logger.info(f"Fallback extraction found {len(fields)} fields")
        return fields


# Singleton
_layoutlm_instance = None

def get_layoutlm(model_path: str, use_gpu: bool = True) -> LayoutLMv3Service:
    global _layoutlm_instance
    if _layoutlm_instance is None:
        _layoutlm_instance = LayoutLMv3Service(model_path, use_gpu)
    return _layoutlm_instance