import os
import logging
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path
from typing import List

logger = logging.getLogger("gst2_fastapi.pdf_converter")

# Path to poppler — update this to where you installed poppler
POPPLER_PATH = r"C:\poppler\Library\bin"

def convert_file_to_images(file_path: str, output_dir: str) -> List[str]:
    """
    Takes any file (PDF, JPG, PNG, JPEG) and returns a list of image paths.
    All pages of a PDF are converted to separate images.
    
    Args:
        file_path: Path to the uploaded file
        output_dir: Directory to save converted images
        
    Returns:
        List of image file paths
    """
    file_path = Path(file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extension = file_path.suffix.lower()
    image_paths = []
    
    logger.info(f"Converting file: {file_path.name} (type: {extension})")
    
    if extension == ".pdf":
        # Convert each PDF page to an image
        try:
            pages = convert_from_path(
                str(file_path),
                dpi=300,          # High DPI for better OCR accuracy
                poppler_path=POPPLER_PATH,
                fmt="PNG"
            )
            for i, page in enumerate(pages):
                img_path = output_dir / f"{file_path.stem}_page_{i+1}.png"
                page.save(str(img_path), "PNG")
                image_paths.append(str(img_path))
                logger.info(f"Converted page {i+1} → {img_path.name}")
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            raise
            
    elif extension in [".jpg", ".jpeg", ".png"]:
        # Already an image — just copy/standardize it
        img = Image.open(str(file_path))
        
        # Convert to RGB if needed (some PNGs have alpha channel)
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Save as high-quality PNG
        img_path = output_dir / f"{file_path.stem}_page_1.png"
        img.save(str(img_path), "PNG")
        image_paths.append(str(img_path))
    
    else:
        raise ValueError(f"Unsupported file type: {extension}. Only PDF, JPG, PNG, JPEG are supported.")
    
    logger.info(f"Converted {len(image_paths)} image(s) from {file_path.name}")
    return image_paths