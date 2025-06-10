import os
import fitz  # PyMuPDF
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from processor.data_models import ElementType
from partitioner_max.libs.hash_utils import compute_sha256

def generate_secure_filename(prefix: str = "img") -> str:
    """Generate a secure random filename using SHA-256."""
    random_bytes = os.urandom(32)
    hash_str = compute_sha256(random_bytes.decode('latin-1'))
    return f"{prefix}_{hash_str[:16]}.png"

def extract_images_from_pdf(
    pdf_path: str, 
    output_dir: str,
    min_width: int = 50,
    min_height: int = 50
) -> List[Dict]:
    """Extract images from a PDF and save them to the output directory.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted images
        min_width: Minimum width of images to extract (pixels)
        min_height: Minimum height of images to extract (pixels)
        
    Returns:
        List of image elements with metadata
    """
    import logging
    logger = logging.getLogger("pdf_processor")
    
    try:
        # Create images directory if it doesn't exist
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        logger.debug(f"Saving images to: {images_dir}")
        
        image_elements = []
        doc = fitz.open(pdf_path)
        logger.info(f"Processing {len(doc)} pages for images")
        
        for page_num in range(len(doc)):
            try:
                page = doc.load_page(page_num)
                image_list = page.get_images(full=True)
                
                if not image_list:
                    logger.debug(f"No images found on page {page_num + 1}")
                    continue
                    
                logger.debug(f"Found {len(image_list)} images on page {page_num + 1}")
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        
                        if not base_image or 'image' not in base_image:
                            logger.warning(f"Could not extract image {img_index} on page {page_num + 1}")
                            continue
                            
                        image_bytes = base_image["image"]
                        
                        # Skip small images (likely icons or decorations)
                        if base_image["width"] < min_width or base_image["height"] < min_height:
                            logger.info(f"Skipping image on page {page_num + 1} due to small size: {base_image['width']}x{base_image['height']}px (min: {min_width}x{min_height}px)")
                            continue
                            
                        # Generate secure filename and save image
                        filename = generate_secure_filename()
                        image_path = os.path.join(images_dir, filename)
                        
                        with open(image_path, "wb") as f:
                            f.write(image_bytes)
                        
                        # Get image position on page
                        bbox = None
                        for image in page.get_image_rects(xref):
                            bbox = (
                                float(image.x0),
                                float(image.y0),
                                float(image.x1),
                                float(image.y1)
                            )
                            break
                        
                        if bbox:
                            img_element = {
                                "text": filename,
                                "element_type": ElementType.FIGURE.value,
                                "bbox": bbox,
                                "page_number": page_num + 1,  # 1-based
                                "metadata": {
                                    "image_path": os.path.relpath(image_path, output_dir).replace('\\', '/'),
                                    "width": base_image["width"],
                                    "height": base_image["height"],
                                    "format": base_image.get("ext", "png").upper(),
                                    "size_kb": round(len(image_bytes) / 1024, 2)
                                }
                            }
                            image_elements.append(img_element)
                            logger.debug(f"Successfully created element for image: {img_element['metadata']['image_path']} on page {page_num + 1} at {bbox}")
                        else:
                            logger.warning(f"Image {img_index} on page {page_num + 1} was extracted and saved as {filename}, but NO BBOX was found by page.get_image_rects(xref={xref}). Element NOT created.")
                            
                    except Exception as img_err:
                        logger.error(f"Error processing image {img_index} on page {page_num + 1}: {img_err}", exc_info=True)
                        continue
                        
            except Exception as page_err:
                logger.error(f"Error processing page {page_num + 1} for images: {page_err}", exc_info=True)
                continue
                
    except Exception as e:
        logger.error(f"Error in extract_images_from_pdf: {e}", exc_info=True)
        return []
        
    finally:
        if 'doc' in locals():
            doc.close()
    
    logger.info(f"extract_images_from_pdf returning {len(image_elements)} image element(s).")
    return image_elements
