import argparse
import json
import os
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

from processor import process_pdf
from utils import save_json, load_json, setup_logger

logger = setup_logger()

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process PDF files to extract and classify text elements"
    )
    
    parser.add_argument(
        "input_path",
        help="Path to input PDF file or directory containing PDFs"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Path to output directory (default: same as input)",
        default=None
    )
    
    parser.add_argument(
        "--config",
        help="Path to config file with LAParams settings",
        default=None
    )
    
    parser.add_argument(
        "--log-file",
        help="Path to log file",
        default=None
    )
    
    parser.add_argument(
        "-v", "--verbose",
        help="Increase output verbosity",
        action="store_true"
    )
    
    parser.add_argument(
        "--visualize",
        help="Generate visualization of the PDF with element bounding boxes",
        action="store_true"
    )
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load LAParams configuration from file."""
    if not config_path:
        # Use default config
        return {
            'line_margin': 0.3,
            'word_margin': 0.1,
            'char_margin': 0.5,
            'boxes_flow': 0.5,
            'detect_vertical': False,
            'all_texts': False
        }
    
    return load_json(config_path)

@dataclass
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float
    element_type: str
    text: str
    page_num: int
    confidence: float

def draw_bbox_on_pdf(pdf_path: str, elements: List[dict], output_dir: str) -> None:
    """Draw bounding boxes directly on the original PDF pages."""
    try:
        # Open the original PDF
        doc = fitz.open(pdf_path)
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output PDF path
        output_pdf = output_dir / f"{Path(pdf_path).stem}_annotated.pdf"
        
        # Group elements by page
        pages = {}
        for elem in elements:
            page_num = elem['page_number'] - 1  # Convert to 0-based index
            if page_num not in pages:
                pages[page_num] = []
            pages[page_num].append(elem)
        
        # Process each page
        for page_num, page_elems in pages.items():
            if page_num >= len(doc):
                continue
                
            page = doc[page_num]
            
            # Draw each bounding box on the original page
            for elem in page_elems:
                bbox = elem.get('bbox')
                if not bbox or len(bbox) != 4:
                    continue
                
                # Define colors based on element type (RGB format)
                colors = {
                    'TITLE': (1, 0, 0),      # Red
                    'HEADING': (0, 1, 0),    # Green
                    'LIST_ITEM': (0, 0, 1),  # Blue
                    'NARRATIVE_TEXT': (0.5, 0, 0.5),  # Purple
                    'FIGURE': (1, 0.5, 0),   # Orange
                    'TABLE': (0, 0.5, 0.5),  # Teal
                    'EQUATION': (1, 0, 1),   # Magenta
                }
                
                color = colors.get(elem['element_type'], (0.5, 0.5, 0.5))  # Default gray
                
                # Draw the bounding box
                rect = fitz.Rect(*bbox)
                page.draw_rect(rect, color=color, width=1.5)
                
                # Add element type label
                label = f"{elem['element_type']}"
                if 'metadata' in elem and 'confidence' in elem['metadata']:
                    label += f" ({elem['metadata']['confidence']:.2f})"
                
                # Add text annotation near the top-left corner of the box
                page.insert_text(
                    (bbox[0] + 2, bbox[1] - 2),
                    label,
                    fontsize=8,
                    color=color
                )
        
        # Save the annotated PDF
        doc.save(output_pdf, deflate=True)
        logger.info(f"Saved annotated PDF to {output_pdf}")
        
    except Exception as e:
        logger.error(f"Error generating annotated PDF: {str(e)}")
        raise
    finally:
        if 'doc' in locals():
            doc.close()

def process_single_pdf(
    pdf_path: str,
    output_dir: str,
    config: Dict[str, Any],
    visualize: bool = False
) -> None:
    """Process a single PDF file."""
    try:
        logger.info(f"Processing {pdf_path}")
        
        # Process PDF
        elements = process_pdf(pdf_path, config)
        
        # Prepare output
        output = {
            'source_file': pdf_path,
            'elements': [
                {
                    'text': elem.text,
                    'element_type': elem.element_type.name,
                    'bbox': elem.bbox,
                    'page_number': elem.page_number,
                    'metadata': {
                        'confidence': elem.metadata.confidence,
                        'style_info': elem.metadata.style_info,
                        'list_group_id': elem.metadata.list_group_id
                    }
                }
                for elem in elements
            ]
        }
        
        # Save output
        output_path = Path(output_dir) / f"{Path(pdf_path).stem}_elements.json"
        save_json(output, str(output_path))
        logger.info(f"Saved output to {output_path}")
        
        # Generate visualization if requested
        if visualize:
            draw_bbox_on_pdf(pdf_path, output['elements'], output_dir)
        
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {str(e)}")
        raise

def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    if args.verbose:
        logger.setLevel("DEBUG")
    if args.log_file:
        setup_logger(log_file=args.log_file)
    
    # Load config
    config = load_config(args.config)
    
    # Handle input path
    input_path = Path(args.input_path)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return
    
    # Determine output directory
    output_dir = args.output if args.output else input_path.parent
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process files
    if input_path.is_file():
        if input_path.suffix.lower() != '.pdf':
            logger.error(f"Input file is not a PDF: {input_path}")
            return
        process_single_pdf(str(input_path), output_dir, config, visualize=args.visualize)
    else:
        # Process all PDFs in directory
        for pdf_path in input_path.glob('*.pdf'):
            process_single_pdf(str(pdf_path), output_dir, config, visualize=args.visualize)

if __name__ == "__main__":
    main()
