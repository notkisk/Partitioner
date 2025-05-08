import argparse
import json
from pathlib import Path
from typing import Dict, Any

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

def process_single_pdf(
    pdf_path: str,
    output_dir: str,
    config: Dict[str, Any]
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
        process_single_pdf(str(input_path), output_dir, config)
    else:
        # Process all PDFs in directory
        for pdf_path in input_path.glob('*.pdf'):
            process_single_pdf(str(pdf_path), output_dir, config)

if __name__ == "__main__":
    main()
