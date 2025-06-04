import argparse
import json
import os
import sys
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
import shutil
from utils.image_utils import extract_images_from_pdf
from processor.caption_detector import find_caption_for_figure

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s [%(name)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Set pdfminer log level to WARNING to reduce verbosity
logging.getLogger("pdfminer").setLevel(logging.WARNING)

# Import after basic config to ensure logging is configured
from processor import process_pdf
from utils import save_json, load_json, setup_logger

# Get the logger
logger = logging.getLogger("pdf_processor")
logger.setLevel(logging.DEBUG)

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
    
    parser.add_argument(
        "--no-split-detection",
        action='store_false', dest='detect_splits',
        help='Disable detection of split content across pages'
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

def draw_bbox_on_pdf(pdf_path: str, elements: List[dict], output_dir: str, structure_info: dict = None) -> None:
    """Draw bounding boxes on the original PDF and save the result.
    
    Args:
        pdf_path: Path to the original PDF
        elements: List of element dictionaries
        output_dir: Directory to save the output PDF
        structure_info: Document structure information including split markers
    """
    try:
        # Create output filename
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_pdf = os.path.join(output_dir, f"{base_name}_annotated.pdf")
        
        # Open the PDF
        doc = fitz.open(pdf_path)
        
        # Group elements by page
        elements_by_page = {}
        for elem in elements:
            page_num = elem.get('page_number', 1) - 1  # Convert to 0-based
            if page_num not in elements_by_page:
                elements_by_page[page_num] = []
            elements_by_page[page_num].append(elem)
        
        # Extract list groups for visualization
        list_groups = [e for e in elements if e.get('element_type') == 'LIST']
        
        # Extract split markers if available
        split_markers = structure_info.get('split_markers', []) if structure_info else []
        
        # Define colors based on element type (RGB format)
        colors = {
            # Text elements
            'TITLE': (1, 0, 0),          # Red
            'TEXT': (0.5, 0, 0.5),       # Purple
            'BLOCK_EQUATION': (1, 0, 1),  # Magenta
            
            # List elements
            'LIST_ITEM': (0, 0, 1),      # Blue
            'LIST': (0.2, 0.6, 1),       # Lighter blue for list groups
            'INDEX': (0, 1, 1),          # Cyan
            
            # Table elements
            'TABLE_BODY': (0, 0.5, 0),   # Dark Green
            'TABLE_CAPTION': (0, 0.8, 0), # Green
            'TABLE_FOOTNOTE': (0, 0.6, 0.3), # Light Green
            
            # Figure elements
            'FIGURE': (0, 0.7, 0.7),        # Teal
            'FIGURE_CAPTION': (0, 0.9, 0.9),  # Light Teal
            'FIGURE_FOOTNOTE': (0, 0.8, 0.8), # Lighter Teal
            
            # Split elements
            'SPLIT_CONTINUATION': (1, 0.5, 0.5),  # Light red for continuations
            'SPLIT_HEADER': (0.5, 0.8, 0.5),     # Light green for headers
            'SPLIT_FOOTER': (0.5, 0.8, 0.5),     # Light green for footers
            'SPLIT_TABLE': (0.5, 0.8, 1.0),      # Light blue for tables
            'SPLIT_LIST': (0.8, 0.6, 1.0),       # Light purple for lists
            'SPLIT_PARAGRAPH': (1.0, 0.8, 0.6),   # Light orange for paragraphs
            
            # Other
            'DISCARDED': (0.5, 0.5, 0.5) # Gray
        }
        
        # Process each page
        for page_num, page in enumerate(doc):
            # Get elements for this page (1-based to 0-based conversion)
            page_elements = elements_by_page.get(page_num, [])
            
            # Get split markers for this page
            page_split_markers = [m for m in split_markers 
                                if m.get('source_page') == page_num + 1 or 
                                   m.get('target_page') == page_num + 1]
            
            # Draw split markers first (so they're underneath other elements)
            for marker in page_split_markers:
                if marker.get('source_page') == page_num + 1 and marker.get('source_bbox'):
                    bbox = marker['source_bbox']
                    split_type = marker.get('split_type', 'SPLIT_CONTINUATION')
                    color = colors.get(f'SPLIT_{split_type}', (1, 0.5, 0.5))
                    
                    # Draw a dashed rectangle for the split area
                    rect = fitz.Rect(*bbox)
                    page.draw_rect(rect, color=color, width=1.0, dashes="[2] 2")
                    
                    # Add arrow pointing to the next page
                    if marker.get('target_bbox') and marker.get('target_page') > marker.get('source_page'):
                        x0, y0 = bbox[2] - 10, bbox[3] - 10
                        x1, y1 = x0 + 10, y0 + 10
                        page.draw_line((x0, y0), (x1, y1), color=color, width=1.0)
                        page.draw_line((x0, y1), (x1, y0), color=color, width=1.0)
            
            # Draw bounding boxes for each element
            for elem in page_elements:
                bbox = elem.get('bbox')
                
                # Handle different bbox formats
                if isinstance(bbox, str):
                    try:
                        # Handle string format like "(x0, y0, x1, y1)" or "[x0, y0, x1, y1]"
                        bbox = bbox.strip('()[]')
                        bbox = [float(x.strip()) for x in bbox.split(',')]
                    except (ValueError, AttributeError) as e:
                        logger.warning(f"Invalid bbox format {bbox}: {e}")
                        continue
                
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    logger.warning(f"Invalid bbox format: {bbox}")
                    continue
                
                # Ensure coordinates are floats
                try:
                    bbox = [float(coord) for coord in bbox]
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error converting bbox coordinates to float: {e}")
                    continue
                
                # Skip invalid coordinates
                if any(not isinstance(coord, (int, float)) for coord in bbox):
                    logger.warning(f"Invalid coordinate values in bbox: {bbox}")
                    continue
                
                # Check if this element is part of a split
                is_split = False
                split_type = None
                if 'metadata' in elem and isinstance(elem['metadata'], dict):
                    if 'is_split' in elem['metadata'] and elem['metadata']['is_split']:
                        is_split = True
                    if 'split_info' in elem['metadata'] and elem['metadata']['split_info']:
                        is_split = True
                        split_type = elem['metadata']['split_info'][0].get('type')
                
                # Get and normalize the element type
                elem_type = str(elem.get('element_type', 'UNKNOWN'))
                
                # Clean up the element type string
                if 'ElementType.' in elem_type:
                    elem_type = elem_type.split('ElementType.')[-1]
                elem_type = elem_type.upper()
                
                # Log the element type for debugging
                logger.debug(f"Drawing element type: {elem_type}")
                
                # Determine color and width based on element type and split status
                if is_split and split_type:
                    color_key = f'SPLIT_{split_type}'
                    color = colors.get(color_key, (1, 0.5, 0.5))
                    width = 2.0  # Thicker border for split elements
                    logger.debug(f"Using split color for {color_key}: {color}")
                else:
                    color = colors.get(elem_type, (0.5, 0.5, 0.5))
                    width = 1.5
                    logger.debug(f"Using color for {elem_type}: {color}")
                
                try:
                    # Draw the bounding box
                    rect = fitz.Rect(*bbox)
                    page.draw_rect(rect, color=color, width=width, dashes="[2] 2" if is_split else None)
                    
                    # Prepare label text
                    display_type = elem_type.replace('_', ' ').title()
                    label = f"{display_type}"
                    
                    # Add split info to label if applicable
                    if is_split and split_type:
                        split_display = split_type.split('_')[-1].title()
                        label = f"{split_display}: {label}"
                    
                    # Add confidence if available
                    if 'metadata' in elem and isinstance(elem['metadata'], dict) and 'confidence' in elem['metadata']:
                        conf = elem['metadata']['confidence']
                        if isinstance(conf, (int, float)):
                            label += f" ({conf:.2f})"
                    
                    # Add text annotation near the top-left corner of the box
                    try:
                        page.insert_text(
                            (bbox[0] + 2, bbox[1] - 2),
                            label,
                            fontsize=8,
                            color=color
                        )
                    except Exception as e:
                        logger.warning(f"Failed to add text annotation: {e}")
                        
                except Exception as e:
                    logger.error(f"Error drawing bbox {bbox} for {elem_type}: {e}")
        
        # Save the annotated PDF
        doc.save(output_pdf, deflate=True)
        logger.info(f"Saved annotated PDF to {output_pdf}")
        
    except Exception as e:
        logger.error(f"Error generating annotated PDF: {str(e)}")
        raise
    finally:
        if 'doc' in locals():
            doc.close()

def process_single_pdf(pdf_path: str, output_dir: str, config: Dict[str, Any] = None, visualize: bool = False) -> Optional[tuple]:
    """Process a single PDF file and optionally visualize the results.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save output files (will create subdirectory for this document)
        config: Configuration dictionary for PDF processing
        visualize: Whether to generate visualization
        
    Returns:
        Tuple containing (elements_json_path, structure_json_path) if successful, None otherwise
    """
    try:
        logger.debug(f"Starting to process PDF: {pdf_path}")
        
        # Create document-specific output directory
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        doc_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(doc_output_dir, exist_ok=True)
        logger.debug(f"Document output directory: {doc_output_dir}")
        
        # Process the PDF with document structure analysis first to get table areas
        elements, structure_info = process_pdf(
            pdf_path=pdf_path,
            image_output_dir=doc_output_dir,
            la_params=config if config and isinstance(config, dict) else None,
            handle_splits=True
        )
        
        # Extract images after processing to ensure output directory exists
        logger.info("Extracting images from PDF...")
        try:
            image_elements = extract_images_from_pdf(pdf_path, doc_output_dir)
            num_extracted = len(image_elements) if isinstance(image_elements, list) else 0
            logger.info(f"Extracted {num_extracted} image(s) from PDF via extract_images_from_pdf.")

            if image_elements and isinstance(image_elements, list) and num_extracted > 0:
                logger.info(f"Attempting to add {num_extracted} image element(s) to the main 'elements' list.")
                elements_len_before = len(elements)
                try:
                    elements.extend(image_elements) # image_elements is List[Dict]
                    elements_len_after = len(elements)
                    
                    if elements_len_after > elements_len_before:
                        added_count = elements_len_after - elements_len_before
                        logger.info(f"SUCCESS: Added {added_count} image element(s). Main 'elements' list size: {elements_len_before} -> {elements_len_after}.")
                        # For the first few added elements (which are dicts), log their type and text
                        for i in range(min(added_count, 2)): # Log first 2 added images
                            new_idx = elements_len_before + i
                            if new_idx < elements_len_after and isinstance(elements[new_idx], dict):
                                logger.debug(f"  Added element [{i+1}/{added_count}] (dict): type='{elements[new_idx].get('element_type', 'N/A')}', text='{str(elements[new_idx].get('text', 'N/A'))[:50]}...'" )
                            else:
                                logger.debug(f"  Added element [{i+1}/{added_count}] (type: {type(elements[new_idx])})")
                    else:
                        logger.warning(f"WARNING: `elements.extend` did not increase list size, though {num_extracted} image(s) were extracted. List size: {elements_len_before} -> {elements_len_after}.")
                except Exception as ex_extend:
                    logger.error(f"CRITICAL ERROR during `elements.extend(image_elements)`: {ex_extend}", exc_info=True)
            elif num_extracted == 0:
                logger.info("No image elements were extracted by extract_images_from_pdf, or the list was empty.")
            else: # Should not happen if num_extracted is derived correctly
                logger.warning(f"image_elements is not a non-empty list (type: {type(image_elements)}, len: {num_extracted}). No image elements added.")
        except Exception as e:
            logger.error(f"Error in image extraction or addition block: {e}", exc_info=True)
        
        # Interleave images with text by sorting now
        elements.sort(key=lambda e: (e['page_number'] if isinstance(e, dict) else e.page_number, -(e['bbox'][1] if isinstance(e, dict) else e.bbox[1])))
        
        def convert_value(value):
            if value is None:
                return None
            if isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, list):
                return [v for v in (convert_value(v) for v in value) if v is not None]
            if isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items() if v is not None}
            if hasattr(value, '__dict__'):
                return convert_value({k: v for k, v in value.__dict__.items() if not k.startswith('_')})
            return str(value)
                
        def serialize_metadata(metadata):
            if not metadata:
                return {}
                
            result = {}
            
            if hasattr(metadata, 'style_info') and metadata.style_info:
                result['style'] = metadata.style_info
                
            if hasattr(metadata, 'coordinates') and metadata.coordinates:
                coords = {
                    'x0': getattr(metadata.coordinates, 'x0', 0),
                    'y0': getattr(metadata.coordinates, 'y0', 0),
                    'x1': getattr(metadata.coordinates, 'x1', 0),
                    'y1': getattr(metadata.coordinates, 'y1', 0),
                    'page_width': getattr(metadata.coordinates, 'page_width', 0),
                    'page_height': getattr(metadata.coordinates, 'page_height', 0),
                    'relative_x': getattr(metadata.coordinates, 'relative_x', 0),
                    'relative_y': getattr(metadata.coordinates, 'relative_y', 0),
                    'indent_ratio': getattr(metadata.coordinates, 'indent_ratio', 0)
                }
                result['coordinates'] = coords
                
            for attr in ['list_group_id', 'confidence', 'page_number']:
                if hasattr(metadata, attr) and getattr(metadata, attr) is not None:
                    result[attr] = getattr(metadata, attr)
                
            if hasattr(metadata, 'cross_page_refs') and metadata.cross_page_refs:
                result['cross_page_refs'] = metadata.cross_page_refs
                
            if hasattr(metadata, 'table_data') and metadata.table_data:
                result['table'] = metadata.table_data
                
            return result
            
        def element_to_dict(element):
            if element is None:
                return None
                
            try:
                if isinstance(element, dict):
                    element_type = element.get('element_type', 'UNKNOWN')
                    # Ensure element_type is a string
                    if isinstance(element_type, dict):
                        if 'value' in element_type:
                            element_type = element_type['value']
                        else:
                            element_type = str(element_type)
                    elif not isinstance(element_type, str):
                        element_type = str(element_type)
                        
                    element_dict = {
                        'text': element.get('text',''),
                        'element_type': element_type or 'TEXT',  # Default to TEXT if empty
                        'bbox': element.get('bbox',(0.0,0.0,0.0,0.0)),  # Use raw bbox values instead of stringifying
                        'page_number': element.get('page_number',1),
                        'metadata': element.get('metadata',{})
                    }
                else:
                    element_text = getattr(element, 'text', '')
                    
                    # Get element type with fallback logic
                    element_type = 'UNKNOWN'
                    if hasattr(element, 'element_type'):
                        if hasattr(element.element_type, 'value'):
                            element_type = element.element_type.value
                        elif element.element_type is not None:
                            element_type = str(element.element_type)
                    
                    # Ensure element_type is a string and not empty
                    if not element_type or element_type == '{}':
                        element_type = 'UNKNOWN'
                    
                    # Handle bbox - ensure it's always a list of 4 floats
                    bbox = [0.0, 0.0, 0.0, 0.0]
                    if hasattr(element, 'bbox') and element.bbox is not None:
                        try:
                            if isinstance(element.bbox, str):
                                # Handle string format "(x0, y0, x1, y1)"
                                bbox = [float(x.strip()) for x in element.bbox.strip('()').split(',')]
                            elif isinstance(element.bbox, (list, tuple)) and len(element.bbox) == 4:
                                bbox = [float(x) for x in element.bbox]
                        except (ValueError, AttributeError, TypeError) as e:
                            logger.warning(f"Error processing bbox {element.bbox}: {e}")
                            bbox = [0.0, 0.0, 0.0, 0.0]
                    
                    # Create element metadata
                    metadata = {}
                    if hasattr(element, 'metadata') and element.metadata is not None:
                        if isinstance(element.metadata, dict):
                            metadata = element.metadata
                        elif hasattr(element.metadata, '__dict__'):
                            metadata = element.metadata.__dict__
                    
                    # Create the element dict
                    element_dict = {
                        'text': element_text,
                        'element_type': element_type if element_type != 'UNKNOWN' else 'TEXT',
                        'bbox': bbox,
                        'page_number': int(getattr(element, 'page_number', 1)),
                        'metadata': metadata
                    }
                
                return element_dict

            except Exception as e:
                logger.error(f"Error converting element to dict: {str(e)}", exc_info=True)
                return None

        elements_dict = []
        logger.info(f"Processing {len(elements)} elements")
        
        # sort elements by page and vertical position (y0)
        elements.sort(key=lambda e: (
            e['page_number'] if isinstance(e, dict) else e.page_number,
            -(e['bbox'][1] if isinstance(e, dict) else e.bbox[1])
        ))
        
        for idx, elem in enumerate(elements):
            try:
                if elem is None:
                    logger.debug(f"Skipping None element at index {idx}")
                    continue
                    
                converted = element_to_dict(elem)
                if converted is not None:
                    processed = convert_value(converted)
                    if processed is not None:
                        elements_dict.append(processed)
                        if len(elements_dict) % 100 == 0:
                            logger.debug(f"Processed {len(elements_dict)} elements so far...")
            except Exception as e:
                logger.error(f"Error processing element at index {idx}: {str(e)}", exc_info=True)
        
        logger.info(f"Successfully processed {len(elements_dict)} elements")
        if not elements_dict:
            logger.warning(f"No valid elements found in {pdf_path}")
            return None
            
        # Associate captions with figures without removing caption elements
        for fig in [el for el in elements_dict if el.get('element_type') == 'FIGURE']:
            idx = find_caption_for_figure(fig, elements_dict)
            if idx is not None:
                cap_el = elements_dict[idx]
                cap_el['element_type'] = 'FIGURE_CAPTION'
                fig.setdefault('metadata', {})['caption'] = cap_el['text'].strip()
        
        elements_json_path = os.path.join(doc_output_dir, f"{base_name}_elements.json")
        
        # group by page
        pages = []
        for pg in sorted({el['page_number'] for el in elements_dict}):
            pages.append({'page_number': pg, 'elements': [el for el in elements_dict if el['page_number'] == pg]})
        with open(elements_json_path, 'w', encoding='utf-8') as f:
            json.dump(pages, f, ensure_ascii=False, indent=2)

        logger.info(f"Processed {pdf_path} and saved results to {elements_json_path}")
        
        # Visualize the results if requested
        if visualize and elements_dict:
            try:
                draw_bbox_on_pdf(pdf_path, elements_dict, doc_output_dir, {})
            except Exception as e:
                logger.error(f"Error generating visualization: {str(e)}", exc_info=True)
        
        # Copy the original PDF to the output directory for reference
        try:
            shutil.copy2(pdf_path, os.path.join(doc_output_dir, os.path.basename(pdf_path)))
        except Exception as e:
            logger.warning(f"Could not copy original PDF: {e}")
            
        return elements_json_path, doc_output_dir
        
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {str(e)}", exc_info=True)
        return None

def main() -> None:
    """Main entry point."""
    try:
        logger.debug("Starting main function")
        args = parse_args()
        
        # Setup logging
        if args.verbose:
            logger.setLevel("DEBUG")
            logger.debug("Verbose mode enabled")
        if args.log_file:
            setup_logger(log_file=args.log_file)
            logger.debug(f"Logging to file: {args.log_file}")
        
        # Load config
        logger.debug("Loading configuration")
        config = load_config(args.config)
        logger.debug(f"Configuration loaded: {config}")
        
        # Handle input path
        input_path = Path(args.input_path).resolve()
        logger.debug(f"Input path: {input_path}")
        
        if not input_path.exists():
            logger.error(f"Input path does not exist: {input_path}")
            return
        
        # Determine output directory
        output_dir = Path(args.output).resolve() if args.output else input_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Output directory: {output_dir}")
        
        # Process files
        if input_path.is_file():
            logger.debug(f"Processing single file: {input_path}")
            if input_path.suffix.lower() != '.pdf':
                logger.error(f"Input file is not a PDF: {input_path}")
                return
            process_single_pdf(str(input_path), str(output_dir), config, visualize=args.visualize)
        else:
            # Process all PDFs in directory
            logger.debug(f"Processing directory: {input_path}")
            pdf_files = list(input_path.glob('*.pdf'))
            logger.debug(f"Found {len(pdf_files)} PDF files to process")
            
            for pdf_path in pdf_files:
                logger.debug(f"Processing file: {pdf_path}")
                process_single_pdf(str(pdf_path), str(output_dir), config, visualize=args.visualize)
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
