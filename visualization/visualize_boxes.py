import fitz  # PyMuPDF
import json
from typing import List, Dict, Any, Tuple
import argparse
from pathlib import Path
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.bbox_utils import calculate_iou, merge_bboxes

def should_merge_elements(elem1: Dict[str, Any], elem2: Dict[str, Any], iou_threshold: float = 0.1, y_threshold: float = 5) -> bool:
    """
    Determine if two elements should be merged based on their spatial relationship and properties.
    """
    # Don't merge elements from different pages
    if elem1['page_number'] != elem2['page_number']:
        return False
        
    # Don't merge elements of different types
    if elem1['element_type'] != elem2['element_type']:
        return False
        
    bbox1 = elem1['bbox']
    bbox2 = elem2['bbox']
    
    # Check vertical overlap
    y_overlap = min(abs(bbox1[3] - bbox2[1]), abs(bbox1[1] - bbox2[3]))
    if y_overlap > y_threshold:
        return False
    
    # Check horizontal proximity and overlap
    iou = calculate_iou(bbox1, bbox2)
    return iou > iou_threshold

def merge_element_group(elements: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge a group of elements into a single element.
    """
    if not elements:
        return None
        
    merged_bbox = merge_bboxes([elem['bbox'] for elem in elements])
    merged_text = ' '.join(elem['text'] for elem in elements)
    
    # Use the metadata from the largest element
    largest_elem = max(elements, key=lambda e: (e['bbox'][2] - e['bbox'][0]) * (e['bbox'][3] - e['bbox'][1]))
    
    return {
        'text': merged_text,
        'element_type': largest_elem['element_type'],
        'bbox': merged_bbox,
        'page_number': largest_elem['page_number'],
        'metadata': largest_elem['metadata']
    }

def merge_overlapping_elements(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge elements that overlap or are very close to each other.
    """
    if not elements:
        return []
    
    # Sort elements by page and vertical position
    elements = sorted(elements, key=lambda e: (e['page_number'], -e['bbox'][1]))
    
    merged_elements = []
    current_group = [elements[0]]
    
    for elem in elements[1:]:
        should_merge = any(should_merge_elements(e, elem) for e in current_group)
        
        if should_merge:
            current_group.append(elem)
        else:
            if current_group:
                merged_elements.append(merge_element_group(current_group))
            current_group = [elem]
    
    if current_group:
        merged_elements.append(merge_element_group(current_group))
    
    return merged_elements

def draw_element_boxes(pdf_path: str, output_path: str, elements: List[Dict[str, Any]], min_confidence: float = 0.0):
    """
    Draw colored boxes around elements in the PDF and save as a new file.
    
    Args:
        pdf_path: Path to the source PDF
        output_path: Path where to save the annotated PDF
        elements: List of elements with bounding boxes
        min_confidence: Minimum confidence threshold for showing elements
    """
    # Open the PDF
    doc = fitz.open(pdf_path)
    
    # Define colors for different element types (RGB format)
    colors = {
        # Text elements
        "TITLE": (102/255, 102/255, 255/255),       # Blue
        "TEXT": (153/255, 0/255, 76/255),           # Dark Red / Maroon
        "BLOCK_EQUATION": (0/255, 255/255, 0/255),  # Pure Green
        
        # List elements
        "LIST_ITEM": (40/255, 169/255, 92/255),     # Dark Green
        "LIST": (40/255, 169/255, 92/255),          # Dark Green (same as LIST_ITEM)
        "INDEX": (40/255, 169/255, 92/255),         # Dark Green (same as LIST_ITEM)
        
        # Table elements (for future use)
        "TABLE_BODY": (204/255, 204/255, 0/255),    # Yellow
        "TABLE_CAPTION": (255/255, 255/255, 102/255), # Light Yellow
        "TABLE_FOOTNOTE": (229/255, 255/255, 204/255), # Light Green
        
        # Image elements (for future use)
        "IMAGE_BODY": (153/255, 255/255, 51/255),    # Bright Green
        "IMAGE_CAPTION": (102/255, 178/255, 255/255), # Light Blue
        "IMAGE_FOOTNOTE": (255/255, 178/255, 102/255), # Orange
        
        # Other
        "DISCARDED": (158/255, 158/255, 158/255)    # Gray
    }
    
    # Process each page
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Get elements for this page
        page_elements = [e for e in elements 
                        if e["page_number"] == page_num + 1 
                        and e["metadata"].get("confidence", 1.0) >= min_confidence]
        
        # Get page dimensions
        page_height = page.rect.height
        
        for element in page_elements:
            # Extract coordinates - handle both string and list/tuple formats
            bbox = element["bbox"]
            
            # Convert string bbox to tuple if needed
            if isinstance(bbox, str):
                try:
                    # Handle string format like "(x0, y0, x1, y1)"
                    bbox = bbox.strip('()')
                    bbox = [float(x.strip()) for x in bbox.split(',')]
                except (ValueError, AttributeError) as e:
                    print(f"Error parsing bbox '{bbox}': {e}")
                    continue
            
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                print(f"Invalid bbox format: {bbox}")
                continue
                
            try:
                x0, y0, x1, y1 = [float(coord) for coord in bbox]
            except (ValueError, TypeError) as e:
                print(f"Error converting bbox coordinates to float: {e}")
                continue
            
            # Get color for element type (black if type unknown)
            element_type = element.get("element_type", "").replace("ElementType.", "")
            color = colors.get(element_type, (0, 0, 0))
            
            # Draw rectangle
            rect = fitz.Rect(x0, y0, x1, y1)
            page.draw_rect(rect, color=color, width=1)
            
            # Add element type label above the box
            confidence = element["metadata"].get("confidence", 1.0)
            label = f"{element['element_type']} ({confidence:.2f})"
            text_point = fitz.Point(x0, y0 - 5)
            page.insert_text(text_point, label, color=color, fontsize=8)
    
    # Save the annotated PDF
    doc.save(output_path)
    doc.close()

def print_statistics(elements: List[Dict[str, Any]]):
    """Print statistics about the elements."""
    # Count elements by type
    element_types = {}
    for element in elements:
        etype = element["element_type"]
        element_types[etype] = element_types.get(etype, 0) + 1
    
    print("\nElement Type Statistics:")
    for etype, count in sorted(element_types.items()):
        print(f"{etype}: {count}")
    
    # Show sample of each type
    print("\nSample Elements by Type:")
    shown_types = set()
    for element in elements:
        etype = element["element_type"]
        if etype not in shown_types:
            shown_types.add(etype)
            confidence = element["metadata"].get("confidence", "N/A")
            print(f"\nElement Type: {etype}")
            print(f"Page: {element['page_number']}")
            print(f"Confidence: {confidence}")
            print(f"Text: {element['text'][:100]}...")
            print(f"BBox: {element['bbox']}")

def main():
    parser = argparse.ArgumentParser(description="Visualize element boxes in a PDF")
    parser.add_argument("--pdf", default="sample.pdf", help="Path to source PDF")
    parser.add_argument("--json", default="sample_elements.json", help="Path to elements JSON")
    parser.add_argument("--output", default=None, help="Path for annotated PDF output")
    parser.add_argument("--min-confidence", type=float, default=0.0, 
                       help="Minimum confidence threshold (0.0 to 1.0)")
    args = parser.parse_args()
    
    # Set default output path if not specified
    if args.output is None:
        pdf_path = Path(args.pdf)
        args.output = str(pdf_path.parent / f"{pdf_path.stem}_annotated.pdf")
    
    # Load elements from JSON
    with open(args.json, 'r') as f:
        data = json.load(f)
        elements = data["elements"]
    
    # Merge overlapping elements
    elements = merge_overlapping_elements(elements)
    
    # Print statistics
    print_statistics(elements)
    
    # Draw boxes and save
    draw_element_boxes(args.pdf, args.output, elements, args.min_confidence)
    print(f"\nAnnotated PDF saved to: {args.output}")

if __name__ == "__main__":
    main()