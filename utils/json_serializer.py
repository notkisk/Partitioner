from typing import List, Dict, Any, Optional
import json
import uuid
import os
from datetime import datetime, timezone
from dataclasses import asdict

from fast_pdf_parser import Element

def element_to_json_dict(element: Element, pdf_filepath: str) -> Dict[str, Any]:
    element_id = uuid.uuid4().hex
    min_x, min_y, max_x, max_y = element.bbox
    
    # Convert points to list format
    points = [
        [min_x, min_y],  # Bottom-left
        [min_x, max_y],  # Top-left
        [max_x, max_y],  # Top-right
        [max_x, min_y]   # Bottom-right
    ]
    
    # Prepare coordinates data
    coordinates_data = {
        "layout_width": element.page_width,
        "layout_height": element.page_height,
        "points": points,
        "system": "PDFSpace"  # Using PDF coordinate system
    }
    
    # Get file metadata
    filename = os.path.basename(pdf_filepath)
    file_directory = os.path.dirname(os.path.abspath(pdf_filepath))
    last_modified_timestamp = os.path.getmtime(pdf_filepath)
    last_modified_iso = datetime.fromtimestamp(
        last_modified_timestamp, 
        tz=timezone.utc
    ).isoformat(timespec='seconds')
    
    # Build metadata dictionary
    metadata_dict = {
        "coordinates": coordinates_data,
        "filename": filename,
        "file_directory": file_directory,
        "filetype": "application/pdf",
        "page_number": element.page_number,
        "last_modified": last_modified_iso
    }
    
    # Add optional metadata fields if available
    if "font_sizes" in element.metadata:
        metadata_dict["font_info"] = {
            "sizes": element.metadata["font_sizes"]
        }
    
    # Add detection probability if available
    if "classification_prob" in element.metadata:
        metadata_dict["detection_class_prob"] = element.metadata["classification_prob"]
    
    # Add parent ID if available
    if "parent_id" in element.metadata:
        metadata_dict["parent_id"] = element.metadata["parent_id"]
    
    # Add languages (defaulting to English for now)
    metadata_dict["languages"] = ["eng"]
    
    return {
        "element_id": element_id,
        "metadata": metadata_dict,
        "text": element.text,
        "type": element.element_type
    }

def save_elements_to_json(elements: List[Element], pdf_filepath: str, 
                         output_json_path: str) -> None:
    json_elements = [
        element_to_json_dict(element, pdf_filepath)
        for element in elements
    ]
    
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(json_elements, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving JSON file: {e}")
