import sys
from pathlib import Path
from processor.pipeline import process_pdf
from utils.coordinate_system import CoordinateSystem
from utils.text_analysis import get_text_metrics, is_list_item, is_title
import json

def test_coordinate_system():
    print("\nTesting Coordinate System...")
    coord_sys = CoordinateSystem(width=612, height=792)  # Standard US Letter size
    
    # Test bbox normalization
    bbox = (100, 200, 300, 400)
    normalized = coord_sys.normalize_bbox(bbox)
    denormalized = coord_sys.denormalize_bbox(normalized)
    
    print(f"Original bbox: {bbox}")
    print(f"Normalized: {normalized}")
    print(f"Denormalized: {denormalized}")
    
    # Test relative position
    rel_pos = coord_sys.get_relative_position(bbox)
    print(f"Relative position metrics: {rel_pos}")

def test_text_analysis():
    print("\nTesting Text Analysis...")
    
    # Test title detection
    test_cases = [
        ("Introduction to Machine Learning", {"font_size": 14, "is_bold": True, "bbox": (100, 700, 300, 720)}, 
         {"median_font_size": 12, "page_height": 792}),
        ("1.1 Basic Concepts", {"font_size": 13, "is_bold": True, "bbox": (100, 650, 250, 670)},
         {"median_font_size": 12, "page_height": 792}),
        ("This is a regular paragraph with more text content.", {"font_size": 12, "is_bold": False, "bbox": (100, 600, 400, 620)},
         {"median_font_size": 12, "page_height": 792})
    ]
    
    for text, style, context in test_cases:
        title_score = is_title(text, style, context)
        print(f"\nText: {text}")
        print(f"Title score: {title_score}")
        metrics = get_text_metrics(text)
        print(f"Text metrics: {metrics}")

def test_list_detection():
    print("\nTesting List Detection...")
    
    test_cases = [
        ("• First bullet point", (50, 500, 400, 520), None),
        ("1. Numbered item", (50, 480, 400, 500), {"bbox": (50, 500, 400, 520), "is_list_item": True}),
        ("a) Alphabetical item", (50, 460, 400, 480), {"bbox": (50, 480, 400, 500), "is_list_item": True}),
        ("Regular text", (50, 440, 400, 460), {"bbox": (50, 460, 400, 480), "is_list_item": True})
    ]
    
    for text, bbox, prev_item in test_cases:
        is_list = is_list_item(text, bbox, prev_item)
        print(f"\nText: {text}")
        print(f"Is list item: {is_list}")

def test_pdf_processing():
    print("\nTesting PDF Processing...")
    
    # Process a sample PDF
    sample_pdf = "sample.pdf"
    if not Path(sample_pdf).exists():
        print(f"Error: {sample_pdf} not found")
        return
        
    try:
        elements = process_pdf(sample_pdf)
        
        # Save results for inspection
        output = []
        for element in elements:
            output.append({
                "text": element.text,
                "type": element.element_type,
                "bbox": element.metadata.coordinates.points,
                "page": element.metadata.page_number
            })
            
        with open("test_output.json", "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
            
        print(f"Processed {len(elements)} elements")
        print("Results saved to test_output.json")
        
    except Exception as e:
        print(f"Error processing PDF: {e}")

if __name__ == "__main__":
    test_coordinate_system()
    test_text_analysis()
    test_list_detection()
    test_pdf_processing()
