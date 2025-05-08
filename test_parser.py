from pdf_analyzer import PDFAnalyzer
import json
import os
from typing import List, Dict, Any

def analyze_pdf(pdf_path: str, output_json: str = None) -> List[Dict[str, Any]]:
    analyzer = PDFAnalyzer()
    elements = analyzer.analyze_document(pdf_path)
    
    # Get document statistics
    stats = analyzer.get_document_statistics()
    
    # Print statistics
    print("\nElement Type Statistics:")
    element_types = {}
    for element in elements:
        element_type = element["element_type"]
        element_types[element_type] = element_types.get(element_type, 0) + 1
    
    for element_type, count in element_types.items():
        print(f"{element_type}: {count}")
    
    # Print sample elements
    print("\nSample Elements by Type:")
    samples = {}
    for element in elements:
        element_type = element["element_type"]
        if element_type not in samples:
            samples[element_type] = element
            print(f"\nElement Type: {element_type}")
            print(f"Page: {element['page_number']}")
            print(f"Text: {element['text'][:50]}...")
            print(f"BBox: {element['bbox']}")
            
            if element['cross_page_refs']:
                print(f"Cross-page references: {element['cross_page_refs']}")
    
    # Save to JSON if requested
    if output_json:
        json_elements = elements
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(json_elements, f, indent=2, ensure_ascii=False)
        print(f"\nJSON output saved to: {output_json}")
    
    return elements

def main():
    pdf_path = "sample.pdf"
    output_json = "sample_output.json"

    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found")
        return

    elements = analyze_pdf(pdf_path, output_json)
    print(f"Extracted {len(elements)} elements")
if __name__ == "__main__":
    main()
