from processor.pipeline import process_pdf
from processor.visualization import visualize_elements
import os

def process_pdf_with_list_handling(pdf_path, output_dir, group_lists=True, max_gap=20.0, max_indent=10.0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, 'output.pdf')
    
    elements = process_pdf(
        pdf_path=pdf_path,
        group_lists=group_lists,
        max_list_gap=max_gap,
        max_indent_diff=max_indent
    )
    
    visualize_elements(pdf_path, elements, output_path)
    return output_path

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python list_handling_demo.py <path_to_pdf>")
        sys.exit(1)
        
    pdf_path = sys.argv[1]
    
    print("Processing with list grouping (default)...")
    grouped_output = process_pdf_with_list_handling(
        pdf_path,
        output_dir="output/grouped",
        group_lists=True
    )
    print(f"Grouped output saved to: {grouped_output}")
    
    print("\nProcessing without list grouping...")
    ungrouped_output = process_pdf_with_list_handling(
        pdf_path,
        output_dir="output/ungrouped",
        group_lists=False
    )
    print(f"Ungrouped output saved to: {ungrouped_output}")
