from typing import List, Dict, Any
import numpy as np
from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTTextBoxHorizontal, LTChar

from utils.coordinate_system import CoordinateSystem
from .spatial_grouper import merge_text_blocks, merge_blocks
from .element_classifier import classify_element
from .list_handler import group_consecutive_list_items
from .data_models import Element, ElementType

def rect_to_bbox(rect: tuple, height: float) -> tuple:
    """Convert PDFMiner rectangle coordinates to bounding box coordinates.
    
    Args:
        rect: Rectangle coordinates (x0, y0, x1, y1) from PDFMiner
        height: Page height for coordinate transformation
        
    Returns:
        tuple: Bounding box coordinates (x0, y0, x1, y1)
    """
    x0, y0, x1, y1 = rect
    # PDFMiner uses bottom-left origin, we use top-left
    return (x0, height - y1, x1, height - y0)

def process_pdf(pdf_path: str, la_params: Dict[str, Any] = None) -> List[Element]:
    """Process a PDF file and extract structured elements.
    
    Args:
        pdf_path: Path to the PDF file
        la_params: Optional LAParams configuration
        
    Returns:
        List[Element]: List of classified elements
    """
    from utils.coordinate_system import CoordinateSystem
    
    # Set default layout analysis parameters if none provided
    if isinstance(la_params, LAParams):
        params = la_params
    else:
        # Create LAParams with default values optimized for our use case
        params = LAParams(
            line_overlap=0.5,   # Controls line overlap detection
            char_margin=2.0,    # Increased to better detect word boundaries
            line_margin=0.5,    # Increased to better merge lines
            word_margin=0.1,    # Kept small to avoid merging separate words
            boxes_flow=0.5,     # Default value for text direction
            detect_vertical=True,# Enable vertical text detection
            all_texts=True      # Extract all text, including those in figures
        )
    
    # Process pages
    all_elements = []
    page_stats = {}
    
    # First pass: collect statistics
    for page_num, page_layout in enumerate(extract_pages(pdf_path, laparams=params), 1):
        coord_system = CoordinateSystem(page_layout.width, page_layout.height)
        page_stats[page_num] = _collect_page_statistics(page_layout, coord_system)
    
    # Calculate document-wide statistics
    doc_stats = _calculate_document_statistics(page_stats)
    doc_stats = _calculate_document_statistics(page_stats)
    
    # Second pass: extract and classify elements
    for page_num, page_layout in enumerate(extract_pages(pdf_path, laparams=params), 1):
        # Get raw text blocks
        raw_blocks = _extract_text_blocks(page_layout, page_layout.height)
        
        # Merge blocks that belong together
        merged_blocks = merge_text_blocks(raw_blocks)
        
        # Classify each block
        page_elements = []
        for block in merged_blocks:
            element = classify_element(
                text=block['text'],
                bbox=block['bbox'],
                style_info=block['style_info'],
                page_info={
                    'width': page_layout.width,
                    'height': page_layout.height,
                    **page_stats[page_num]
                },
                page_number=page_num,
                context=doc_stats
            )
            page_elements.append(element)
        
        # Group list items
        page_elements = group_consecutive_list_items(page_elements)
        
        all_elements.extend(page_elements)
    
    return all_elements

def _extract_text_blocks(page_layout, height) -> List[Dict[str, Any]]:
    """Extract text blocks with style information from a page layout."""
    blocks = []
    current_block = None
    
    for element in page_layout:
        if isinstance(element, LTTextBoxHorizontal):
            # Get text and bounding box
            text = element.get_text().strip()
            if not text:
                continue
                
            # Skip single characters or punctuation marks
            if len(text) <= 1 and not text.isalnum():
                continue
                
            bbox = rect_to_bbox(element.bbox, height)
                
            # Extract style information from most common character style
            style_counts = {}
            for line in element:
                for char in line:
                    if isinstance(char, LTChar):
                        style_key = (char.fontname, char.size)
                        style_counts[style_key] = style_counts.get(style_key, 0) + 1
            
            # Use most common style
            if style_counts:
                most_common_style = max(style_counts.items(), key=lambda x: x[1])[0]
                style_info = {
                    'font_name': most_common_style[0],
                    'font_size': most_common_style[1],
                    'is_bold': 'Bold' in most_common_style[0] or 'bold' in most_common_style[0],
                    'is_italic': 'Italic' in most_common_style[0] or 'italic' in most_common_style[0],
                    'is_monospace': 'Mono' in most_common_style[0] or 'Courier' in most_common_style[0]
                }
            
            if not style_info:
                continue
                
            # Create new block
            new_block = {
                'text': text,
                'bbox': bbox,
                'style_info': style_info
            }
                
            # Try to merge with previous block
            if current_block and _can_merge_blocks(current_block, new_block):
                # Merge blocks
                current_block = merge_blocks([current_block, new_block])
            else:
                # Add current block to list if it exists
                if current_block:
                    blocks.append(current_block)
                # Start new block
                current_block = new_block
                
    # Add the last block
    if current_block:
        blocks.append(current_block)
                
    return blocks
    
    return blocks

def _collect_page_statistics(page_layout, coord_system: CoordinateSystem) -> Dict[str, Any]:
    """Collect statistics about text blocks on a page.
    
    Args:
        page_layout: PDFMiner page layout
        coord_system: Coordinate system for the page
        
    Returns:
        Dict containing page statistics
    """
    stats = {
        'font_sizes': [],
        'line_heights': [],
        'word_spacing': [],
        'fonts': {},  # Will store font frequencies
        'styles': {}  # Will store style frequencies
    }
    
    for obj in page_layout:
        if isinstance(obj, LTTextBoxHorizontal):
            # Get text and bounding box
            text = obj.get_text().strip()
            if not text:
                continue
                
            bbox = obj.bbox
            normalized_bbox = coord_system.normalize_bbox(bbox)
            
            # Collect font information
            for line in obj:
                if hasattr(line, 'get_text'):
                    # Get style information from first character
                    for char in line:
                        if isinstance(char, LTChar):
                            stats['font_sizes'].append(char.size)
                            
                            # Update font frequency
                            if char.fontname not in stats['fonts']:
                                stats['fonts'][char.fontname] = 0
                            stats['fonts'][char.fontname] += 1
                            
                            # Update style frequency
                            style_key = f"{char.fontname}_{char.size if char.size > 0 else 'regular'}"
                            if style_key not in stats['styles']:
                                stats['styles'][style_key] = 0
                            stats['styles'][style_key] += 1
                            break  # Only need first character's style
                        
                        # Calculate line metrics
                        line_height = line.height
                        stats['line_heights'].append(line_height)
                        
                        # Calculate word spacing
                        words = text.split()
                        if len(words) > 1:
                            avg_word_space = (line.width - sum(char.width for char in line if isinstance(char, LTChar))) / (len(words) - 1)
                            stats['word_spacing'].append(avg_word_space)
    
    # Calculate statistics
    stats['median_font_size'] = np.median(stats['font_sizes']) if stats['font_sizes'] else 12
    stats['median_line_height'] = np.median(stats['line_heights']) if stats['line_heights'] else 14
    stats['median_word_spacing'] = np.median(stats['word_spacing']) if stats['word_spacing'] else 4
    
    return stats
    """Collect statistics about text styles and layout from a page."""
    font_sizes = {}
    line_heights = []
    font_names = {}
    
    for element in page_layout:
        if isinstance(element, LTTextBoxHorizontal):
            # Get line heights
            line_heights.extend(line.height for line in element)
            
            # Get font information
            for line in element:
                for char in line:
                    if isinstance(char, LTChar):
                        font_size = round(char.size, 1)
                        font_sizes[font_size] = font_sizes.get(font_size, 0) + 1
                        font_names[char.fontname] = font_names.get(char.fontname, 0) + 1
    
    return {
        'font_sizes': font_sizes,
        'line_heights': line_heights,
        'font_names': font_names
    }

def _calculate_document_statistics(page_stats: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate document-wide statistics from per-page statistics.
    
    Args:
        page_stats: Dictionary of page statistics keyed by page number
        
    Returns:
        Dict containing document-wide statistics
    """
    """Calculate document-wide statistics from per-page statistics."""
    doc_stats = {
        'avg_font_size': 0,
        'median_font_size': 0,
        'fonts': {},
        'styles': {}
    }
    
    # Collect all values
    all_font_sizes = []
    for stats in page_stats.values():
        # Collect font sizes
        all_font_sizes.extend(stats['font_sizes'])
        
        # Merge font frequencies
        for font_name, count in stats['fonts'].items():
            doc_stats['fonts'][font_name] = doc_stats['fonts'].get(font_name, 0) + count
            
        # Merge style frequencies
        for style_key, count in stats['styles'].items():
            doc_stats['styles'][style_key] = doc_stats['styles'].get(style_key, 0) + count
    
    # Calculate averages and most common
    if all_font_sizes:
        doc_stats['median_font_size'] = float(np.median(all_font_sizes))
        doc_stats['avg_font_size'] = sum(all_font_sizes) / len(all_font_sizes)
        
    if doc_stats['fonts']:
        doc_stats['most_common_font'] = max(doc_stats['fonts'].items(), key=lambda x: x[1])[0]
    
    return doc_stats

def _can_merge_blocks(block1: Dict[str, Any], block2: Dict[str, Any]) -> bool:
    """Check if two blocks can be merged based on style and position."""
    # Check style consistency
    style1 = block1['style_info']
    style2 = block2['style_info']
    
    if style1['font_name'] != style2['font_name']:
        return False
    
    if abs(style1['font_size'] - style2['font_size']) > 0.1:
        return False
    
    # Get bounding boxes
    bbox1 = block1['bbox']
    bbox2 = block2['bbox']
    
    # Calculate gaps
    x_gap = bbox2[0] - bbox1[2]  # x0 of block2 - x1 of block1
    y_gap = abs(bbox1[1] - bbox2[1])  # Vertical difference
    
    # Maximum allowed gaps (based on font size)
    font_size = style1['font_size']
    max_x_gap = font_size * 0.8  # Wider gap allowance for better merging
    max_y_gap = font_size * 0.3  # Slightly larger vertical gap allowance
    
    # Check for same line merging
    if y_gap <= max_y_gap:
        # On same line, check horizontal gap
        if x_gap <= max_x_gap:
            # Check for hyphenation
            if block1['text'].rstrip().endswith('-'):
                return True
            
            # Check for sentence continuation
            first_char = block2['text'].lstrip()[0] if block2['text'].strip() else ''
            if first_char.islower():
                return True
            
            # Check if blocks are close enough horizontally
            if x_gap <= font_size * 0.5:
                return True
    
    # Check for line wrapping
    if y_gap <= font_size * 1.5:  # Allow slightly more than one line height
        # Check if second block is indented or aligned with first
        if abs(bbox2[0] - bbox1[0]) <= font_size * 0.5:
            return True
        
        # Check for sentence continuation across lines
        first_char = block2['text'].lstrip()[0] if block2['text'].strip() else ''
        if first_char.islower():
            return True
    
    return False
