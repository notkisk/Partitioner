from typing import List, Dict, Any, Tuple, Optional, Union, Set
import numpy as np
import re
import logging
logger = logging.getLogger(__name__)
import json
from pathlib import Path
from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTTextBoxHorizontal, LTChar, LTPage, LTFigure
from core.pdf_text_extractor import PDFTextExtractor
from utils.coordinate_system import CoordinateSystem
from .table_utils import is_inside_table
from .spatial_grouper import merge_text_blocks, merge_blocks
from .element_classifier import classify_element
from .list_handler import group_consecutive_list_items
from .list_grouper import group_consecutive_list_items as group_list_items
from .data_models import Element, ElementType, ElementMetadata, CoordinatesMetadata
from .document_structure import DocumentStructureAnalyzer, SplitType
from .table_extractor import process_pdf_tables

def _bbox_overlap(bbox1, bbox2):
    """Calculate the overlap ratio between two bounding boxes.
    
    Args:
        bbox1: (x0, y0, x1, y1) coordinates of first bounding box
        bbox2: (x0, y0, x1, y1) coordinates of second bounding box
        
    Returns:
        float: Overlap ratio (0.0 to 1.0)
    """
    x0 = max(bbox1[0], bbox2[0])
    y0 = max(bbox1[1], bbox2[1])
    x1 = min(bbox1[2], bbox2[2])
    y1 = min(bbox1[3], bbox2[3])
    
    if x0 >= x1 or y0 >= y1:
        return 0.0
        
    intersection = (x1 - x0) * (y1 - y0)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    # Return intersection over union
    return intersection / (area1 + area2 - intersection)

# Constants for table handling
TABLE_OVERLAP_THRESHOLD = 0.8  # 80% overlap to consider as same table

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

def process_pdf(
    pdf_path: str,
    image_output_dir: Optional[str] = None,
    la_params: Dict[str, Any] = None,
    group_lists: bool = True,
    max_list_gap: float = 20.0,
    max_indent_diff: float = 10.0,
    handle_splits: bool = True,
    extract_tables: bool = True,
    table_flavor: str = 'lattice',
    **kwargs  # Add this to handle any additional arguments
) -> Tuple[List[Element], Dict[str, Any]]:
    logger.debug(f"Starting PDF processing for: {pdf_path}")
    logger.debug(f"Extract tables: {extract_tables}, Table flavor: {table_flavor}")
    logger.debug(f"Handle splits: {handle_splits}, Group lists: {group_lists}")
    """Process a PDF file and extract structured elements.
    
    Args:
        pdf_path: Path to the PDF file
        image_output_dir: Directory to save extracted images
        la_params: Optional LAParams configuration
        group_lists: Whether to group list items together
        max_list_gap: Maximum vertical gap between list items in points
        max_indent_diff: Maximum horizontal indentation difference for grouping
        handle_splits: Whether to detect and handle split content across pages
        extract_tables: Whether to extract tables using Camelot
        table_flavor: Table extraction flavor ('lattice' or 'stream')
        
    Returns:
        Tuple containing:
        - List[Element]: List of classified elements
        - Dict: Document structure information including split markers and tables
    """
    from utils.coordinate_system import CoordinateSystem
    from utils.image_utils import extract_images_from_pdf
    
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
    
    # Initialize document structure
    all_elements = []
    # early image extraction for proper caption detection
    if image_output_dir:
        img_dicts = extract_images_from_pdf(pdf_path, image_output_dir)
        # incorporate extracted image dicts directly
        for img in img_dicts:
            all_elements.append(img)
    
    page_stats = {}
    page_data = []
    table_areas = set()
    
    # Process tables if enabled
    if extract_tables:
        logger.info("Extracting tables using Camelot...")
        try:
            tables_data = process_pdf_tables(pdf_path, flavor=table_flavor)
            logger.info(f"Extracted {len(tables_data)} tables from the document")
            
            TABLE_FILTERING_MARGIN = 8.0  # Margin for expanding table areas for text filtering

            for i, table_dict in enumerate(tables_data, 1):
                try:
                    original_bbox_list = table_dict.get('bbox', [0.0, 0.0, 0.0, 0.0])
                    page_num = table_dict.get('page', 1)
                    
                    actual_page_width = table_dict.get('page_width')
                    actual_page_height = table_dict.get('page_height')

                    if actual_page_width is None or actual_page_height is None:
                        logger.error(f"CRITICAL: Table {i} on page {page_num} is missing page dimensions from table_extractor. Bbox expansion and element creation will be compromised.")
                        # Fallback to avoid crashing, but this indicates an issue in table_extractor
                        # These defaults are unlikely to be correct for all PDFs.
                        actual_page_width = actual_page_width or 612.0 
                        actual_page_height = actual_page_height or 792.0

                    # Create expanded bbox for filtering, using accurate page dimensions for clamping
                    x0, y0, x1, y1 = original_bbox_list
                    expanded_bbox_for_filtering = (
                        max(0.0, x0 - TABLE_FILTERING_MARGIN),
                        max(0.0, y0 - TABLE_FILTERING_MARGIN),
                        min(actual_page_width, x1 + TABLE_FILTERING_MARGIN),
                        min(actual_page_height, y1 + TABLE_FILTERING_MARGIN)
                    )
                    table_areas.add((page_num, expanded_bbox_for_filtering))
                    
                    # Clean HTML content
                    html_content = table_dict.get('html', '').replace('\n', '').replace('\r', '').strip()
                    
                    # Create table element using the ORIGINAL bbox and accurate page dimensions
                    table_element = Element(
                        text=f"[Table {i}]",
                        element_type=ElementType.TABLE_BODY,
                        bbox=[float(c) for c in original_bbox_list], # Ensure original_bbox_list components are float
                        page_number=page_num,
                        metadata=ElementMetadata(
                            style_info={},
                            coordinates=CoordinatesMetadata(
                                x0=float(original_bbox_list[0]),
                                y0=float(original_bbox_list[1]),
                                x1=float(original_bbox_list[2]),
                                y1=float(original_bbox_list[3]),
                                page_width=float(actual_page_width),
                                page_height=float(actual_page_height)
                            ),
                            table_data={
                                'html': html_content,
                                'accuracy': table_dict.get('accuracy', 0),
                                'flavor': table_dict.get('flavor', table_flavor),
                                'order': i,
                                'caption': table_dict.get('caption', '')
                            }
                        )
                    )
                    all_elements.append(table_element)
                    logger.debug(f"Added table {i} on page {page_num}. Original bbox: {original_bbox_list}. Expanded for filtering: {expanded_bbox_for_filtering}. Page dims: {actual_page_width}x{actual_page_height}")
                    
                except Exception as e:
                    logger.error(f"Error processing table data for table index {i} (page {table_dict.get('page', 'unknown')}): {str(e)}", exc_info=True)
            
            logger.debug(f"Populated {len(table_areas)} expanded table areas for filtering.")

        except Exception as e:
            logger.error(f"Error during table extraction phase: {str(e)}", exc_info=True)
    
    # count table elements (Element objects only)
    num_tables = sum(1 for e in all_elements if hasattr(e, 'element_type') and e.element_type == ElementType.TABLE_BODY)
    logger.debug(f"Starting page processing for text and other elements after processing {num_tables} tables")
    
    # First pass: collect statistics and raw page data
    try:
        pages = list(extract_pages(pdf_path, laparams=params))
        logger.debug(f"Found {len(pages)} pages in the PDF")
        
        # The 'table_areas' set already contains (page_num, expanded_bbox_for_filtering) tuples.
        # We just need to filter these for the current page when calling _extract_text_blocks.

        for page_num, page_layout in enumerate(pages, 1):
            logger.debug(f"Processing page {page_num} (Width: {page_layout.width}, Height: {page_layout.height})")
            coord_system = CoordinateSystem(page_layout.width, page_layout.height)
            page_stats[page_num] = _collect_page_statistics(page_layout, coord_system)
            
            # Get the pre-expanded table areas for the current page
            # These were already expanded by TABLE_FILTERING_MARGIN and clamped in the table processing block.
            table_areas_for_extraction_on_this_page = set()
            for pn, bbox_tuple in table_areas:
                if pn == page_num:
                    table_areas_for_extraction_on_this_page.add((pn, bbox_tuple))
            
            if table_areas_for_extraction_on_this_page:
                logger.debug(f"Using {len(table_areas_for_extraction_on_this_page)} pre-expanded table areas for filtering on page {page_num}")
                for area in table_areas_for_extraction_on_this_page:
                    logger.debug(f"  - Page {area[0]}, Bbox: {area[1]}")
            
            # Extract raw blocks, excluding anything that overlaps with tables
            raw_blocks = _extract_text_blocks(
                page_layout=page_layout, 
                page_height=page_layout.height,
                table_areas=table_areas_for_extraction_on_this_page,
                table_overlap_threshold=0.1  # 10% overlap is enough to consider as part of table
            )
            
            # Log how many blocks were filtered out due to table overlap
            if table_areas_for_extraction_on_this_page:
                logger.info(f"Page {page_num}: Filtered text blocks using {len(table_areas_for_extraction_on_this_page)} table areas")
            
            # Extract just the bounding boxes for the current page that were used for filtering
            current_page_filter_bboxes = []
            for pn_filter, bbox_filter in table_areas_for_extraction_on_this_page:
                if pn_filter == page_num: # Ensure it's for the current page
                    current_page_filter_bboxes.append(bbox_filter)

            page_data.append({
                'page_num': page_num,
                'width': page_layout.width,
                'height': page_layout.height,
                'blocks': raw_blocks,
                'table_areas': current_page_filter_bboxes  # Store table areas for reference
            })
            
    except Exception as e:
        logger.error(f"Error during page processing: {str(e)}")
        raise
    
    # Calculate document-wide statistics
    doc_stats = _calculate_document_statistics(page_stats)
    
    # Add tables to document structure
    doc_stats['tables'] = [
        {
            'page': elem.page_number,
            'bbox': elem.bbox,
            'accuracy': elem.metadata.table_data.get('accuracy', 0),
            'flavor': elem.metadata.table_data.get('flavor', 'lattice')
        }
        for elem in all_elements if hasattr(elem, 'metadata') and hasattr(elem.metadata, 'table_data')
    ]
    
    # Initialize structure info with empty pages list
    structure_info = {'pages': [], 'split_markers': []}
    
    # Log table extraction results
    table_count = len([e for e in all_elements if hasattr(e, 'metadata') and hasattr(e.metadata, 'table_data')])
    if table_count > 0:
        logger.info(f"Successfully extracted {table_count} tables from the document")
    else:
        logger.info("No tables were extracted from the document")
    
    # Second pass: extract and classify elements
    logger.info(f"Starting to process {len(page_data)} pages")
    
    for page_info in page_data:
        page_num = page_info['page_num']
        logger.debug(f"Processing page {page_num}")
        
        try:
            page_layout = next(extract_pages(pdf_path, page_numbers=[page_num-1], laparams=params))
            logger.debug(f"Page {page_num} layout loaded successfully")
        except StopIteration:
            logger.warning(f"Page {page_num} not found in PDF, skipping")
            continue
            
        # Get raw text blocks
        raw_blocks = page_info.get('blocks', [])
        logger.debug(f"Found {len(raw_blocks)} raw blocks on page {page_num}")
        
        if not raw_blocks:
            logger.warning(f"No blocks found on page {page_num}")
            continue
        
        # Merge blocks that belong together
        try:
            merged_blocks = merge_text_blocks(raw_blocks) if raw_blocks else []
            logger.debug(f"Merged to {len(merged_blocks)} blocks on page {page_num}")
        except Exception as e:
            logger.error(f"Error merging blocks on page {page_num}: {str(e)}")
            merged_blocks = raw_blocks  # Fall back to raw blocks if merge fails
        
        # Create elements from blocks
        page_elements = []
        for block_idx, block in enumerate(merged_blocks):
            try:
                if not block or not isinstance(block, dict):
                    logger.debug(f"Skipping invalid block type on page {page_num}, block {block_idx}: {type(block)}")
                    continue
                    
                if 'text' not in block or 'bbox' not in block:
                    logger.debug(f"Skipping block {block_idx} on page {page_num}: missing required fields")
                    continue
                
                # Log block content for debugging
                block_text = str(block.get('text', ''))[:100] + ('...' if len(str(block.get('text', ''))) > 100 else '')
                logger.debug(f"Processing block {block_idx} on page {page_num}: {block_text}")
                
                # Ensure bbox has 4 elements
                bbox = block.get('bbox', [0, 0, 0, 0])
                if len(bbox) != 4:
                    logger.warning(f"Block {block_idx} on page {page_num} has invalid bbox: {bbox}")
                    bbox = [0, 0, 0, 0]  # Default to empty bbox
                
                # Try to determine element type from block metadata or content
                element_type = ElementType.UNKNOWN
                block_text = str(block.get('text', '')).strip()
                
                # Check for table data in metadata
                if block.get('metadata', {}).get('table_data'):
                    element_type = ElementType.TABLE_BODY
                # Check for common patterns in text
                elif re.match(r'^Table\s+\d+[:.]?\s*$', block_text, re.IGNORECASE):
                    element_type = ElementType.TABLE_CAPTION
                elif re.match(r'^Figure\s+\d+[:.]?\s*$', block_text, re.IGNORECASE):
                    element_type = ElementType.FIGURE_CAPTION
                elif re.match(r'^\s*\d+[.)]\s+', block_text) or re.match(r'^\s*[•\-*]\s+', block_text):
                    element_type = ElementType.LIST_ITEM
                # Check style info for headings
                elif block.get('style_info', {}).get('is_bold') and len(block_text.split()) < 10:
                    element_type = ElementType.HEADING
                
                element = Element(
                    text=block_text,
                    element_type=element_type,
                    bbox=tuple(float(x) for x in bbox),
                    page_number=int(page_num),
                    metadata=ElementMetadata(
                        style_info=block.get('style_info', {}),
                        coordinates=CoordinatesMetadata(
                            x0=float(bbox[0]) if len(bbox) > 0 else 0,
                            y0=float(bbox[1]) if len(bbox) > 1 else 0,
                            x1=float(bbox[2]) if len(bbox) > 2 else 0,
                            y1=float(bbox[3]) if len(bbox) > 3 else 0,
                            page_width=float(getattr(page_layout, 'width', 612)),
                            page_height=float(getattr(page_layout, 'height', 792))
                        )
                    )
                )
                page_elements.append(element)
                logger.debug(f"Created element {len(page_elements)-1} on page {page_num}")
                
            except Exception as e:
                logger.error(f"Error creating element from block {block_idx} on page {page_num}: {str(e)}", exc_info=True)
        
        # Get split markers for this page
        page_markers = []
        
        # Classify elements with context
        for i, element in enumerate(page_elements):
            try:
                if not hasattr(element, 'text') or not hasattr(element, 'bbox') or not hasattr(element, 'metadata'):
                    logger.debug(f"Skipping invalid element at index {i} on page {page_num}")
                    continue
                    
                # Get elements before and after for context
                elements_before = page_elements[max(0, i-3):i]
                elements_after = page_elements[i+1:min(i+4, len(page_elements))]
                
                # Skip classification for table elements
                if hasattr(element.metadata, 'table_data') and element.metadata.table_data:
                    element.element_type = ElementType.TABLE_BODY
                    continue
                
                try:
                    # Classify with context
                    classified = classify_element(
                        text=element.text,
                        bbox=element.bbox,
                        style_info=getattr(element.metadata, 'style_info', {}),
                        page_info={
                            'width': getattr(page_layout, 'width', 612),
                            'height': getattr(page_layout, 'height', 792),
                            **page_stats.get(page_num, {})
                        },
                        page_number=page_num,
                        context=doc_stats,
                        elements_before=elements_before,
                        elements_after=elements_after
                    )
                    
                    # Update the element with classification
                    if hasattr(classified, 'element_type') and classified.element_type != ElementType.UNKNOWN:
                        element.element_type = classified.element_type
                    
                    # Fallback classification if still UNKNOWN
                    if element.element_type == ElementType.UNKNOWN:
                        if any(keyword in element.text.lower() for keyword in ['table', 'figure', 'chart', 'diagram']):
                            element.element_type = ElementType.CAPTION
                        elif len(element.text.split()) < 5 and element.text.endswith(':'):
                            element.element_type = ElementType.HEADING
                        elif element.text.strip().isdigit() and len(element.text.strip()) < 5:
                            element.element_type = ElementType.PAGE_NUMBER
                        elif element.text.strip() in ['•', '-', '*', '•', '‣', '⁃']:
                            element.element_type = ElementType.LIST_ITEM
                    
                    # Update confidence
                    if hasattr(classified, 'metadata') and hasattr(classified.metadata, 'confidence'):
                        if not hasattr(element.metadata, 'confidence'):
                            element.metadata.confidence = classified.metadata.confidence
                        else:
                            element.metadata.confidence = max(
                                float(getattr(element.metadata, 'confidence', 0)),
                                float(getattr(classified.metadata, 'confidence', 0))
                            )
                    
                    # Ensure we have a valid element type
                    if element.element_type is None or element.element_type == {}:
                        element.element_type = ElementType.TEXT
                        
                except Exception as e:
                    logger.warning(f"Error classifying element on page {page_num}: {str(e)}")
                    element.element_type = ElementType.TEXT
                    if hasattr(element.metadata, 'confidence'):
                        element.metadata.confidence = 0.0
            except Exception as e:
                logger.error(f"Unexpected error processing element {i} on page {page_num}: {str(e)}")
        
        # Group list items
        page_elements = group_consecutive_list_items(
            elements=page_elements,
            group_lists=group_lists,
            max_gap=max_list_gap,
            max_indent_diff=max_indent_diff
        )
        
        if group_lists:
            page_elements = group_list_items(page_elements, max_gap=max_list_gap)
        
        all_elements.extend(page_elements)
    
    # Skip split elements processing since we're not using it
    logger.debug(f"PDF processing completed. Extracted {len(all_elements)} elements total")
    # Sort elements by page number and vertical position
    def get_sort_key(element):
        if hasattr(element, 'page_number') and hasattr(element, 'bbox') and element.bbox:
            return (element.page_number, element.bbox[1] if len(element.bbox) > 1 else 0)
        elif isinstance(element, dict):
            return (element.get('page_number', 0), element.get('bbox', [0, 0])[1] if 'bbox' in element and element['bbox'] and len(element['bbox']) > 1 else 0)
        return (0, 0)
    
    all_elements.sort(key=get_sort_key)
    
    return all_elements, structure_info

def _bbox_overlap(bbox1, bbox2, threshold=0.7):
    """Calculate overlap ratio between two bounding boxes."""
    if not bbox1 or not bbox2 or len(bbox1) != 4 or len(bbox2) != 4:
        return 0.0
        
    # Calculate intersection
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = bbox1_area + bbox2_area - intersection_area
    
    # Calculate IoU (Intersection over Union)
    iou = intersection_area / union_area if union_area > 0 else 0
    
    # Calculate overlap ratio relative to first bbox
    overlap_ratio = intersection_area / bbox1_area if bbox1_area > 0 else 0
    
    return overlap_ratio if overlap_ratio >= threshold else 0.0

def _process_split_elements(elements: List[Union[Element, dict]], split_markers: List[dict]) -> List[Union[Element, dict]]:
    """Process elements to handle split content.
    
    Args:
        elements: List of elements to process, can be Element objects or dictionaries
        split_markers: List of split markers with source and target information
        
    Returns:
        List of processed elements with split information added
    """
    if not split_markers or not elements:
        return elements
    
    # Group elements by page
    elements_by_page = {}
    for element in elements:
        if isinstance(element, dict):
            page = element.get('page_number')
            if page is None:
                continue
        else:
            page = element.page_number
            
        if page not in elements_by_page:
            elements_by_page[page] = []
        elements_by_page[page].append(element)
    
    # Process each split marker
    for marker in split_markers:
        if not isinstance(marker, dict):
            continue
            
        src_page = marker.get('source_page')
        tgt_page = marker.get('target_page')
        
        if not src_page or not tgt_page or src_page not in elements_by_page or tgt_page not in elements_by_page:
            continue
            
        # Find source and target elements
        def get_bbox(element):
            if isinstance(element, dict):
                return element.get('bbox', (0, 0, 0, 0))
            return element.bbox
            
        src_elements = [e for e in elements_by_page[src_page] 
                       if _bbox_overlap(get_bbox(e), marker.get('source_bbox', (0, 0, 0, 0))) > 0.7]
        tgt_elements = [e for e in elements_by_page[tgt_page] 
                       if _bbox_overlap(get_bbox(e), marker.get('target_bbox', (0, 0, 0, 0))) > 0.7]
        
        # If we found matching elements, mark them as split
        if src_elements and tgt_elements:
            for elem in src_elements + tgt_elements:
                if isinstance(elem, dict):
                    if 'metadata' not in elem:
                        elem['metadata'] = {}
                    if 'split_info' not in elem['metadata']:
                        elem['metadata']['split_info'] = []
                    elem['metadata']['split_info'].append({
                        'type': marker.get('split_type'),
                        'source_page': src_page,
                        'target_page': tgt_page,
                    })
                else:
                    if not hasattr(elem.metadata, 'metadata'):
                        elem.metadata.metadata = {}
                    if 'split_info' not in elem.metadata.metadata:
                        elem.metadata.metadata['split_info'] = []
                    elem.metadata.metadata['split_info'].append({
                        'type': marker.get('split_type'),
                        'source_page': src_page,
                        'target_page': tgt_page,
                    })
    
    return elements

def _is_inside_table(bbox: tuple, table_areas: Set[tuple], page_num: int, threshold: float = 0.1) -> bool:
    """Check if a bounding box is inside or significantly overlaps with any table area.
    
    Args:
        bbox: Tuple of (x0, y0, x1, y1) coordinates
        table_areas: Set of (page_num, (x0, y0, x1, y1)) table areas
        page_num: Current page number
        threshold: Minimum overlap ratio (0.0 to 1.0) to consider as inside table
        
    Returns:
        bool: True if the bbox is inside or significantly overlaps with a table area
    """
    if not table_areas:
        return False
    
    try:
        x0, y0, x1, y1 = map(float, bbox)
        bbox_area = (x1 - x0) * (y1 - y0)
        
        # Skip very small text elements that might be part of table borders
        if bbox_area < 50:  # Less than 50 square points
            return True
            
        for table_area in table_areas:
            try:
                # Handle different table area formats
                if isinstance(table_area, (list, tuple)) and len(table_area) == 2:
                    table_page, table_bbox = table_area
                    if not isinstance(table_bbox, (list, tuple)) or len(table_bbox) != 4:
                        continue
                else:
                    continue
                    
                if int(table_page) != int(page_num):
                    continue
                    
                tx0, ty0, tx1, ty1 = map(float, table_bbox)
                
                # Skip if no overlap at all
                if (x1 <= tx0 or x0 >= tx1 or y1 <= ty0 or y0 >= ty1):
                    continue
                    
                # Calculate intersection area
                inter_x0 = max(x0, tx0)
                inter_y0 = max(y0, ty0)
                inter_x1 = min(x1, tx1)
                inter_y1 = min(y1, ty1)
                
                inter_area = max(0, inter_x1 - inter_x0) * max(0, inter_y1 - inter_y0)
                overlap_ratio = inter_area / bbox_area if bbox_area > 0 else 0
                
                # Consider as inside table if significant overlap
                if overlap_ratio > threshold:
                    return True
                    
                # Also check if the text is completely inside the table
                if (x0 >= tx0 and x1 <= tx1 and y0 >= ty0 and y1 <= ty1):
                    return True
                    
            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing table area {table_area}: {e}")
                continue
                
    except (ValueError, TypeError) as e:
        logger.warning(f"Error processing bbox {bbox}: {e}")
        return False

def _extract_text_blocks(
    page_layout: Union[LTPage, LTFigure], 
    page_height: float,
    table_areas: Set[tuple] = None,
    min_text_size: float = 5.0,  # Minimum text size to consider (points)
    table_overlap_threshold: float = 0.1  # 10% overlap to consider as inside table
) -> List[Dict]:
    """Extract text blocks with style information from a page layout.
    
    Args:
        page_layout: PDFMiner page or figure layout
        page_height: Height of the page for coordinate conversion
        table_areas: Set of (page_num, (x0, y0, x1, y1)) tuples to exclude from extraction
        min_text_size: Minimum text size to consider (points)
        table_overlap_threshold: Minimum overlap ratio to consider text as inside table
        
    Returns:
        List of text blocks with metadata
    """
    blocks = []
    page_num = getattr(page_layout, 'pageid', 1)
    page_width = getattr(page_layout, 'width', 612)
    
    def process_text_box(text_box):
        try:
            # Convert bbox coordinates and ensure they're valid
            x0, y0, x1, y1 = rect_to_bbox(text_box.bbox, page_height)
            text = text_box.get_text().strip()
            
            if not text:
                return None
                
            # Validate coordinates
            if not all(isinstance(coord, (int, float)) for coord in (x0, y0, x1, y1)):
                return None
                
            # Calculate text dimensions
            text_height = abs(y1 - y0)
            text_width = abs(x1 - x0)
            text_area = text_width * text_height
            
            # Skip very small text elements (likely noise or table borders)
            if text_height < min_text_size or text_width < min_text_size or text_area < 50:
                return None
                
            # Skip text that's inside or significantly overlaps with tables
            if table_areas:
                for table_area in table_areas:
                    try:
                        if len(table_area) == 2:  # (page_num, bbox)
                            table_page, table_bbox = table_area
                            if int(table_page) != int(page_num):
                                continue
                                
                            if len(table_bbox) == 4:
                                tx0, ty0, tx1, ty1 = map(float, table_bbox)
                                
                                # Calculate intersection
                                ix0 = max(x0, tx0)
                                iy0 = max(y0, ty0)
                                ix1 = min(x1, tx1)
                                iy1 = min(y1, ty1)
                                
                                if ix0 < ix1 and iy0 < iy1:  # If there's an overlap
                                    # Calculate areas
                                    intersection_area = (ix1 - ix0) * (iy1 - iy0)
                                    overlap_ratio = intersection_area / text_area if text_area > 0 else 0
                                    
                                    # If significant overlap, skip this text box
                                    if overlap_ratio > table_overlap_threshold:
                                        logger.debug(f"Skipping text in table: '{text[:50]}...' (overlap: {overlap_ratio:.2f})")
                                        return None
                    except Exception as e:
                        logger.debug(f"Error checking table overlap: {e}")
                        continue
            
            # Extract style information from the first character
            char_styles = {}
            try:
                for line in text_box:
                    if hasattr(line, '_objs'):  # Check if it's a text line
                        for char in line._objs:
                            if hasattr(char, 'fontname'):
                                char_styles = {
                                    'font': char.fontname,
                                    'size': char.size,
                                    'is_bold': 'bold' in char.fontname.lower() if char.fontname else False,
                                    'is_italic': 'italic' in char.fontname.lower() if char.fontname else False,
                                    'median_font_size': char.size  # Add font size for consistency
                                }
                                break
                        if char_styles:
                            break
            except Exception as e:
                logger.debug(f"Could not extract style info: {e}")
            
            return {
                'text': text,
                'bbox': (x0, y0, x1, y1),
                'page_number': page_num,
                'page_width': page_width,
                'page_height': page_height,
                'style_info': char_styles if char_styles else {}
            }
            
        except Exception as e:
            logger.warning(f"Error processing text box: {e}", exc_info=True)
            return None

    # Process the page to handle columns
    try:
        text_boxes = PDFTextExtractor.process_page_columns(page_layout)
    except Exception as e:
        logger.warning(f"Error processing page columns: {e}")
        text_boxes = [b for b in page_layout if isinstance(b, LTTextBoxHorizontal)]

    for element in text_boxes:
        if not isinstance(element, LTTextBoxHorizontal):
            continue
            
        block = process_text_box(element)
        if block:
            blocks.append(block)
    
    return blocks


class PDFProcessor:
    def __init__(self, la_params=None):
        self.la_params = la_params or LAParams()
        
    def process_page(self, page_layout, page_num, table_areas=None):
        elements = []
        
        # Get page dimensions
        page_width = getattr(page_layout, 'width', 612)
        page_height = getattr(page_layout, 'height', 792)
        
        # Process all elements in the page layout
        for element in page_layout:
            if isinstance(element, LTTextBoxHorizontal):
                # Process text boxes
                block = self._process_text_box(element, page_layout, table_areas)
                if block:
                    # Create element metadata
                    element_metadata = ElementMetadata(
                        style_info=block.get('style_info', {}),
                        coordinates=CoordinatesMetadata(
                            x0=block['bbox'][0],
                            y0=block['bbox'][1],
                            x1=block['bbox'][2],
                            y1=block['bbox'][3],
                            page_width=block.get('page_width', page_width),
                            page_height=block.get('page_height', page_height)
                        )
                    )
                    
                    # Classify element type
                    element_type = classify_element(
                        block['text'], 
                        block.get('style_info', {}), 
                        block['bbox'], 
                        block.get('page_width', page_width), 
                        block.get('page_height', page_height)
                    )
                    
                    # Create element
                    element = Element(
                        text=block['text'],
                        element_type=element_type,
                        bbox=block['bbox'],
                        page_number=page_num,
                        metadata=element_metadata
                    )
                    
                    elements.append(element)
                    
            elif isinstance(element, LTFigure):
                # Process figures and other layout elements
                bbox = rect_to_bbox(element.bbox, page_height)
                element_metadata = ElementMetadata(
                    coordinates=CoordinatesMetadata(
                        x0=bbox[0],
                        y0=bbox[1],
                        x1=bbox[2],
                        y1=bbox[3],
                        page_width=page_width,
                        page_height=page_height
                    )
                )
                
                element = Element(
                    text=f"[Figure {len(elements) + 1}]",
                    element_type=ElementType.FIGURE,
                    bbox=bbox,
                    page_number=page_num,
                    metadata=element_metadata
                )
                
                elements.append(element)
        
        return elements
    
    def _process_text_box(self, text_box, page_layout, table_areas=None):
        x0, y0, x1, y1 = rect_to_bbox(text_box.bbox, page_layout.height)
        text = text_box.get_text().strip()
        
        if not text:
            return None
            
        # Skip if this area is covered by a table
        if table_areas:
            page_num = getattr(page_layout, 'pageid', 1)
            for table_page, table_bbox in table_areas:
                if page_num == table_page and _bbox_overlap((x0, y0, x1, y1), table_bbox) > 0.3:
                    return None

        # Get style information from the first character
        char_styles = {}
        for line in text_box:
            for char in line:
                if hasattr(char, 'fontname'):
                    char_styles = {
                        'font': char.fontname,
                        'size': char.size,
                        'is_bold': 'bold' in char.fontname.lower() if char.fontname else False,
                        'is_italic': 'italic' in char.fontname.lower() if char.fontname else False
                    }
                    break
            if char_styles:
                break

        style_info = {
            'font_name': char_styles.get('font'),
            'font_size': char_styles.get('size', 0),
            'is_bold': char_styles.get('is_bold', False),
            'is_italic': char_styles.get('is_italic', False),
            'is_monospace': False,
            'char_count': len(text),
            'avg_char_width': (x1 - x0) / max(1, len(text)) if text else 0,
            'line_count': len(text.split('\n'))
        }
        
        return {
            'text': text,
            'bbox': (x0, y0, x1, y1),
            'style_info': style_info,
            'page_number': getattr(page_layout, 'pageid', 1),
            'page_width': getattr(page_layout, 'width', 612),
            'page_height': getattr(page_layout, 'height', 792)
        }

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

def _calculate_document_statistics(page_stats: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate document-wide statistics from page statistics."""
    if not page_stats:
        return {}
        
    # Initialize statistics
    stats = {
        'total_pages': len(page_stats),
        'avg_line_height': 0,
        'avg_word_spacing': 0,
        'avg_char_width': 0,
        'common_font_sizes': [],
        'common_font_families': []
    }
    
    if not page_stats:
        return stats
        
    # Calculate averages
    line_heights = []
    word_spacings = []
    char_widths = []
    font_sizes = {}
    font_families = {}
    
    for page in page_stats.values():
        # Handle line heights
        if isinstance(page, dict) and 'line_heights' in page and isinstance(page['line_heights'], (list, tuple)):
            line_heights.extend(h for h in page['line_heights'] if h > 0)
            
        # Handle word spacings
        if isinstance(page, dict) and 'word_spacing' in page and isinstance(page['word_spacing'], (list, tuple)):
            word_spacings.extend(ws for ws in page['word_spacing'] if ws > 0)
            
        # Handle character widths
        if isinstance(page, dict) and 'char_widths' in page and isinstance(page['char_widths'], (list, tuple)):
            char_widths.extend(cw for cw in page['char_widths'] if cw > 0)
            
        # Handle font sizes
        if isinstance(page, dict) and 'font_sizes' in page:
            if isinstance(page['font_sizes'], dict):
                for size, count in page['font_sizes'].items():
                    font_sizes[size] = font_sizes.get(size, 0) + count
            elif isinstance(page['font_sizes'], (list, tuple)):
                for size in page['font_sizes']:
                    if size > 0:
                        font_sizes[round(size, 1)] = font_sizes.get(round(size, 1), 0) + 1
                        
        # Handle font families
        if isinstance(page, dict) and 'font_families' in page:
            if isinstance(page['font_families'], dict):
                for family, count in page['font_families'].items():
                    font_families[family] = font_families.get(family, 0) + count
    
    # Calculate averages
    if line_heights:
        stats['avg_line_height'] = float(np.median(line_heights))
    if word_spacings:
        stats['avg_word_spacing'] = float(np.median(word_spacings))
    if char_widths:
        stats['avg_char_width'] = float(np.median(char_widths))
    
    # Sort and get most common font sizes and families
    if font_sizes:
        common_sizes = sorted(font_sizes.items(), key=lambda x: x[1], reverse=True)
        stats['common_font_sizes'] = [size for size, _ in common_sizes[:5]]
        
    if font_families:
        common_families = sorted(font_families.items(), key=lambda x: x[1], reverse=True)
        stats['common_font_families'] = [family for family, _ in common_families[:5]]
    
    return stats

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
