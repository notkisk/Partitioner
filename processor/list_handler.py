from typing import List, Dict, Any, Optional
import uuid
from .data_models import Element, ElementType
from .patterns import is_list_item_start

def identify_potential_list_items(text: str, style_info: Dict[str, Any], coords_info: Dict[str, float]) -> bool:
    """
    Identify if a text block is likely to be a list item based on multiple factors.
    
    Args:
        text: The text content to analyze
        style_info: Dictionary containing font and style information
        coords_info: Dictionary containing spatial information (x0, y0, etc.)
        
    Returns:
        bool: True if the block is likely a list item
    """
    # Check for list item pattern at start
    if not is_list_item_start(text):
        return False
        
    # Additional checks to confirm list item likelihood
    
    # 1. Indentation check - list items often have consistent indentation
    indent_ratio = coords_info.get('indent_ratio', 0)
    if indent_ratio > 0.15:  # More than 15% indent from left margin
        return False  # Too indented, might be code block
        
    # 2. Length check - list items typically aren't very long
    if len(text.split('\n')) > 5:  # More than 5 lines
        return False  # Too long for typical list item
        
    # 3. Style consistency check
    if style_info.get('is_monospace', False):
        return False  # Likely code block
        
    return True

def group_consecutive_list_items(elements: List[Element]) -> List[Element]:
    """
    Group consecutive list items that belong to the same logical list.
    
    Args:
        elements: List of elements from a page
        
    Returns:
        List[Element]: Updated elements with list_group_id assigned to related items
    """
    current_group_id = None
    prev_list_item = None
    
    for element in elements:
        if element.element_type != ElementType.LIST_ITEM:
            prev_list_item = None
            continue
            
        if prev_list_item is None:
            # Start new list group
            current_group_id = uuid.uuid4().hex
            element.metadata.list_group_id = current_group_id
        else:
            # Check if this item belongs to the same list as previous
            if _are_list_items_related(prev_list_item, element):
                element.metadata.list_group_id = current_group_id
            else:
                # Start new list group
                current_group_id = uuid.uuid4().hex
                element.metadata.list_group_id = current_group_id
                
        prev_list_item = element
        
    return elements

def _are_list_items_related(item1: Element, item2: Element) -> bool:
    """
    Determine if two list items belong to the same logical list.
    
    Args:
        item1: First list item
        item2: Second list item
        
    Returns:
        bool: True if items appear to belong to same list
    """
    # 1. Vertical proximity check
    y_diff = abs(item1.bbox[3] - item2.bbox[1])
    if y_diff > 20:  # More than 20 points gap
        return False
        
    # 2. Horizontal alignment check
    x_diff = abs(item1.bbox[0] - item2.bbox[0])
    if x_diff > 10:  # More than 10 points difference in indentation
        return False
        
    # 3. Style consistency check
    if item1.metadata.style_info.get('font_name') != item2.metadata.style_info.get('font_name'):
        return False
        
    if abs(item1.metadata.style_info.get('font_size', 0) - 
           item2.metadata.style_info.get('font_size', 0)) > 0.1:
        return False
        
    return True

def handle_wrapped_list_item_lines(block1: Dict[str, Any], block2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Determine if block2 is a continuation of block1 (which is a list item).
    
    Args:
        block1: First text block (potential list item)
        block2: Second text block (potential continuation)
        
    Returns:
        Optional[Dict]: Merged block if blocks should be combined, None otherwise
    """
    # Only consider merging if first block is a list item
    if not is_list_item_start(block1['text']):
        return None
        
    # Check if second block might be a continuation
    if is_list_item_start(block2['text']):
        return None  # Second block is start of new list item
        
    # Vertical proximity check
    y_diff = abs(block1['bbox'][3] - block2['bbox'][1])
    if y_diff > 5:  # Very close vertically
        return None
        
    # Indentation check - continuation should align with text after bullet
    # Find x position after bullet in block1
    first_line = block1['text'].split('\n')[0]
    bullet_match = is_list_item_start(first_line)
    if bullet_match:
        # Estimate position after bullet
        bullet_width = 20  # Approximate width for bullet and spacing
        text_start_x = block1['bbox'][0] + bullet_width
        
        # Check if block2 aligns with text (not bullet)
        if abs(block2['bbox'][0] - text_start_x) > 5:
            return None
            
    # Style consistency check
    if block1['style_info'].get('font_name') != block2['style_info'].get('font_name'):
        return None
        
    # Merge blocks
    merged = block1.copy()
    merged['text'] = block1['text'] + '\n' + block2['text']
    merged['bbox'] = (
        min(block1['bbox'][0], block2['bbox'][0]),
        min(block1['bbox'][1], block2['bbox'][1]),
        max(block1['bbox'][2], block2['bbox'][2]),
        max(block1['bbox'][3], block2['bbox'][3])
    )
    
    return merged
