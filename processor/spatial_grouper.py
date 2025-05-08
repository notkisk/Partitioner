from typing import List, Dict, Any, Tuple
from .list_handler import handle_wrapped_list_item_lines

def merge_text_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge text blocks that are likely part of the same logical unit.
    
    Args:
        blocks: List of text blocks with bbox and style information
        
    Returns:
        List[Dict[str, Any]]: Merged blocks
    """
    if not blocks:
        return []
    
    # Sort blocks by vertical position (top to bottom) and then horizontal position
    sorted_blocks = sorted(blocks, key=lambda b: (-b['bbox'][1], b['bbox'][0]))
    
    # Parameters for merging
    y_tolerance = 3  # Reduced tolerance for vertical alignment
    x_tolerance = 15  # Maximum horizontal gap between blocks
    font_size_ratio = 0.2  # Maximum font size difference ratio
    
    merged_blocks = []
    current_group = [sorted_blocks[0]]
    
    for block in sorted_blocks[1:]:
        prev_block = current_group[-1]
        
        # Get bounding boxes
        prev_bbox = prev_block['bbox']
        curr_bbox = block['bbox']
        
        # Calculate gaps
        y_gap = abs(prev_bbox[1] - curr_bbox[1])  # Vertical gap
        x_gap = curr_bbox[0] - prev_bbox[2]       # Horizontal gap
        
        # Get font sizes
        prev_font = prev_block['style_info'].get('font_size', 0)
        curr_font = block['style_info'].get('font_size', 0)
        font_diff_ratio = abs(prev_font - curr_font) / max(prev_font, curr_font) if prev_font and curr_font else 1
        
        # Check if blocks should be merged
        same_line = y_gap <= y_tolerance
        close_enough = x_gap <= x_tolerance
        similar_font = font_diff_ratio <= font_size_ratio
        same_style = (
            prev_block['style_info'].get('is_bold') == block['style_info'].get('is_bold') and
            prev_block['style_info'].get('is_italic') == block['style_info'].get('is_italic')
        )
        
        if same_line and close_enough and similar_font and same_style:
            current_group.append(block)
        else:
            # Merge current group and start new one
            if len(current_group) > 0:
                merged_block = merge_blocks(current_group)
                if merged_block:
                    merged_blocks.append(merged_block)
            current_group = [block]
    
    # Handle the last group
    if current_group:
        merged_block = merge_blocks(current_group)
        if merged_block:
            merged_blocks.append(merged_block)
    
    return blocks
            
    merged_blocks.append(current_block)
    
    return merged_blocks

def should_merge_blocks(block1: Dict[str, Any], block2: Dict[str, Any]) -> bool:
    # Get style information
    style1 = block1.get('style_info', {})
    style2 = block2.get('style_info', {})
    
    # Get bounding boxes
    bbox1 = block1.get('bbox', (0, 0, 0, 0))
    bbox2 = block2.get('bbox', (0, 0, 0, 0))
    
    # Calculate vertical and horizontal gaps
    y_gap = abs(bbox1[1] - bbox2[1])
    x_gap = bbox2[0] - bbox1[2]
    
    # Get font sizes
    font_size1 = style1.get('font_size', 0)
    font_size2 = style2.get('font_size', 0)
    
    # Calculate thresholds based on font size
    max_font = max(font_size1, font_size2)
    y_threshold = max_font * 0.3  # 30% of font size
    x_threshold = max_font * 0.6  # 60% of font size
    
    # Check if blocks should be merged
    same_line = y_gap <= y_threshold
    close_enough = x_gap <= x_threshold
    same_style = (
        style1.get('font_name') == style2.get('font_name') and
        abs(font_size1 - font_size2) <= 0.1 and
        style1.get('is_bold') == style2.get('is_bold') and
        style1.get('is_italic') == style2.get('is_italic')
    )
    
    return same_line and close_enough and same_style
    """
    Determine if two blocks should be merged based on spatial and style properties.
    
    Args:
        block1: First text block
        block2: Second text block
        
    Returns:
        bool: True if blocks should be merged
    """
    # Get bounding boxes
    bbox1 = block1['bbox']
    bbox2 = block2['bbox']
    
    # Get style information
    style1 = block1['style_info']
    style2 = block2['style_info']
    
    # Calculate horizontal gap
    x_gap = bbox2[0] - bbox1[2]  # x0 of block2 - x1 of block1
    
    # Maximum allowed gap between words (adjust based on font size)
    font_size = style1.get('font_size', 12)
    max_word_gap = font_size * 0.6  # Allow for reasonable word spacing
    
    # Check if blocks are too far apart horizontally
    if x_gap > max_word_gap:
        return False
    
    # Check style consistency
    if style1.get('font_name') != style2.get('font_name'):
        return False
    
    if abs(style1.get('font_size', 0) - style2.get('font_size', 0)) > 0.1:
        return False
    
    # Check for hyphenation
    if block1['text'].rstrip().endswith('-'):
        return True
    
    # Check for punctuation that suggests blocks should stay separate
    last_char = block1['text'].rstrip()[-1] if block1['text'].strip() else ''
    if last_char in '.!?':
        return False
    
    # Check if second block starts with lowercase (likely continuation)
    first_char = block2['text'].lstrip()[0] if block2['text'].strip() else ''
    if first_char.islower():
        return True
    
    # If blocks are very close, merge them
    return x_gap <= (font_size * 0.3)  # Even tighter threshold for normal merging

def merge_blocks(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple text blocks into one."""
    if not blocks:
        return None
    
    if len(blocks) == 1:
        return blocks[0]
        
    # Merge bounding boxes
    x0 = min(block['bbox'][0] for block in blocks)
    y0 = min(block['bbox'][1] for block in blocks)
    x1 = max(block['bbox'][2] for block in blocks)
    y1 = max(block['bbox'][3] for block in blocks)
    
    # Merge text (add space if needed)
    merged_text = ''
    for i, block in enumerate(blocks):
        if i > 0:
            prev_text = merged_text.rstrip()
            curr_text = block['text'].lstrip()
            needs_space = not (prev_text.endswith(' ') or curr_text.startswith(' '))
            merged_text = prev_text + (' ' if needs_space else '') + curr_text
        else:
            merged_text = block['text']
    
    # Use style info from largest block
    largest_block = max(blocks, key=lambda b: 
        (b['bbox'][2] - b['bbox'][0]) * (b['bbox'][3] - b['bbox'][1]))
    
    return {
        'text': merged_text,
        'bbox': (x0, y0, x1, y1),
        'style_info': largest_block['style_info']
    }

def calculate_minimum_containing_bbox(bboxes: List[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    """
    Calculate the minimum bounding box that contains all given boxes.
    
    Args:
        bboxes: List of bounding boxes (x0, y0, x1, y1)
        
    Returns:
        Tuple[float, float, float, float]: Minimum containing bounding box
    """
    if not bboxes:
        return (0, 0, 0, 0)
        
    x0 = min(box[0] for box in bboxes)
    y0 = min(box[1] for box in bboxes)
    x1 = max(box[2] for box in bboxes)
    y1 = max(box[3] for box in bboxes)
    
    return (x0, y0, x1, y1)
