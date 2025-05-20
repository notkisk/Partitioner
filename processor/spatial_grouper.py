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
    y_tolerance = 3
    x_tolerance = 60
    font_size_ratio = 0.2
    
    merged_blocks = []
    current_group = [sorted_blocks[0]]
    
    import re
    for block in sorted_blocks[1:]:
        prev_block = current_group[-1]
        prev_bbox = prev_block['bbox']
        curr_bbox = block['bbox']
        y_gap = abs(prev_bbox[1] - curr_bbox[1])
        x_gap = curr_bbox[0] - prev_bbox[2]
        prev_font = prev_block['style_info'].get('font_size', 0)
        curr_font = block['style_info'].get('font_size', 0)
        font_diff_ratio = abs(prev_font - curr_font) / max(prev_font, curr_font) if prev_font and curr_font else 1
        prev_text = prev_block.get('text', '').strip()
        curr_text = block.get('text', '').strip()
        section_number_pat = r"^\d+(?:\.\d+)*$"
        prev_is_section_number = bool(re.match(section_number_pat, prev_text))
        prev_center = (prev_bbox[1] + prev_bbox[3]) / 2
        curr_center = (curr_bbox[1] + curr_bbox[3]) / 2
        center_gap = abs(prev_center - curr_center)
        similar_font = font_diff_ratio <= font_size_ratio
        same_style = (
            prev_block['style_info'].get('is_bold') == block['style_info'].get('is_bold') and
            prev_block['style_info'].get('is_italic') == block['style_info'].get('is_italic')
        )
        section_merge = prev_is_section_number and center_gap <= 40 and similar_font and same_style
        same_line = y_gap <= y_tolerance
        close_enough = x_gap <= x_tolerance
        page_height = prev_block.get('page_height') or prev_block.get('style_info', {}).get('page_height')
        curr_center_x = (curr_bbox[0] + curr_bbox[2]) / 2
        prev_center_x = (prev_bbox[0] + prev_bbox[2]) / 2
        is_centered = False
        is_page_number = False
        if page_height:
            page_mid_x = (prev_bbox[2] + prev_bbox[0]) / 2
            page_width = prev_block.get('page_width') or prev_block.get('style_info', {}).get('page_width')
            if page_width:
                is_centered = abs(curr_center_x - page_width/2) < page_width * 0.15
            is_page_number = is_centered and curr_bbox[1] > page_height * 0.85 and len(curr_text.strip()) <= 5 and curr_text.strip().isdigit()
        large_y_gap = y_gap > 30
        if is_page_number or (is_centered and large_y_gap and curr_bbox[1] > prev_bbox[3]):
            if len(current_group) > 0:
                merged_block = merge_blocks(current_group)
                if merged_block:
                    merged_blocks.append(merged_block)
            current_group = [block]
        elif same_line and close_enough and similar_font and same_style:
            current_group.append(block)
        else:
            if len(current_group) > 0:
                merged_block = merge_blocks(current_group)
                if merged_block:
                    merged_blocks.append(merged_block)
            current_group = [block]

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
            if current_group:
                merged_block = merge_blocks(current_group)
                if merged_block:
                    merged_blocks.append(merged_block)
            current_group = [block]

    def check_coords_within_boundary(coords, boundary, horizontal_threshold=0.2, vertical_threshold=0.3):
        x0, y0, x1, y1 = coords
        bx0, by0, bx1, by1 = boundary
        width = bx1 - bx0
        height = by1 - by0
        x_within = (x0 > bx0 - horizontal_threshold * width) and (x1 < bx1 + horizontal_threshold * width)
        y_within = (y0 > by0 - vertical_threshold * height) and (y1 < by1 + vertical_threshold * height)
        return x_within and y_within

    grouped = []
    tmp = None
    tmp_bbox = None
    for block in merged_blocks:
        if block.get('element_type') == 'list_item':
            if tmp is None:
                tmp = block.copy()
                tmp_bbox = list(block['bbox'])
            else:
                coords = block['bbox']
                if check_coords_within_boundary(coords, tmp_bbox, horizontal_threshold=0.08, vertical_threshold=0.12):
                    tmp['text'] = tmp['text'] + ' ' + block['text']
                    tmp_bbox = (
                        min(tmp_bbox[0], coords[0]),
                        min(tmp_bbox[1], coords[1]),
                        max(tmp_bbox[2], coords[2]),
                        max(tmp_bbox[3], coords[3])
                    )
                    tmp['bbox'] = tmp_bbox
                else:
                    grouped.append(tmp)
                    tmp = block.copy()
                    tmp_bbox = list(block['bbox'])
        else:
            if tmp is not None:
                grouped.append(tmp)
                tmp = None
                tmp_bbox = None
            grouped.append(block)
    if tmp is not None:
        grouped.append(tmp)

    def iou(a, b):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        ix0 = max(ax0, bx0)
        iy0 = max(ay0, by0)
        ix1 = min(ax1, bx1)
        iy1 = min(ay1, by1)
        inter_area = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        a_area = (ax1 - ax0) * (ay1 - ay0)
        b_area = (bx1 - bx0) * (by1 - by0)
        union_area = a_area + b_area - inter_area
        return inter_area / (union_area + 1e-8), inter_area / (a_area + 1e-8), a_area, b_area

    filtered = []
    for i, a in enumerate(grouped):
        if a.get('element_type') != 'list_item':
            filtered.append(a)
            continue
        keep = True
        for j, b in enumerate(grouped):
            if i != j and b.get('element_type') == 'list_item':
                iou_val, overlap_a, area_a, area_b = iou(a['bbox'], b['bbox'])
                if overlap_a > 0.92 and area_a < area_b * 0.9:
                    keep = False
                    break
        if keep:
            filtered.append(a)
    seen = set()
    deduped = []
    for block in filtered:
        key = (block.get('element_type'), tuple(block.get('bbox', [])), block.get('text', ''))
        if key not in seen:
            seen.add(key)
            deduped.append(block)
    def area(b):
        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    def overlap_ratio(a, b):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        ix0 = max(ax0, bx0)
        iy0 = max(ay0, by0)
        ix1 = min(ax1, bx1)
        iy1 = min(ay1, by1)
        inter_area = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        area_a = area(a)
        area_b = area(b)
        return inter_area / (area_a + 1e-8), inter_area / (area_b + 1e-8)
    merged = [False] * len(deduped)
    results = []
    for i, a in enumerate(deduped):
        if merged[i]:
            continue
        a_bbox = a.get('bbox', [])
        a_text = a.get('text', '')
        a_type = a.get('element_type')
        out_bbox = list(a_bbox)
        out_text = a_text
        for j, b in enumerate(deduped):
            if i != j and not merged[j]:
                b_bbox = b.get('bbox', [])
                b_text = b.get('text', '')
                b_type = b.get('element_type')
                r1, r2 = overlap_ratio(a_bbox, b_bbox)
                if (r1 > 0.8 or r2 > 0.8):
                    out_bbox = [min(out_bbox[0], b_bbox[0]), min(out_bbox[1], b_bbox[1]), max(out_bbox[2], b_bbox[2]), max(out_bbox[3], b_bbox[3])]
                    out_text = out_text + ' ' + b_text if b_bbox[0] >= out_bbox[0] else b_text + ' ' + out_text
                    merged[j] = True
        merged[i] = True
        merged_block = a.copy()
        merged_block['bbox'] = tuple(out_bbox)
        merged_block['text'] = out_text
        results.append(merged_block)
    return results

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
