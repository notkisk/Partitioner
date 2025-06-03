def is_inside_table(bbox: tuple, table_areas: set, page_num: int) -> bool:
    """Check if a bounding box is inside any table area.
    
    Args:
        bbox: Tuple of (x0, y0, x1, y1) coordinates
        table_areas: Set of (page_num, (x0, y0, x1, y1)) table areas
        page_num: Current page number
        
    Returns:
        bool: True if the bbox is inside a table area, False otherwise
    """
    x0, y0, x1, y1 = bbox
    
    for table_page, table_bbox in table_areas:
        if table_page != page_num:
            continue
            
        # Check if bbox is completely inside table
        if (x0 >= table_bbox[0] and x1 <= table_bbox[2] and 
            y0 >= table_bbox[1] and y1 <= table_bbox[3]):
            return True
            
        # Check for significant overlap (more than 10% area)
        if _bbox_overlap(bbox, table_bbox) > 0.1:
            return True
            
    return False

def _bbox_overlap(bbox1: tuple, bbox2: tuple) -> float:
    """Calculate the overlap ratio between two bounding boxes.
    
    Args:
        bbox1: (x0, y0, x1, y1) coordinates of first bounding box
        bbox2: (x0, y0, x1, y1) coordinates of second bounding box
        
    Returns:
        float: Overlap ratio (0.0 to 1.0)
    """
    # Calculate intersection area
    x0 = max(bbox1[0], bbox2[0])
    y0 = max(bbox1[1], bbox2[1])
    x1 = min(bbox1[2], bbox2[2])
    y1 = min(bbox1[3], bbox2[3])
    
    if x0 >= x1 or y0 >= y1:
        return 0.0
        
    intersection = (x1 - x0) * (y1 - y0)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    if area1 == 0:
        return 0.0 # Cannot determine overlap if the first box has no area
        
    # Return ratio of intersection area to area of the first bounding box (bbox1)
    return intersection / area1
