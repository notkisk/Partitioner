from typing import Tuple, List

def calculate_iou(box1: Tuple[float, float, float, float],
                 box2: Tuple[float, float, float, float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First bounding box (x0, y0, x1, y1)
        box2: Second bounding box (x0, y0, x1, y1)
        
    Returns:
        float: IoU score between 0 and 1
    """
    # Calculate intersection coordinates
    x0 = max(box1[0], box2[0])
    y0 = max(box1[1], box2[1])
    x1 = min(box1[2], box2[2])
    y1 = min(box1[3], box2[3])
    
    # Check if boxes overlap
    if x1 < x0 or y1 < y0:
        return 0.0
        
    # Calculate areas
    intersection = (x1 - x0) * (y1 - y0)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def is_within_boundary(inner_box: Tuple[float, float, float, float],
                      outer_box: Tuple[float, float, float, float],
                      tolerance: float = 2.0) -> bool:
    """
    Check if one bounding box is within another, with some tolerance.
    
    Args:
        inner_box: Box that should be inside (x0, y0, x1, y1)
        outer_box: Box that should contain inner_box (x0, y0, x1, y1)
        tolerance: Allowed deviation in points
        
    Returns:
        bool: True if inner_box is within outer_box
    """
    return (
        inner_box[0] >= outer_box[0] - tolerance and
        inner_box[1] >= outer_box[1] - tolerance and
        inner_box[2] <= outer_box[2] + tolerance and
        inner_box[3] <= outer_box[3] + tolerance
    )

def calculate_minimum_containing_box(boxes: List[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    """
    Calculate the minimum box that contains all given boxes.
    
    Args:
        boxes: List of bounding boxes (x0, y0, x1, y1)
        
    Returns:
        Tuple[float, float, float, float]: Minimum containing box
    """
    if not boxes:
        return (0, 0, 0, 0)
        
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[2] for box in boxes)
    y1 = max(box[3] for box in boxes)
    
    return (x0, y0, x1, y1)

def get_relative_coordinates(box: Tuple[float, float, float, float],
                           page_width: float,
                           page_height: float) -> Tuple[float, float, float, float]:
    """
    Convert absolute coordinates to relative coordinates (0-1 range).
    
    Args:
        box: Bounding box in absolute coordinates (x0, y0, x1, y1)
        page_width: Width of the page
        page_height: Height of the page
        
    Returns:
        Tuple[float, float, float, float]: Box in relative coordinates
    """
    return (
        box[0] / page_width,
        box[1] / page_height,
        box[2] / page_width,
        box[3] / page_height
    )
