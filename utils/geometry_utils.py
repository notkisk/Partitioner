from typing import List, Tuple, Optional
import numpy as np

def validate_bbox(bbox: Tuple[float, float, float, float]) -> bool:
    """Validate that a bounding box has valid coordinates"""
    if len(bbox) != 4:
        return False
    x1, y1, x2, y2 = bbox
    return x1 <= x2 and y1 <= y2

def calculate_bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    """Calculate area of a bounding box"""
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def minimum_containing_coords(*regions: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Find the minimum bounding box that contains all input regions"""
    if not regions:
        raise ValueError("No regions provided")
    
    x1s, y1s, x2s, y2s = zip(*regions)
    return (min(x1s), min(y1s), max(x2s), max(y2s))

def boxes_intersection_area(box1: Tuple[float, float, float, float],
                          box2: Tuple[float, float, float, float]) -> float:
    """Calculate intersection area of two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x1 < x2 and y1 < y2:
        return (x2 - x1) * (y2 - y1)
    return 0.0

def boxes_iou(box1: Tuple[float, float, float, float],
             box2: Tuple[float, float, float, float]) -> float:
    """Calculate Intersection over Union (IoU) of two boxes"""
    if not (validate_bbox(box1) and validate_bbox(box2)):
        return 0.0
        
    intersection = boxes_intersection_area(box1, box2)
    if intersection == 0:
        return 0.0
        
    area1 = calculate_bbox_area(box1)
    area2 = calculate_bbox_area(box2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def is_box_subregion(box1: Tuple[float, float, float, float],
                     box2: Tuple[float, float, float, float],
                     threshold: float = 0.9) -> bool:
    """Check if box1 is almost completely contained within box2"""
    if not (validate_bbox(box1) and validate_bbox(box2)):
        return False
        
    intersection = boxes_intersection_area(box1, box2)
    area1 = calculate_bbox_area(box1)
    
    return area1 > 0 and (intersection / area1) >= threshold

def merge_overlapping_boxes(boxes: List[Tuple[float, float, float, float]],
                          iou_threshold: float = 0.7) -> List[Tuple[float, float, float, float]]:
    """Merge boxes that have IoU above threshold"""
    if not boxes:
        return []
        
    result = boxes.copy()
    i = 0
    while i < len(result):
        j = i + 1
        while j < len(result):
            if boxes_iou(result[i], result[j]) > iou_threshold:
                result[i] = minimum_containing_coords(result[i], result[j])
                result.pop(j)
            else:
                j += 1
        i += 1
    
    return result

def sort_boxes(boxes: List[Tuple[float, float, float, float]], 
               mode: str = "xy") -> List[Tuple[float, float, float, float]]:
    """Sort boxes by specified mode (xy: top-to-bottom then left-to-right)"""
    if mode == "xy":
        return sorted(boxes, key=lambda b: (-b[1], b[0]))  # -y for top-to-bottom
    return boxes
