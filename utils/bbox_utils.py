import numpy as np
from typing import Tuple, List, Optional

BBox = Tuple[float, float, float, float]

def rect_to_bbox(bbox: BBox, page_height: float) -> BBox:
    x1, y1, x2, y2 = bbox
    # Convert from PDFMiner's bottom-left origin to top-left origin
    return (x1, page_height - y2, x2, page_height - y1)

def calculate_bbox_area(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return width * height if width > 0 and height > 0 else 0.0

def calculate_intersection_area(bbox1: BBox, bbox2: BBox) -> float:
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    x_intersection = max(x1_1, x1_2)
    y_intersection = max(y1_1, y1_2)
    x2_intersection = min(x2_1, x2_2)
    y2_intersection = min(y2_1, y2_2)
    
    if x_intersection < x2_intersection and y_intersection < y2_intersection:
        return calculate_bbox_area((x_intersection, y_intersection, x2_intersection, y2_intersection))
    return 0.0

def calculate_iou(bbox1: BBox, bbox2: BBox) -> float:
    intersection_area = calculate_intersection_area(bbox1, bbox2)
    if intersection_area == 0:
        return 0.0
    
    area1 = calculate_bbox_area(bbox1)
    area2 = calculate_bbox_area(bbox2)
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

def merge_bboxes(bboxes: List[BBox]) -> Optional[BBox]:
    if not bboxes:
        return None
        
    x1 = min(bbox[0] for bbox in bboxes)
    y1 = min(bbox[1] for bbox in bboxes)
    x2 = max(bbox[2] for bbox in bboxes)
    y2 = max(bbox[3] for bbox in bboxes)
    
    return (x1, y1, x2, y2)

def should_merge_bboxes(bbox1: BBox, bbox2: BBox, 
                       vertical_threshold: float = 0.5,
                       horizontal_overlap_threshold: float = 0.5) -> bool:
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate vertical gap
    vertical_gap = abs(min(y2_1, y2_2) - max(y1_1, y1_2))
    
    # Calculate horizontal overlap
    overlap_x1 = max(x1_1, x1_2)
    overlap_x2 = min(x2_1, x2_2)
    horizontal_overlap = max(0, overlap_x2 - overlap_x1)
    
    min_width = min(x2_1 - x1_1, x2_2 - x1_2)
    horizontal_overlap_ratio = horizontal_overlap / min_width if min_width > 0 else 0
    
    return vertical_gap <= vertical_threshold and horizontal_overlap_ratio >= horizontal_overlap_threshold

def get_bbox_points(bbox: BBox) -> List[Tuple[float, float]]:
    x1, y1, x2, y2 = bbox
    return [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]  # Clockwise from top-left
