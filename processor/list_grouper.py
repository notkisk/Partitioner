from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class ListGroup:
    items: List[Dict[str, Any]]
    bbox: Tuple[float, float, float, float]

def group_consecutive_list_items(elements: List[Dict[str, Any]], max_gap: float = 20.0) -> List[Dict[str, Any]]:
    """
    Group consecutive list items into ListGroup objects.
    
    Args:
        elements: List of elements to process
        max_gap: Maximum vertical gap between list items (in points) to consider them part of the same group
        
    Returns:
        List of elements with list items grouped
    """
    if not elements:
        return []
    
    result = []
    current_group = []
    
    for element in elements:
        if not isinstance(element, dict):
            element_dict = element.__dict__
        else:
            element_dict = element
            
        if 'element_type' in element_dict and element_dict['element_type'] == 'list_item':
            if not current_group:
                current_group.append(element_dict)
            else:
                # Check vertical gap with previous item
                last_item = current_group[-1]
                last_bbox = last_item.get('bbox', [0, 0, 0, 0])
                current_bbox = element_dict.get('bbox', [0, 0, 0, 0])
                
                if last_bbox and len(last_bbox) >= 4 and current_bbox and len(current_bbox) >= 4:
                    vertical_gap = last_bbox[1] - current_bbox[3]  # y0_prev - y1_current
                    if 0 < vertical_gap <= max_gap:
                        current_group.append(element_dict)
                        continue
                
                # If we get here, the gap is too large - finalize current group
                if current_group:
                    result.append(create_list_group(current_group))
                current_group = [element_dict]
        else:
            # If we encounter a non-list item, finalize current group
            if current_group:
                result.append(create_list_group(current_group))
                current_group = []
            result.append(element_dict)
    
    # Add any remaining items in the current group
    if current_group:
        result.append(create_list_group(current_group))
    
    return result

def create_list_group(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a list group from individual list items."""
    if not items:
        return {}
    
    # Calculate the bounding box that contains all items
    x0 = min(item.get('bbox', [float('inf')])[0] for item in items)
    y0 = min(item.get('bbox', [0, float('inf')])[1] for item in items)
    x1 = max(item.get('bbox', [0, 0, 0])[2] for item in items)
    y1 = max(item.get('bbox', [0, 0, 0, 0])[3] for item in items)
    
    # Combine text from all items
    combined_text = '\n'.join(item.get('text', '') for item in items)
    
    return {
        'text': combined_text,
        'element_type': 'list',
        'bbox': [x0, y0, x1, y1],
        'items': items,  # Keep reference to original items
        'metadata': {
            'is_group': True,
            'item_count': len(items)
        }
    }
