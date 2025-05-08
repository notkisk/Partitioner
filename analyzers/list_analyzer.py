from typing import List, Dict, Optional, Set, Tuple
import re
import uuid
from dataclasses import dataclass
from bbox_utils import BBox, should_merge_bboxes, get_bbox_points

@dataclass
class ListContext:
    list_id: str
    pattern_type: str  # bullet, number, letter, etc.
    indent_level: float
    last_number: Optional[int] = None

class ListAnalyzer:
    BULLET_PATTERN = r'^[\u2022\*\-\u2023\u2043\u204C\u204D\u2219\u25AA\u25CF\u25E6\u29BE\u29BF]'
    NUMBER_PATTERN = r'^(\d+)[\.\)]'
    LETTER_PATTERN = r'^([a-z])[\.\)]'
    ROMAN_PATTERN = r'^((?:i|ii|iii|iv|v|vi|vii|viii|ix|x)+)[\.\)]'
    
    @staticmethod
    def get_list_pattern_type(text: str) -> Optional[Tuple[str, Optional[int]]]:
        text = text.strip().lower()
        
        if re.match(ListAnalyzer.BULLET_PATTERN, text):
            return ("bullet", None)
            
        number_match = re.match(ListAnalyzer.NUMBER_PATTERN, text)
        if number_match:
            return ("number", int(number_match.group(1)))
            
        letter_match = re.match(ListAnalyzer.LETTER_PATTERN, text)
        if letter_match:
            return ("letter", ord(letter_match.group(1)) - ord('a') + 1)
            
        roman_match = re.match(ListAnalyzer.ROMAN_PATTERN, text)
        if roman_match:
            return ("roman", None)  # Roman numeral conversion omitted for simplicity
            
        return None
    
    @staticmethod
    def get_indent_level(bbox: BBox) -> float:
        return bbox[0]  # x0 coordinate
    
    @staticmethod
    def should_continue_list(current_context: ListContext, 
                           text: str, bbox: BBox,
                           vertical_gap: float,
                           max_gap: float = 20.0,
                           indent_tolerance: float = 5.0) -> bool:
        if vertical_gap > max_gap:
            return False
            
        pattern_info = ListAnalyzer.get_list_pattern_type(text)
        if not pattern_info:
            return False
            
        pattern_type, number = pattern_info
        
        # Check if indent level matches
        indent = ListAnalyzer.get_indent_level(bbox)
        if abs(indent - current_context.indent_level) > indent_tolerance:
            return False
            
        # If it's a numbered list, check sequence
        if pattern_type == "number" and current_context.pattern_type == "number":
            if current_context.last_number is not None:
                if number != current_context.last_number + 1:
                    return False
                    
        # Pattern type should match (all bullets, all numbers, etc.)
        return pattern_type == current_context.pattern_type
    
    @staticmethod
    def analyze_list_items(elements: List['Element']) -> List['Element']:
        result = []
        current_list: Optional[ListContext] = None
        last_y = None
        
        for element in elements:
            if element.element_type != "ListItem":
                if current_list is not None:
                    current_list = None
                result.append(element)
                continue
                
            vertical_gap = abs(element.bbox[1] - last_y) if last_y is not None else 0
            pattern_info = ListAnalyzer.get_list_pattern_type(element.text)
            
            if pattern_info:
                pattern_type, number = pattern_info
                indent = ListAnalyzer.get_indent_level(element.bbox)
                
                if current_list is None or not ListAnalyzer.should_continue_list(
                    current_list, element.text, element.bbox, vertical_gap
                ):
                    # Start new list
                    current_list = ListContext(
                        list_id=str(uuid.uuid4()),
                        pattern_type=pattern_type,
                        indent_level=indent,
                        last_number=number
                    )
                
                # Update element with list context
                element.metadata["list_id"] = current_list.list_id
                element.metadata["list_index"] = number if number is not None else 0
                current_list.last_number = number
            
            result.append(element)
            last_y = element.bbox[3]  # y1 coordinate
            
        return result
