from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, Set
from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTPage, LTTextBoxHorizontal, LTTextLineHorizontal, LTChar, LTAnno
import math
import re
from text_cleaners import TextCleaner
from bbox_utils import (
    rect_to_bbox, merge_bboxes, should_merge_bboxes, 
    calculate_iou, get_bbox_points, BBox
)
from font_analyzer import FontAnalyzer
from list_analyzer import ListAnalyzer

@dataclass
class CoordinateSystem:
    width: float
    height: float
    origin: str = "top-left"

@dataclass
class Element:
    text: str
    bbox: BBox
    page_number: int
    page_width: float
    page_height: float
    element_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def points(self) -> List[Tuple[float, float]]:
        return get_bbox_points(self.bbox)

class ElementClassifier:
    PAGE_NUMBER_PATTERNS = [
        r'^\d+$',
        r'^Page\s+\d+(\s+of\s+\d+)?$',
        r'^\d+\s*/\s*\d+$',
        r'^-\s*\d+\s*-$'
    ]
    
    @staticmethod
    def is_page_number(text: str, bbox: BBox, page_height: float, page_width: float,
                      font_info: Dict[str, Any], page_stats: Dict[str, float]) -> Tuple[bool, float]:
        x0, y0, x1, y1 = bbox
        text = text.strip()
        
        # Content pattern check
        pattern_match = any(re.match(pattern, text) for pattern in ElementClassifier.PAGE_NUMBER_PATTERNS)
        if not pattern_match:
            return False, 0.0
        
        # Position check
        margin = min(page_height, page_width) * 0.1
        is_in_margin = (y0 < margin) or (y0 > page_height - margin)
        
        # Font size check - page numbers typically smaller than body text
        is_small_font = (
            font_info["avg_font_size"] <= page_stats["body_font_size"]
            if "body_font_size" in page_stats
            else True
        )
        
        # Short text check
        is_short = len(text) < 20
        
        if not (is_in_margin and is_short and is_small_font):
            return False, 0.0
            
        confidence = 0.8 + (0.1 if is_small_font else 0) + (0.1 if is_in_margin else 0)
        return True, confidence

    @staticmethod
    def is_header_footer(text: str, bbox: BBox, page_height: float,
                        font_info: Dict[str, Any], page_stats: Dict[str, float]) -> Tuple[bool, float]:
        x0, y0, x1, y1 = bbox
        
        # Position check
        margin = page_height * 0.15
        is_top = y0 < margin
        is_bottom = y1 > page_height - margin
        
        if not (is_top or is_bottom):
            return False, 0.0
        
        # Content checks
        text_lines = text.split('\n')
        is_short = len(text_lines) <= 2
        
        if not is_short:
            return False, 0.0
        
        # Font checks
        is_distinct_font = False
        if "body_font_size" in page_stats:
            font_diff = abs(font_info["avg_font_size"] - page_stats["body_font_size"])
            is_distinct_font = font_diff > 1.0
        
        confidence = 0.7
        confidence += 0.1 if is_distinct_font else 0
        confidence += 0.1 if len(text_lines) == 1 else 0
        confidence += 0.1 if font_info["is_bold"] else 0
        
        return True, confidence

    @staticmethod
    def is_title(text: str, bbox: BBox, page_width: float,
                 font_info: Dict[str, Any], page_stats: Dict[str, float]) -> Tuple[bool, float]:
        x0, y0, x1, y1 = bbox
        text = text.strip()
        
        # Font size check
        if "body_font_size" not in page_stats:
            return False, 0.0
            
        size_ratio = font_info["avg_font_size"] / page_stats["body_font_size"]
        is_large_font = size_ratio > 1.2
        
        if not is_large_font:
            return False, 0.0
        
        # Position and alignment
        text_width = x1 - x0
        center_x = (x0 + x1) / 2
        page_center_x = page_width / 2
        is_centered = abs(center_x - page_center_x) < page_width * 0.2
        
        # Content checks
        is_short = len(text.split()) <= 20
        ends_with_punct = text[-1] not in '.!?:' if text else False
        
        if not (is_short and (is_centered or ends_with_punct)):
            return False, 0.0
        
        confidence = 0.7
        confidence += 0.1 if font_info["is_bold"] else 0
        confidence += 0.1 if is_centered else 0
        confidence += 0.1 if size_ratio > 1.5 else 0
        
        return True, confidence

    @staticmethod
    def is_list_item(text: str, bbox: BBox = None) -> Tuple[bool, float]:
        text = text.strip()
        pattern_info = ListAnalyzer.get_list_pattern_type(text)
        
        if not pattern_info:
            return False, 0.0
            
        confidence = 0.9  # List patterns are quite reliable
        return True, confidence

    @staticmethod
    def classify_element(text: str, bbox: BBox, page_number: int,
                        page_width: float, page_height: float,
                        font_info: Dict[str, Any],
                        page_stats: Dict[str, float]) -> Tuple[str, float]:
        # Try classifications in order of reliability
        is_page_num, conf = ElementClassifier.is_page_number(
            text, bbox, page_height, page_width, font_info, page_stats
        )
        if is_page_num:
            return "PageNumber", conf
            
        is_list, conf = ElementClassifier.is_list_item(text, bbox)
        if is_list:
            return "ListItem", conf
            
        is_title, conf = ElementClassifier.is_title(
            text, bbox, page_width, font_info, page_stats
        )
        if is_title:
            return "Title", conf
            
        is_header_footer, conf = ElementClassifier.is_header_footer(
            text, bbox, page_height, font_info, page_stats
        )
        if is_header_footer:
            element_type = "Header" if bbox[1] < page_height / 2 else "Footer"
            return element_type, conf
            
        if text.strip():
            return "NarrativeText", 0.7
            
        return "UncategorizedText", 0.5

class FastPDFParser:
    def __init__(self, line_margin: float = 0.5, char_margin: float = 2.0,
                 word_margin: float = 0.1, line_overlap: float = 0.5):
        self.la_params = LAParams(
            line_margin=line_margin,
            char_margin=char_margin,
            word_margin=word_margin,
            line_overlap=line_overlap
        )
        self.text_cleaner = TextCleaner()
        self.font_analyzer = FontAnalyzer()

    def extract_elements_fast(self, pdf_filepath: str, **kwargs) -> List[Element]:
        elements = []
        
        for page_num, page in enumerate(extract_pages(pdf_filepath, laparams=self.la_params), 1):
            if not isinstance(page, LTPage):
                continue
                
            page_width = page.width
            page_height = page.height
            
            # Collect text boxes for font analysis
            text_boxes = [obj for obj in page._objs if isinstance(obj, LTTextBoxHorizontal)]
            
            # Get page-level font statistics
            page_stats = FontAnalyzer.get_page_stats(text_boxes)
            
            page_elements = []
            
            for obj in text_boxes:
                text = obj.get_text()
                if not text.strip():
                    continue
                    
                # Clean text
                cleaned_text = self.text_cleaner.clean_text(text)
                if not cleaned_text:
                    continue
                
                # Get font information
                font_info = FontAnalyzer.analyze_text_box(obj)
                if not font_info:
                    continue
                
                # Transform coordinates
                bbox = rect_to_bbox(obj.bbox, page_height)
                
                # Classify element with confidence
                element_type, confidence = ElementClassifier.classify_element(
                    cleaned_text, bbox, page_num,
                    page_width, page_height,
                    font_info, page_stats
                )
                
                element = Element(
                    text=cleaned_text,
                    bbox=bbox,
                    page_number=page_num,
                    page_width=page_width,
                    page_height=page_height,
                    element_type=element_type,
                    metadata={
                        "font_info": font_info,
                        "classification_prob": confidence
                    }
                )
                page_elements.append(element)
            
            # Group list items on the page
            page_elements = ListAnalyzer.analyze_list_items(page_elements)
            elements.extend(page_elements)
        
        return elements

def extract_elements_fast(pdf_filepath: str, output_json: Optional[str] = None, **kwargs) -> List[Element]:
    parser = FastPDFParser(**kwargs)
    elements = parser.extract_elements_fast(pdf_filepath)
    
    if output_json:
        from json_serializer import save_elements_to_json
        save_elements_to_json(elements, pdf_filepath, output_json)
    
    return elements
