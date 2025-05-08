from typing import List, Dict, Any, Optional, Tuple
import re
from dataclasses import dataclass
from text_cleaners import TextCleaner
from pdfminer.layout import LTTextBoxHorizontal, LTTextLineHorizontal, LTChar

@dataclass
class PageContext:
    page_number: int
    page_width: float
    page_height: float
    avg_font_size: float
    dominant_font: str
    header_zone: float = 0.1  # Top 10% of page
    footer_zone: float = 0.1  # Bottom 10% of page

class ElementClassifier:
    TITLE_PATTERNS = [
        r"^(?:chapter|section|appendix)\s+\d+",
        r"^(?:table of contents|index|glossary|references|bibliography)$",
        r"^\d+\.\d+\s+[A-Z]"
    ]
    
    LIST_MARKERS = [
        r"^\s*(?:\d+\.|\(\d+\)|\[?\d+\]|\w+\.|\(\w+\)|\[?\w+\])\s+",
        r"^\s*[•●○◆▪-]\s+"
    ]
    
    @staticmethod
    def classify_element(text: str, metrics: Dict[str, Any], page_context: PageContext,
                        elements: List[Dict] = None, idx: int = None) -> Tuple[str, float]:
        """Classifies a text element based on its content and metrics
        Returns: Tuple of (classification, confidence_score)"""
        if not text.strip():
            return "Whitespace", 1.0
            
        scores = {
            "Title": 0.0,
            "Header": 0.0,
            "Footer": 0.0,
            "PageNumber": 0.0,
            "ListItem": 0.0,
            "NarrativeText": 0.0
        }
            
        metrics["page_height"] = page_context.page_height
        relative_y = metrics["bbox"][1] / page_context.page_height
        
        # Check for headers/footers first
        if (relative_y > (1 - page_context.footer_zone) or 
            relative_y < page_context.header_zone):
            # Check for page numbers with context
            if elements is not None and idx is not None:
                is_page_num, page_conf = ElementClassifier._is_page_number(text, metrics, elements, idx)
                if is_page_num:
                    scores["PageNumber"] = page_conf
                    
            # Check for headers/footers
            is_header, header_conf = ElementClassifier._is_header_footer(text, metrics)
            if is_header:
                if relative_y < page_context.header_zone:
                    scores["Header"] = header_conf
                else:
                    scores["Footer"] = header_conf
        # Check for titles
        is_title, title_conf = ElementClassifier._is_title(text, metrics, page_context)
        if is_title:
            scores["Title"] = title_conf
            
        # Check for list items
        is_list, list_conf = ElementClassifier._is_list_item(text, metrics, elements, idx)
        if is_list:
            scores["ListItem"] = list_conf
            
        # Default to narrative text if no strong classifications
        if max(scores.values()) < 0.4:
            scores["NarrativeText"] = 0.6
            
        # Return the highest confidence classification
        best_class = max(scores.items(), key=lambda x: x[1])
        return best_class[0], best_class[1]
                
        # Check for titles
        if ElementClassifier._is_title(text, metrics, page_context):
            return "Title"
            
        # Check for list items
        if ElementClassifier._is_list_item(text, metrics):
            return "ListItem"
            
        # Default to narrative text
        return "NarrativeText"
    
    @staticmethod
    def _is_page_number(text: str, metrics: Dict[str, Any], elements: List[Dict], idx: int) -> Tuple[bool, float]:
        text = text.strip().lower()
        coords = metrics["bbox"]
        y_pos = coords[1]
        confidence = 0.0
        
        # Check if text is a simple number or roman numeral
        if len(text) <= 4:
            if text.isdigit() or re.match(r"^[ivxlcdm]+$", text):
                page_height = metrics["page_height"]
                if page_height > 0:
                    rel_pos = y_pos / page_height
                    if rel_pos < 0.1 or rel_pos > 0.9:
                        return True
        
        # Check for explicit page labels
        if re.search(r"page\s+\d+", text, re.IGNORECASE):
            return True
        
        # Check surrounding elements for context
        context_size = 2
        start_idx = max(0, idx - context_size)
        end_idx = min(len(elements), idx + context_size + 1)
        surrounding = elements[start_idx:end_idx]
        
        # If nearby elements contain header/footer markers, more likely to be a page number
        footer_markers = ["copyright", "all rights reserved", "confidential", "draft", "internal use"]
        has_footer_context = any(
            any(marker in str(el["text"]).lower() for marker in footer_markers)
            for el in surrounding
        )
        
        if has_footer_context and (len(text) <= 4 or "page" in text.lower()):
            return True
        
        return False
    
    @staticmethod
    def _is_header_footer(text: str, metrics: Dict[str, Any]) -> Tuple[bool, float]:
        """Detects headers and footers"""
        text = text.strip().lower()
        
        # Common header/footer markers
        markers = [
            "copyright", "confidential", "all rights reserved",
            "draft", "private", "internal use"
        ]
        
        if any(marker in text for marker in markers):
            return True
            
        # Short text with special font
        if (len(text) < 50 and 
            metrics["font_info"]["avg_font_size"] < metrics["font_info"].get("body_font_size", 12)):
            return True
            
        return False
    
    @staticmethod
    def _is_title(text: str, metrics: Dict[str, Any], page_context: PageContext) -> Tuple[bool, float]:
        """Enhanced title detection"""
        text = text.strip()
        
        # Pattern-based checks
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in ElementClassifier.TITLE_PATTERNS):
            return True
            
        # Style-based checks
        font_ratio = metrics["font_info"]["avg_font_size"] / page_context.avg_font_size
        if (font_ratio > 1.2 and 
            metrics["font_info"].get("is_bold", False) and 
            len(text.split('\n')) <= 2):
            return True
            
        # All caps with special styling
        if (text.isupper() and 
            len(text) <= 100 and 
            len(text.split('\n')) <= 2 and
            (font_ratio > 1.1 or metrics["font_info"].get("is_bold", False))):
            return True
            
        return False
    
    @staticmethod
    def _is_list_item(text: str, metrics: Dict[str, Any], elements: List[Dict] = None, idx: int = None) -> Tuple[bool, float]:
        """Enhanced list item detection with context awareness"""
        text = text.strip()
        
        # Basic pattern checks
        if any(re.match(pattern, text) for pattern in ElementClassifier.LIST_MARKERS):
            return True
            
        # Extract potential ordered bullets
        a, b, c = TextCleaner.extract_ordered_bullets(text)
        if a is not None:
            return True
            
        # Context-based checks if available
        if elements and idx is not None:
            # Look at previous and next elements
            prev_idx = idx - 1
            next_idx = idx + 1
            
            # Check if this element is part of a list sequence
            if prev_idx >= 0:
                prev_text = elements[prev_idx]["text"].strip()
                # Check if previous element was a list item
                if elements[prev_idx].get("element_type") == "ListItem":
                    # Check for similar indentation
                    prev_x0 = elements[prev_idx]["metrics"]["bbox"][0]
                    curr_x0 = metrics["bbox"][0]
                    if abs(prev_x0 - curr_x0) < 5:  # Allow small horizontal variation
                        return True
            
            # Check for list-like formatting
            if len(text.split("\n")) == 1:  # Single line items more likely to be list items
                # Check for consistent indentation with other potential list items
                x0 = metrics["bbox"][0]
                page_width = metrics.get("page_width", 612)  # Default letter width
                rel_indent = x0 / page_width
                
                # Typical list indentation is between 5-15% of page width
                if 0.05 <= rel_indent <= 0.15:
                    return True
        
        return False
    
    @staticmethod
    def group_list_items(elements: List[Dict[str, Any]], page_context: PageContext) -> List[Dict[str, Any]]:
        """Groups related list items together with advanced context awareness"""
        if not elements:
            return []
            
        grouped = []
        current_group = None
        current_indent = None
        avg_line_height = page_context.avg_font_size * 1.2  # Approximate line height
        
        def is_continuation(prev: Dict, curr: Dict) -> bool:
            """Check if current element is a continuation of previous"""
            prev_bbox = prev["metrics"]["bbox"]
            curr_bbox = curr["metrics"]["bbox"]
            
            # Check horizontal alignment
            indent_diff = abs(curr_bbox[0] - prev_bbox[0])
            if indent_diff > 5:  # Allow small horizontal variation
                return False
                
            # Check vertical spacing
            vertical_gap = prev_bbox[1] - curr_bbox[3]  # Distance between bottom of prev and top of curr
            if vertical_gap > avg_line_height * 1.5:  # Allow 1.5x line height gap
                return False
                
            # Check for similar styling
            prev_font = prev["metrics"]["font_info"]
            curr_font = curr["metrics"]["font_info"]
            if (prev_font.get("dominant_font") != curr_font.get("dominant_font") or
                abs(prev_font.get("avg_font_size", 0) - curr_font.get("avg_font_size", 0)) > 0.5):
                return False
                
            return True
        
        for idx, elem in enumerate(elements):
            if elem["element_type"] != "ListItem":
                if current_group:
                    # Check if this might be a continuation of the list item
                    prev_elem = current_group[-1]
                    if is_continuation(prev_elem, elem):
                        # Merge with previous list item
                        prev_elem["text"] += "\n" + elem["text"]
                        # Update bounding box
                        prev_bbox = list(prev_elem["metrics"]["bbox"])
                        curr_bbox = elem["metrics"]["bbox"]
                        prev_bbox[1] = max(prev_bbox[1], curr_bbox[1])  # max top
                        prev_bbox[3] = min(prev_bbox[3], curr_bbox[3])  # min bottom
                        prev_elem["metrics"]["bbox"] = tuple(prev_bbox)
                        continue
                    
                    grouped.extend(current_group)
                    current_group = None
                grouped.append(elem)
                continue
                
            if current_group is None:
                current_group = [elem]
                current_indent = elem["metrics"]["bbox"][0]
                continue
                
            # Check if this item belongs to the current group
            prev_elem = current_group[-1]
            if is_continuation(prev_elem, elem):
                current_group.append(elem)
            else:
                # Check if this might be a sub-list
                curr_indent = elem["metrics"]["bbox"][0]
                indent_diff = curr_indent - current_indent
                if 10 <= indent_diff <= 50:  # Typical sub-list indentation
                    # Mark as sub-list item
                    elem["metadata"] = elem.get("metadata", {})
                    elem["metadata"]["list_level"] = len(current_group)
                    current_group.append(elem)
                else:
                    grouped.extend(current_group)
                    current_group = [elem]
                    current_indent = curr_indent
                
        if current_group:
            grouped.extend(current_group)
            
        return grouped
