from typing import List, Dict, Any, Optional, Tuple
import re
from dataclasses import dataclass
from pdfminer.layout import (
    LTItem, LTContainer, LTTextBox, LTTextBoxHorizontal,
    LTTextLineHorizontal, LTChar, LTText
)

@dataclass
class TextElement:
    text: str
    bbox: Tuple[float, float, float, float]
    font_info: Dict[str, Any]
    page_number: int
    element_type: str = "NarrativeText"
    metadata: Dict[str, Any] = None

class PDFTextExtractor:
    @staticmethod
    def extract_text_from_item(item: LTItem) -> str:
        """Recursively extracts text from any LTItem, handling containers properly"""
        if isinstance(item, LTText) and not isinstance(item, LTContainer):
            return item.get_text()
            
        if isinstance(item, LTContainer):
            texts = []
            for child in item:
                if isinstance(child, (LTChar, LTText)):
                    texts.append(PDFTextExtractor.extract_text_from_item(child))
            return "".join(texts)
            
        return ""

    @staticmethod
    def extract_font_info(item: LTItem) -> Dict[str, Any]:
        """Extracts font information from an item by analyzing its characters"""
        chars = []
        if isinstance(item, LTContainer):
            for child in item._objs:
                if isinstance(child, LTTextLineHorizontal):
                    chars.extend(c for c in child._objs if isinstance(c, LTChar))
                elif isinstance(child, LTChar):
                    chars.append(child)
                    
        if not chars:
            return {}
            
        sizes = [c.size for c in chars]
        fonts = [c.fontname for c in chars]
        
        return {
            "avg_font_size": sum(sizes) / len(sizes),
            "dominant_font": max(set(fonts), key=fonts.count),
            "is_bold": any("bold" in f.lower() for f in fonts),
            "char_count": len(chars)
        }

    @staticmethod
    def get_text_block_metrics(item: LTTextBoxHorizontal) -> Dict[str, Any]:
        """Analyzes a text block for various metrics useful in classification"""
        text = PDFTextExtractor.extract_text_from_item(item)
        lines = text.split('\n')
        font_info = PDFTextExtractor.extract_font_info(item)
        
        return {
            "text": text,
            "bbox": (item.x0, item.y0, item.x1, item.y1),
            "font_info": font_info,
            "line_count": len(lines),
            "avg_line_length": sum(len(line) for line in lines) / len(lines) if lines else 0,
            "indentation": item.x0,
            "is_all_caps": text.strip().isupper(),
            "vertical_spacing": [
                abs(item._objs[i+1].y0 - item._objs[i].y0)
                for i in range(len(item._objs)-1)
                if isinstance(item._objs[i], LTTextLineHorizontal)
                and isinstance(item._objs[i+1], LTTextLineHorizontal)
            ]
        }

    @staticmethod
    def analyze_text_block_spacing(items: List[LTTextBoxHorizontal]) -> Dict[str, float]:
        """Analyzes spacing between text blocks to determine document metrics"""
        all_char_widths = []
        all_line_heights = []
        
        for item in items:
            for line in item._objs:
                if isinstance(line, LTTextLineHorizontal):
                    for char in line._objs:
                        if isinstance(char, LTChar):
                            all_char_widths.append(char.width)
                    if len(line._objs) > 1:
                        # Approximate line height from character heights
                        chars = [c for c in line._objs if isinstance(c, LTChar)]
                        if chars:
                            line_height = max(c.height for c in chars)
                            all_line_heights.append(line_height)
        
        return {
            "avg_char_width": sum(all_char_widths) / len(all_char_widths) if all_char_widths else 1.0,
            "avg_line_height": sum(all_line_heights) / len(all_line_heights) if all_line_heights else 12.0,
            "std_line_height": (
                (sum((x - (sum(all_line_heights) / len(all_line_heights))) ** 2 
                    for x in all_line_heights) / len(all_line_heights)) ** 0.5
                if all_line_heights else 0.0
            )
        }
