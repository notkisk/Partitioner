from typing import Dict, List, Tuple, Optional, Set
from collections import Counter
from pdfminer.layout import LTTextBoxHorizontal, LTTextLineHorizontal, LTChar
import statistics

class FontInfo:
    def __init__(self, size: float, name: str, is_bold: bool = False):
        self.size = size
        self.name = name
        self.is_bold = is_bold
        
    @property
    def key(self) -> Tuple[float, str, bool]:
        return (self.size, self.name, self.is_bold)

class FontAnalyzer:
    BOLD_INDICATORS = {'bold', 'bd', 'b', 'black', 'heavy', 'extrabold', 'semibold', 'demi'}
    ITALIC_INDICATORS = {'italic', 'it', 'i', 'oblique'}
    HEADER_FONTS = {'helvetica', 'arial', 'times'}
    
    @staticmethod
    def is_font_bold(font_name: str) -> bool:
        font_name_lower = font_name.lower()
        return any(indicator in font_name_lower for indicator in FontAnalyzer.BOLD_INDICATORS)
    
    @staticmethod
    @staticmethod
    def is_font_italic(font_name: str) -> bool:
        font_name_lower = font_name.lower()
        return any(indicator in font_name_lower for indicator in FontAnalyzer.ITALIC_INDICATORS)
    
    @staticmethod
    def is_header_font(font_name: str) -> bool:
        font_base = font_name.lower().split('-')[0]
        return any(header in font_base for header in FontAnalyzer.HEADER_FONTS)
    
    def analyze_text_box(obj: LTTextBoxHorizontal) -> Dict[str, any]:
        font_infos: List[FontInfo] = []
        char_count = 0
        line_heights = []
        prev_baseline = None
        
        for line in obj._objs:
            if isinstance(line, LTTextLineHorizontal):
                if prev_baseline is not None:
                    line_heights.append(abs(line.y0 - prev_baseline))
                prev_baseline = line.y0
                
                for char in line._objs:
                    if isinstance(char, LTChar):
                        font_infos.append(FontInfo(
                            size=char.size,
                            name=char.fontname,
                            is_bold=FontAnalyzer.is_font_bold(char.fontname)
                        ))
                        char_count += 1
        
        if not font_infos:
            return {}
            
        # Calculate dominant font size
        sizes = [f.size for f in font_infos]
        avg_size = statistics.mean(sizes)
        median_size = statistics.median(sizes)
        
        # Calculate dominant font
        fonts_counter = Counter(f.key for f in font_infos)
        dominant_font = max(fonts_counter.items(), key=lambda x: x[1])[0]
        
        # Calculate bold ratio
        bold_count = sum(1 for f in font_infos if f.is_bold)
        bold_ratio = bold_count / len(font_infos) if font_infos else 0
        
        # Calculate line spacing stats
        avg_line_height = statistics.mean(line_heights) if line_heights else 0
        line_height_std = statistics.stdev(line_heights) if len(line_heights) > 1 else 0
        
        # Calculate italic ratio
        italic_count = sum(1 for f in font_infos if FontAnalyzer.is_font_italic(f.name))
        italic_ratio = italic_count / len(font_infos) if font_infos else 0
        
        # Check if using header-style font
        uses_header_font = any(FontAnalyzer.is_header_font(f.name) for f in font_infos)
        
        return {
            "avg_font_size": avg_size,
            "median_font_size": median_size,
            "dominant_font_size": dominant_font[0],
            "dominant_font_name": dominant_font[1],
            "is_bold": bold_ratio > 0.5,
            "bold_ratio": bold_ratio,
            "is_italic": italic_ratio > 0.5,
            "italic_ratio": italic_ratio,
            "uses_header_font": uses_header_font,
            "avg_line_height": avg_line_height,
            "line_height_std": line_height_std,
            "char_count": char_count
        }
    
    @staticmethod
    def get_page_stats(text_boxes: List[LTTextBoxHorizontal]) -> Dict[str, float]:
        all_sizes = []
        for box in text_boxes:
            font_info = FontAnalyzer.analyze_text_box(box)
            if font_info:
                all_sizes.extend([font_info["dominant_font_size"]] * font_info["char_count"])
        
        if not all_sizes:
            return {}
            
        # Calculate the mode of font sizes (most common size is likely body text)
        size_counter = Counter(round(size, 1) for size in all_sizes)
        body_font_size = max(size_counter.items(), key=lambda x: x[1])[0]
        
        return {
            "body_font_size": body_font_size,
            "min_font_size": min(all_sizes),
            "max_font_size": max(all_sizes)
        }
