from typing import List, Dict, Any, Tuple, Optional
import re
from collections import Counter
from dataclasses import dataclass
from pdfminer.layout import LTTextBoxHorizontal, LTTextLineHorizontal, LTChar

@dataclass
class TextBlock:
    text: str
    bbox: Tuple[float, float, float, float]
    font_info: Dict[str, Any]
    line_count: int
    char_count: int
    avg_line_length: float
    indentation: float
    is_all_caps: bool

class TextAnalyzer:
    # Refined patterns for better text analysis
    SENTENCE_END = r'[.!?][\'")\]]?\s+'
    TITLE_END = r'[:.!?]?\s*$'
    HEADING_PATTERNS = [
        r'^(?:Chapter|Section|Part)\s+\d+',
        r'^\d+\.\d+\s+[A-Z]',
        r'^[IVX]+\.\s+[A-Z]'
    ]
    
    @staticmethod
    def analyze_text_block(text: str, bbox: Tuple[float, float, float, float],
                          font_info: Dict[str, Any]) -> TextBlock:
        lines = text.split('\n')
        line_lengths = [len(line.strip()) for line in lines]
        
        return TextBlock(
            text=text,
            bbox=bbox,
            font_info=font_info,
            line_count=len(lines),
            char_count=sum(line_lengths),
            avg_line_length=sum(line_lengths) / len(lines) if lines else 0,
            indentation=bbox[0],  # x0 coordinate
            is_all_caps=text.strip().isupper()
        )
    
    @staticmethod
    def is_likely_title(block: TextBlock, page_stats: Dict[str, float]) -> Tuple[bool, float]:
        if not block.text.strip():
            return False, 0.0
            
        # Content-based checks
        is_short = block.line_count <= 2 and block.avg_line_length < 100
        ends_properly = bool(re.search(TextAnalyzer.TITLE_END, block.text))
        is_heading = any(re.match(pattern, block.text) for pattern in TextAnalyzer.HEADING_PATTERNS)
        
        # Style-based checks
        has_large_font = False
        if "body_font_size" in page_stats:
            font_ratio = block.font_info["avg_font_size"] / page_stats["body_font_size"]
            has_large_font = font_ratio > 1.2
        
        # Calculate confidence
        confidence = 0.0
        if is_short and (has_large_font or block.is_all_caps or is_heading):
            confidence = 0.7
            confidence += 0.1 if has_large_font else 0
            confidence += 0.1 if block.font_info["is_bold"] else 0
            confidence += 0.1 if ends_properly else 0
            confidence += 0.1 if is_heading else 0
            confidence = min(confidence, 1.0)
            
        return confidence > 0.7, confidence
    
    @staticmethod
    def is_likely_header_footer(block: TextBlock, page_stats: Dict[str, float],
                              y_position: str) -> Tuple[bool, float]:
        if not block.text.strip():
            return False, 0.0
            
        # Content checks
        is_short = block.line_count <= 2
        has_reasonable_length = block.avg_line_length < 80
        
        if not (is_short and has_reasonable_length):
            return False, 0.0
        
        # Style checks
        has_distinct_font = False
        if "body_font_size" in page_stats:
            font_diff = abs(block.font_info["avg_font_size"] - page_stats["body_font_size"])
            has_distinct_font = font_diff > 0.5
        
        # Position-based confidence
        confidence = 0.7  # Base confidence for position
        confidence += 0.1 if has_distinct_font else 0
        confidence += 0.1 if block.font_info["is_bold"] else 0
        confidence += 0.1 if block.line_count == 1 else 0
        
        return confidence > 0.7, confidence
    
    @staticmethod
    def analyze_text_continuity(prev_block: Optional[TextBlock],
                              curr_block: TextBlock) -> bool:
        """Check if two text blocks are likely part of the same semantic unit."""
        if not prev_block:
            return False
            
        # Font consistency check
        same_font = (
            prev_block.font_info["dominant_font_name"] == 
            curr_block.font_info["dominant_font_name"]
        )
        similar_size = abs(
            prev_block.font_info["avg_font_size"] - 
            curr_block.font_info["avg_font_size"]
        ) < 0.1
        
        # Indentation check
        similar_indent = abs(prev_block.indentation - curr_block.indentation) < 5
        
        # Sentence continuity check
        prev_ends_sentence = bool(re.search(TextAnalyzer.SENTENCE_END, prev_block.text))
        
        return (
            same_font and similar_size and similar_indent and 
            not prev_ends_sentence
        )
