from typing import List, Dict, Any, Optional, Tuple, Union
import re
import numpy as np
from dataclasses import dataclass
from pdfminer.layout import (
    LTItem, LTContainer, LTTextBox, LTTextBoxHorizontal,
    LTTextLineHorizontal, LTChar, LTText, LTPage
)
from collections import defaultdict

@dataclass
class TextElement:
    text: str
    bbox: Tuple[float, float, float, float]
    font_info: Dict[str, Any]
    page_number: int
    element_type: str = "NarrativeText"
    metadata: Dict[str, Any] = None

class PDFTextExtractor:
    @classmethod
    def detect_columns(cls, page: LTPage, min_gap_ratio: float = 0.15) -> List[Tuple[float, float]]:
        """Detect columns in a page based on horizontal gaps between text blocks.
        
        Args:
            page: The page to analyze
            min_gap_ratio: Minimum gap between columns as a ratio of page width
            
        Returns:
            List of (x0, x1) coordinates for each column
        """
        # Get all text blocks and filter out very small ones (likely noise)
        text_boxes = [
            b for b in page 
            if (isinstance(b, LTTextBoxHorizontal) and 
                len(b.get_text().strip()) > 1 and  # At least 2 characters
                (b.x1 - b.x0) > 10 and  # Minimum width
                (b.y1 - b.y0) > 5)  # Minimum height
        ]
        
        if not text_boxes:
            return [(0, page.width)]
            
        # Sort text boxes by x-coordinate
        text_boxes.sort(key=lambda b: b.x0)
        
        # Calculate page width and minimum gap size
        page_width = page.width
        min_gap = page_width * min_gap_ratio
        
        # Find gaps between text blocks
        gaps = []
        for i in range(1, len(text_boxes)):
            prev = text_boxes[i-1]
            curr = text_boxes[i]
            
            # Only consider gaps between blocks that are vertically aligned
            if (curr.y1 > prev.y0 and curr.y0 < prev.y1):
                gap = curr.x0 - prev.x1
                if gap > min_gap:
                    # Calculate the vertical overlap percentage
                    overlap = min(curr.y1, prev.y1) - max(curr.y0, prev.y0)
                    overlap_pct = overlap / max(curr.y1 - curr.y0, prev.y1 - prev.y0)
                    
                    if overlap_pct > 0.3:  # At least 30% vertical overlap
                        gaps.append((prev.x1, curr.x0, gap, overlap_pct))
        
        # If no significant gaps found, treat as single column
        if not gaps:
            return [(0, page_width)]
        
        # Sort gaps by size (largest first)
        gaps.sort(key=lambda x: -x[2])
        
        # Take the largest gaps as column separators
        max_columns = 3  # Maximum number of columns to detect
        column_gaps = [g[1] for g in gaps[:max_columns-1]]
        column_gaps.sort()
        
        # Define column boundaries
        columns = []
        prev_boundary = 0
        
        for gap in column_gaps:
            columns.append((prev_boundary, gap))
            prev_boundary = gap
        
        # Add the last column
        columns.append((prev_boundary, page_width))
        
        # Merge columns that are too narrow (less than 15% of page width)
        min_col_width = page_width * 0.15
        merged_columns = []
        i = 0
        
        while i < len(columns):
            current = columns[i]
            
            # If this is the last column or the next column is wide enough
            if i == len(columns) - 1 or (columns[i+1][1] - columns[i+1][0]) > min_col_width:
                merged_columns.append(current)
                i += 1
            else:
                # Merge with next column
                merged_columns.append((current[0], columns[i+1][1]))
                i += 2  # Skip the next column as it's been merged
        
        return merged_columns
    
    @classmethod
    def assign_columns(cls, text_boxes: List[LTTextBoxHorizontal], columns: List[Tuple[float, float]]) -> Dict[int, List[LTTextBoxHorizontal]]:
        """Assign text boxes to columns based on their x-coordinates."""
        column_boxes = defaultdict(list)
        
        for box in text_boxes:
            # Find which column this box belongs to
            box_center = (box.x0 + box.x1) / 2
            for i, (col_start, col_end) in enumerate(columns):
                if col_start <= box_center <= col_end:
                    column_boxes[i].append(box)
                    break
        
        return column_boxes
    
    @staticmethod
    def sort_boxes_column_major(column_boxes: Dict[int, List[LTTextBoxHorizontal]]) -> List[LTTextBoxHorizontal]:
        """Sort boxes in column-major order (top to bottom within each column, then left to right)."""
        sorted_boxes = []
        
        # Sort columns by their x-coordinate
        columns = sorted(column_boxes.items(), key=lambda x: min(box.x0 for box in x[1]) if x[1] else 0)
        
        for _, boxes in columns:
            # Sort boxes within column by y-coordinate (top to bottom)
            boxes_sorted = sorted(boxes, key=lambda b: -b.y1)  # y1 is top of box
            sorted_boxes.extend(boxes_sorted)
            
        return sorted_boxes
    
    @classmethod
    def process_page_columns(cls, page: LTPage) -> List[LTTextBoxHorizontal]:
        """Process a page with multiple columns, returning text boxes in reading order.
        
        Args:
            page: The PDF page to process
            
        Returns:
            List of text boxes in reading order (top-to-bottom, left-to-right)
        """
        # Get all text boxes, filtering out small fragments and empty ones
        text_boxes = [
            b for b in page 
            if (isinstance(b, LTTextBoxHorizontal) and 
                len(b.get_text().strip()) > 0 and
                (b.x1 - b.x0) > 5 and  # Minimum width
                (b.y1 - b.y0) > 2)     # Minimum height
        ]
        
        if not text_boxes:
            return []
            
        # Detect columns based on text box positions
        columns = cls.detect_columns(page)
        
        # If only one column, sort by vertical position (top to bottom)
        if len(columns) <= 1:
            return sorted(text_boxes, key=lambda b: (-b.y1, b.x0))
        
        # Assign text boxes to columns
        column_boxes = {i: [] for i in range(len(columns))}
        
        for box in text_boxes:
            box_center = (box.x0 + box.x1) / 2
            
            # Find which column this box belongs to
            for i, (col_start, col_end) in enumerate(columns):
                if col_start <= box_center <= col_end:
                    column_boxes[i].append(box)
                    break
        
        # Sort boxes within each column (top to bottom)
        for col_idx in column_boxes:
            column_boxes[col_idx].sort(key=lambda b: -b.y1)  # Sort by top edge (descending)
        
        # Sort columns by their left edge
        sorted_columns = sorted(column_boxes.items(), key=lambda x: columns[x[0]][0])
        
        # Flatten the list of boxes in reading order
        result = []
        for col_idx, boxes in sorted_columns:
            result.extend(boxes)
            
        return result
    
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
