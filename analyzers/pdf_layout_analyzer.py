from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass, field
from pdfminer.layout import (
    LTPage, LTTextBox, LTTextBoxHorizontal, LTTextLineHorizontal,
    LTChar, LTImage, LTFigure, LTLine, LTRect, LTCurve
)
from geometry_utils import (
    validate_bbox, boxes_iou, minimum_containing_coords,
    is_box_subregion, merge_overlapping_boxes, sort_boxes
)

@dataclass
class TextRegion:
    text: str = ""
    bbox: Tuple[float, float, float, float] = None
    font_info: Dict[str, Any] = field(default_factory=dict)
    style_stats: Dict[str, Any] = field(default_factory=dict)
    element_type: str = "NarrativeText"
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    children: List["TextRegion"] = field(default_factory=list)

class PDFLayoutAnalyzer:
    def __init__(self, page_height: float, dpi: int = 72):
        self.page_height = page_height
        self.dpi = dpi
        self.scale = dpi / 72.0
        self.style_stats = {
            "font_sizes": [],
            "font_names": set(),
            "line_heights": [],
            "char_widths": []
        }

    def analyze_text_block(self, block: LTTextBoxHorizontal) -> TextRegion:
        """Analyze a text block for style and structure"""
        chars = []
        lines = []
        text_parts = []
        
        # Extract characters and lines
        for line in block:
            if isinstance(line, LTTextLineHorizontal):
                line_chars = [c for c in line if isinstance(c, LTChar)]
                if line_chars:
                    chars.extend(line_chars)
                    lines.append({
                        "chars": line_chars,
                        "bbox": (line.x0, line.y0, line.x1, line.y1),
                        "text": line.get_text()
                    })
                    text_parts.append(line.get_text())

        if not chars:
            return None

        # Analyze font properties
        sizes = [c.size for c in chars]
        fonts = [c.fontname for c in chars]
        widths = [c.width for c in chars]
        
        # Calculate line heights
        line_heights = []
        for i in range(len(lines) - 1):
            height = lines[i]["bbox"][1] - lines[i+1]["bbox"][3]
            line_heights.append(height)

        # Update document statistics
        self.style_stats["font_sizes"].extend(sizes)
        self.style_stats["font_names"].update(fonts)
        self.style_stats["line_heights"].extend(line_heights)
        self.style_stats["char_widths"].extend(widths)

        # Calculate block metrics
        font_info = {
            "avg_font_size": np.mean(sizes),
            "std_font_size": np.std(sizes),
            "dominant_font": max(set(fonts), key=fonts.count),
            "is_bold": any("bold" in f.lower() for f in fonts),
            "is_italic": any("italic" in f.lower() for f in fonts),
            "unique_fonts": len(set(fonts))
        }

        # Analyze text properties
        text = "\\n".join(text_parts)
        words = text.split()
        style_stats = {
            "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
            "line_count": len(lines),
            "char_count": len(chars),
            "avg_line_height": np.mean(line_heights) if line_heights else 0,
            "avg_char_width": np.mean(widths),
            "text_density": len(chars) / (block.height * block.width) if block.height * block.width > 0 else 0
        }

        return TextRegion(
            text=text,
            bbox=(block.x0, block.y0, block.x1, block.y1),
            font_info=font_info,
            style_stats=style_stats
        )

    def analyze_layout_structure(self, page: LTPage) -> List[TextRegion]:
        """Analyze the structural layout of a page"""
        regions = []
        
        # First pass: collect all text blocks and their properties
        for obj in page:
            if isinstance(obj, LTTextBoxHorizontal):
                region = self.analyze_text_block(obj)
                if region:
                    regions.append(region)

        if not regions:
            return []

        # Calculate document-wide statistics
        doc_stats = {
            "median_font_size": np.median(self.style_stats["font_sizes"]),
            "median_line_height": np.median(self.style_stats["line_heights"]) if self.style_stats["line_heights"] else 0,
            "median_char_width": np.median(self.style_stats["char_widths"])
        }

        # Second pass: analyze spatial relationships
        sorted_regions = sorted(regions, key=lambda r: (-r.bbox[1], r.bbox[0]))  # Top-to-bottom, left-to-right
        
        # Group related regions
        grouped_regions = []
        current_group = None
        
        for region in sorted_regions:
            if current_group is None:
                current_group = region
                continue

            # Check if regions should be merged
            should_merge = self._should_merge_regions(current_group, region, doc_stats)
            
            if should_merge:
                # Merge regions
                current_group = self._merge_regions(current_group, region)
            else:
                grouped_regions.append(current_group)
                current_group = region

        if current_group:
            grouped_regions.append(current_group)

        return grouped_regions

    def _should_merge_regions(self, r1: TextRegion, r2: TextRegion, doc_stats: Dict[str, float]) -> bool:
        """Determine if two regions should be merged based on spatial and style properties"""
        # Check vertical spacing
        vertical_gap = r1.bbox[3] - r2.bbox[1]  # Distance between bottom of r1 and top of r2
        if vertical_gap > doc_stats["median_line_height"] * 1.5:
            return False

        # Check horizontal alignment
        x_diff = abs(r1.bbox[0] - r2.bbox[0])
        if x_diff > doc_stats["median_char_width"] * 2:
            return False

        # Check font consistency
        font_size_diff = abs(r1.font_info["avg_font_size"] - r2.font_info["avg_font_size"])
        if font_size_diff > 0.5:
            return False

        if r1.font_info["dominant_font"] != r2.font_info["dominant_font"]:
            return False

        return True

    def _merge_regions(self, r1: TextRegion, r2: TextRegion) -> TextRegion:
        """Merge two text regions into one"""
        # Combine text
        text = r1.text + "\\n" + r2.text

        # Update bounding box
        bbox = minimum_containing_coords(r1.bbox, r2.bbox)

        # Update font info
        font_info = r1.font_info.copy()
        if r2.font_info["avg_font_size"] > font_info["avg_font_size"]:
            font_info = r2.font_info.copy()

        # Combine style stats
        style_stats = {
            "avg_word_length": (r1.style_stats["avg_word_length"] + r2.style_stats["avg_word_length"]) / 2,
            "line_count": r1.style_stats["line_count"] + r2.style_stats["line_count"],
            "char_count": r1.style_stats["char_count"] + r2.style_stats["char_count"],
            "avg_line_height": (r1.style_stats["avg_line_height"] + r2.style_stats["avg_line_height"]) / 2,
            "avg_char_width": (r1.style_stats["avg_char_width"] + r2.style_stats["avg_char_width"]) / 2,
            "text_density": (r1.style_stats["text_density"] + r2.style_stats["text_density"]) / 2
        }

        return TextRegion(
            text=text,
            bbox=bbox,
            font_info=font_info,
            style_stats=style_stats,
            element_type=r1.element_type,
            confidence=max(r1.confidence, r2.confidence)
        )
