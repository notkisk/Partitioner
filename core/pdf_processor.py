from typing import List, Dict, Any, Tuple, Optional
from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams
import re
from dataclasses import dataclass
from pdfminer.layout import (
    LTPage, LTTextBox, LTTextBoxHorizontal, LTTextLineHorizontal,
    LTChar, LTImage, LTFigure, LTLine, LTRect, LTCurve
)
from geometry_utils import (
    validate_bbox, boxes_iou, minimum_containing_coords,
    is_box_subregion, merge_overlapping_boxes, sort_boxes
)

@dataclass
class LayoutElement:
    text: str
    bbox: Tuple[float, float, float, float]
    font_info: Dict[str, Any]
    element_type: str = "NarrativeText"
    metadata: Dict[str, Any] = None
    page_number: int = 1
    source: str = "pdfminer"

class PDFProcessor:
    def __init__(self, dpi: int = 72):
        self.dpi = dpi
        self.scale = dpi / 72.0
        self.la_params = LAParams(
            line_margin=0.5,
            char_margin=2.0,
            word_margin=0.1,
            line_overlap=0.5
        )

    def process_pages(self, pdf_path: str) -> List[LTPage]:
        """Process all pages in a PDF document"""
        pages = []
        for page_layout in extract_pages(pdf_path, laparams=self.la_params):
            if isinstance(page_layout, LTPage):
                pages.append(page_layout)
        return pages

    def rect_to_bbox(self, rect: Tuple[float, float, float, float], 
                    page_height: float) -> Tuple[float, float, float, float]:
        """Convert PDFMiner coordinates to normalized bbox"""
        x0, y0, x1, y1 = rect
        # Convert from bottom-left to top-left origin
        return (
            x0 * self.scale,
            (page_height - y1) * self.scale,
            x1 * self.scale,
            (page_height - y0) * self.scale
        )

    def extract_text_objects(self, obj: LTTextBox) -> List[Dict[str, Any]]:
        """Extract text and style info from text objects"""
        elements = []
        if isinstance(obj, LTTextBox):
            for line in obj:
                if isinstance(line, LTTextLineHorizontal):
                    chars = [c for c in line if isinstance(c, LTChar)]
                    if not chars:
                        continue

                    # Extract font information
                    sizes = [c.size for c in chars]
                    fonts = [c.fontname for c in chars]
                    font_info = {
                        "avg_font_size": sum(sizes) / len(sizes),
                        "dominant_font": max(set(fonts), key=fonts.count),
                        "is_bold": any("bold" in f.lower() for f in fonts),
                        "char_count": len(chars)
                    }

                    elements.append({
                        "text": line.get_text().strip(),
                        "bbox": (line.x0, line.y0, line.x1, line.y1),
                        "font_info": font_info
                    })

        return elements

    def extract_image_objects(self, obj: LTImage) -> Dict[str, Any]:
        """Extract image information"""
        return {
            "bbox": (obj.x0, obj.y0, obj.x1, obj.y1),
            "type": "Image",
            "metadata": {
                "width": obj.width,
                "height": obj.height,
                "stream": bool(obj.stream),
                "name": getattr(obj, "name", None)
            }
        }

    def process_page_layout(self, page: LTPage) -> List[LayoutElement]:
        """Process a PDFMiner page layout into LayoutElements"""
        raw_elements = []
        page_height = page.height

        # First pass: extract all text and image objects
        for obj in page:
            if isinstance(obj, LTTextBoxHorizontal):
                text_elements = self.extract_text_objects(obj)
                for elem in text_elements:
                    elem["bbox"] = self.rect_to_bbox(elem["bbox"], page_height)
                    raw_elements.append(elem)
            elif isinstance(obj, LTImage):
                img_info = self.extract_image_objects(obj)
                img_info["bbox"] = self.rect_to_bbox(img_info["bbox"], page_height)
                raw_elements.append(img_info)

        # Remove duplicates
        filtered_elements = []
        for i, elem in enumerate(raw_elements):
            is_duplicate = False
            for j in range(i + 1, len(raw_elements)):
                if boxes_iou(elem["bbox"], raw_elements[j]["bbox"]) > 0.9:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_elements.append(elem)

        # Sort elements top-to-bottom, left-to-right
        filtered_elements.sort(key=lambda x: (-x["bbox"][1], x["bbox"][0]))

        # Group elements
        grouped_elements = self.group_elements(filtered_elements)

        # Convert to LayoutElements
        layout_elements = []
        for elem in grouped_elements:
            layout_elements.append(LayoutElement(
                text=elem["text"],
                bbox=elem["bbox"],
                font_info=elem["font_info"],
                element_type=elem.get("type", "NarrativeText"),
                metadata=elem.get("metadata", {}),
                page_number=page.pageid,
                source="pdfminer"
            ))

        return layout_elements

    def group_elements(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group related elements based on spatial and style relationships"""
        if not elements:
            return []

        grouped = []
        current_group = None
        
        for i, elem in enumerate(elements):
            if current_group is None:
                current_group = elem.copy()
                continue

            # Check if current element should be merged with current group
            should_merge = False
            
            # Vertical proximity check
            vertical_gap = elem["bbox"][1] - current_group["bbox"][3]
            avg_font_size = current_group["font_info"]["avg_font_size"]
            
            if vertical_gap <= avg_font_size * 1.5:
                # Check horizontal alignment
                x_diff = abs(elem["bbox"][0] - current_group["bbox"][0])
                if x_diff < avg_font_size:
                    # Check font consistency
                    if (elem["font_info"]["dominant_font"] == current_group["font_info"]["dominant_font"] and
                        abs(elem["font_info"]["avg_font_size"] - avg_font_size) < 0.5):
                        should_merge = True

            if should_merge:
                # Merge elements
                current_group["text"] += "\\n" + elem["text"]
                current_group["bbox"] = minimum_containing_coords(
                    current_group["bbox"], elem["bbox"]
                )
                # Update font info if needed
                if elem["font_info"]["avg_font_size"] > avg_font_size:
                    current_group["font_info"] = elem["font_info"]
            else:
                # Start new group
                grouped.append(current_group)
                current_group = elem.copy()

        if current_group:
            grouped.append(current_group)

        return grouped
