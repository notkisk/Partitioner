from typing import List, Dict, Any, Tuple, Optional
from pdf_layout_analyzer import PDFLayoutAnalyzer, TextRegion
from text_block_analyzer import TextBlockAnalyzer, BlockAnalysis
from pdf_processor import PDFProcessor
from geometry_utils import validate_bbox, boxes_iou
from dataclasses import dataclass

@dataclass
class AnalyzedElement:
    text: str
    bbox: Tuple[float, float, float, float]
    element_type: str
    confidence: float
    page_number: int
    metadata: Dict[str, Any]
    style_info: Dict[str, Any]
    cross_page_refs: List[int]

class PDFAnalyzer:
    def __init__(self, dpi: int = 72):
        """Initialize the PDF analyzer"""
        self.dpi = dpi
        self.processor = PDFProcessor(dpi)
        self.text_analyzer = TextBlockAnalyzer()
        self.document_stats = {
            "font_sizes": [],
            "line_heights": [],
            "median_font_size": 0,
            "median_line_height": 0,
            "dominant_body_font": None,
            "element_types": {}
        }
        self.analyzed_pages = {}

    def analyze_document(self, pdf_path: str) -> List[Dict]:
        """Process and analyze a PDF document with all available analyzers"""
        # Process PDF pages
        pages = self.processor.process_pages(pdf_path)
        analyzed_elements = []

        # First pass: collect document statistics
        all_font_sizes = []
        all_line_heights = []
        font_counts = {}
        
        # Reset document statistics
        self.document_stats = {
            "font_sizes": [],
            "line_heights": [],
            "median_font_size": 0,
            "median_line_height": 0,
            "dominant_body_font": None,
            "element_types": {}
        }
        
        for page_num, page in enumerate(pages, 1):
            layout_analyzer = PDFLayoutAnalyzer(page.height, self.dpi)
            regions = layout_analyzer.analyze_layout_structure(page)
            
            # Update document stats
            all_font_sizes.extend(layout_analyzer.style_stats["font_sizes"])
            all_line_heights.extend(layout_analyzer.style_stats["line_heights"])
            
            # Track font usage
            for region in regions:
                font = region.font_info.get("dominant_font")
                if font:
                    font_counts[font] = font_counts.get(font, 0) + len(region.text.split())
            
            self.analyzed_pages[page_num] = {
                "regions": regions,
                "stats": layout_analyzer.style_stats
            }
            
        # Update document-wide metrics
        if all_font_sizes:
            self.document_stats["font_sizes"] = all_font_sizes
            self.document_stats["median_font_size"] = sorted(all_font_sizes)[len(all_font_sizes) // 2]
        
        if all_line_heights:
            self.document_stats["line_heights"] = all_line_heights
            self.document_stats["median_line_height"] = sorted(all_line_heights)[len(all_line_heights) // 2]
            
        if font_counts:
            self.document_stats["dominant_body_font"] = max(font_counts.items(), key=lambda x: x[1])[0]
        
        # Second pass: analyze with cross-page context
        for page_num, page_data in self.analyzed_pages.items():
            page_elements_dicts = []
            for region in page_data["regions"]:
                # Clean text first
                clean_text = self.text_analyzer._clean_text(region.text)
                
                # Prepare position info
                position_info = {
                    "relative_y": region.bbox[1] / pages[page_num-1].height,
                    "indent_ratio": region.bbox[0] / pages[page_num-1].width,
                    "width_ratio": (region.bbox[2] - region.bbox[0]) / pages[page_num-1].width
                }

                analysis = self.text_analyzer.analyze_block_structure(
                    clean_text,
                    region.font_info,
                    position_info,
                    page_num
                )
                
                # Get element type first
                element_type = self._determine_element_type(analysis)
                
                # Create element dict
                element_dict = {
                    "text": clean_text,
                    "bbox": region.bbox,
                    "element_type": element_type,
                    "confidence": analysis.confidence,
                    "page_number": page_num,
                    "metadata": {
                        "style_stats": region.style_stats,
                        "analysis_details": analysis.__dict__
                    },
                    "style_info": region.font_info,
                    "cross_page_refs": analysis.cross_page_refs or []
                }
                
                # Update element type statistics
                self.document_stats["element_types"][element_type] = \
                    self.document_stats["element_types"].get(element_type, 0) + 1
                
                # Special handling for list items
                if element_type == "ListItem" and page_elements_dicts:
                    prev_element = page_elements_dicts[-1]
                    if prev_element["element_type"] == "NarrativeText" and \
                       prev_element["bbox"][0] > element_dict["bbox"][0]:
                        # This is likely a continuation of the previous list item
                        element_dict["element_type"] = "ListItem"
                        # Update statistics
                        self.document_stats["element_types"]["NarrativeText"] -= 1
                        self.document_stats["element_types"]["ListItem"] = \
                            self.document_stats["element_types"].get("ListItem", 0) + 1
                
                page_elements_dicts.append(element_dict)
            
            # Merge blocks within each page
            merged_elements = self.text_analyzer._merge_blocks(page_elements_dicts)
            analyzed_elements.extend(merged_elements)

        return analyzed_elements

    def _determine_element_type(self, analysis: BlockAnalysis) -> str:
        """Determine final element type based on analysis results"""
        if analysis.is_title and analysis.confidence > 0.7:
            return "Title"
        elif analysis.is_header_footer:
            return "Header" if analysis.metadata.get("is_header", False) else "Footer"
        elif analysis.is_list_item:
            return "ListItem"
        elif analysis.is_page_number:
            return "PageNumber"
        elif analysis.is_footnote:
            return "Footnote"
        elif analysis.is_annotation:
            return "Annotation"
        return "NarrativeText"

    def get_document_statistics(self) -> Dict[str, Any]:
        """Get document-wide statistics and metrics"""
        return {
            "element_counts": self.document_stats["element_types"],
            "median_font_size": self.document_stats.get("median_font_size", 0),
            "median_line_height": self.document_stats.get("median_line_height", 0),
            "total_elements": sum(self.document_stats["element_types"].values()),
            "pages_analyzed": len(self.analyzed_pages)
        }
