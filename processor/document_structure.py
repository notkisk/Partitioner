from typing import List, Dict, Any, Tuple, Optional
import re
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto

class SplitType(Enum):
    NONE = auto()
    CONTINUATION = auto()  # "continued on next page"
    SPLIT_HEADER = auto()  # Header split across pages
    SPLIT_FOOTER = auto()  # Footer split across pages
    SPLIT_TABLE = auto()   # Table split across pages
    SPLIT_LIST = auto()    # List split across pages
    SPLIT_PARAGRAPH = auto() # Paragraph split across pages

@dataclass
class SplitMarker:
    split_type: SplitType
    source_page: int
    target_page: int
    source_bbox: Tuple[float, float, float, float] = None
    target_bbox: Tuple[float, float, float, float] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

def detect_split_continuations(pages: List[Dict[str, Any]]) -> List[SplitMarker]:
    """Detect continuation markers like 'continued on next page'."""
    markers = []
    continuation_patterns = [
        r'continued\s+on\s+next\s+page',
        r'to\s+be\s+continued',
        r'cont\.?\s*$',
        r'continues\s+next',
    ]
    
    for i in range(len(pages) - 1):
        current_page = pages[i]
        next_page = pages[i + 1]
        
        # Check last block of current page
        if current_page['blocks']:
            last_block = current_page['blocks'][-1]
            for pattern in continuation_patterns:
                if re.search(pattern, last_block['text'], re.IGNORECASE):
                    markers.append(SplitMarker(
                        split_type=SplitType.CONTINUATION,
                        source_page=i + 1,  # 1-based page numbers
                        target_page=i + 2,
                        source_bbox=last_block['bbox'],
                        confidence=0.9
                    ))
        
        # Check first block of next page
        if next_page['blocks']:
            first_block = next_page['blocks'][0]
            for pattern in continuation_patterns:
                if re.search(pattern, first_block['text'], re.IGNORECASE):
                    markers.append(SplitMarker(
                        split_type=SplitType.CONTINUATION,
                        source_page=i + 1,
                        target_page=i + 2,
                        source_bbox=first_block['bbox'],
                        confidence=0.9
                    ))
    
    return markers

def detect_split_headers_footers(pages: List[Dict[str, Any]]) -> List[SplitMarker]:
    """Detect headers and footers that continue across pages."""
    markers = []
    
    if len(pages) < 2:
        return []
    
    # Compare first and last blocks of consecutive pages
    for i in range(len(pages) - 1):
        current_page = pages[i]
        next_page = pages[i + 1]
        
        # Check headers (first block of each page)
        if current_page['blocks'] and next_page['blocks']:
            current_header = current_page['blocks'][0]
            next_header = next_page['blocks'][0]
            
            # Simple similarity check (can be enhanced with more sophisticated comparison)
            if (current_header['text'].strip() == next_header['text'].strip() and 
                abs(current_header['bbox'][0] - next_header['bbox'][0]) < 20 and  # Similar x-position
                current_header['style_info'].get('font') == next_header['style_info'].get('font')):
                
                markers.append(SplitMarker(
                    split_type=SplitType.SPLIT_HEADER,
                    source_page=i + 1,
                    target_page=i + 2,
                    source_bbox=current_header['bbox'],
                    target_bbox=next_header['bbox'],
                    confidence=0.8
                ))
        
        # Check footers (last block of each page)
        if current_page['blocks'] and next_page['blocks']:
            current_footer = current_page['blocks'][-1]
            next_footer = next_page['blocks'][-1]
            
            if (current_footer['text'].strip() == next_footer['text'].strip() and 
                abs(current_footer['bbox'][0] - next_footer['bbox'][0]) < 20 and
                current_footer['style_info'].get('font') == next_footer['style_info'].get('font')):
                
                markers.append(SplitMarker(
                    split_type=SplitType.SPLIT_FOOTER,
                    source_page=i + 1,
                    target_page=i + 2,
                    source_bbox=current_footer['bbox'],
                    target_bbox=next_footer['bbox'],
                    confidence=0.8
                ))
    
    return markers

def detect_split_tables(pages: List[Dict[str, Any]]) -> List[SplitMarker]:
    """Detect tables that are split across pages."""
    markers = []
    table_patterns = [
        r'table\s+\d+',
        r'tab\s*\d+',
        r'tbl\s*\d+',
    ]
    
    for i in range(len(pages) - 1):
        current_page = pages[i]
        next_page = pages[i + 1]
        
        # Check if last block of current page is part of a table
        if current_page['blocks']:
            last_block = current_page['blocks'][-1]
            if any(re.search(pattern, last_block['text'], re.IGNORECASE) for pattern in table_patterns):
                # Check if first block of next page continues the table
                if next_page['blocks']:
                    first_block = next_page['blocks'][0]
                    if any(re.search(pattern, first_block['text'], re.IGNORECASE) for pattern in table_patterns):
                        markers.append(SplitMarker(
                            split_type=SplitType.SPLIT_TABLE,
                            source_page=i + 1,
                            target_page=i + 2,
                            source_bbox=last_block['bbox'],
                            target_bbox=first_block['bbox'],
                            confidence=0.85
                        ))
    
    return markers

def detect_split_lists(pages: List[Dict[str, Any]]) -> List[SplitMarker]:
    """Detect lists that are split across pages."""
    markers = []
    
    for i in range(len(pages) - 1):
        current_page = pages[i]
        next_page = pages[i + 1]
        
        # Check if last block of current page is a list item
        if current_page['blocks']:
            last_block = current_page['blocks'][-1]
            if last_block.get('element_type') == 'LIST_ITEM':
                # Check if first block of next page is also a list item
                if next_page['blocks']:
                    first_block = next_page['blocks'][0]
                    if first_block.get('element_type') == 'LIST_ITEM':
                        # Check if they have similar indentation
                        if abs(last_block['bbox'][0] - first_block['bbox'][0]) < 10:
                            markers.append(SplitMarker(
                                split_type=SplitType.SPLIT_LIST,
                                source_page=i + 1,
                                target_page=i + 2,
                                source_bbox=last_block['bbox'],
                                target_bbox=first_block['bbox'],
                                confidence=0.9
                            ))
    
    return markers

def detect_split_paragraphs(pages: List[Dict[str, Any]]) -> List[SplitMarker]:
    """Detect paragraphs that are split across pages."""
    markers = []
    
    for i in range(len(pages) - 1):
        current_page = pages[i]
        next_page = pages[i + 1]
        
        if not current_page['blocks'] or not next_page['blocks']:
            continue
            
        last_block = current_page['blocks'][-1]
        first_block = next_page['blocks'][0]
        
        # Check if last block of current page is a paragraph
        if (last_block.get('element_type') == 'TEXT' and 
            first_block.get('element_type') == 'TEXT' and
            not last_block['text'].strip().endswith(('.', '!', '?')) and
            not first_block['text'].strip().startswith(('•', '-', '•', '▪', '▫', '○', '■', '□', '▪', '▫', '●', '○', '◆', '◇', '▶', '▷'))):
            
            # Check if the text flows naturally (same indentation, similar style)
            if (abs(last_block['bbox'][0] - first_block['bbox'][0]) < 15 and
                last_block['style_info'].get('font') == first_block['style_info'].get('font') and
                abs(last_block['style_info'].get('size', 0) - first_block['style_info'].get('size', 0)) < 0.5):
                
                markers.append(SplitMarker(
                    split_type=SplitType.SPLIT_PARAGRAPH,
                    source_page=i + 1,
                    target_page=i + 2,
                    source_bbox=last_block['bbox'],
                    target_bbox=first_block['bbox'],
                    confidence=0.8
                ))
    
    return markers

class DocumentStructureAnalyzer:
    """Analyzes document structure and detects split content across pages."""
    
    def __init__(self):
        self.split_markers = []
    
    def analyze(self, pages: List[Dict[str, Any]]) -> List[SplitMarker]:
        """Analyze document pages and detect split content."""
        self.split_markers = []
        
        # Run all detection functions
        self.split_markers.extend(detect_split_continuations(pages))
        self.split_markers.extend(detect_split_headers_footers(pages))
        self.split_markers.extend(detect_split_tables(pages))
        self.split_markers.extend(detect_split_lists(pages))
        self.split_markers.extend(detect_split_paragraphs(pages))
        
        # Sort markers by source page and y-coordinate
        self.split_markers.sort(key=lambda m: (m.source_page, -m.source_bbox[1] if m.source_bbox else 0))
        
        return self.split_markers
    
    def get_continuation_markers(self) -> List[SplitMarker]:
        """Get all continuation markers."""
        return [m for m in self.split_markers if m.split_type == SplitType.CONTINUATION]
    
    def get_split_elements(self) -> List[SplitMarker]:
        """Get all split content markers (excluding continuations)."""
        return [m for m in self.split_markers if m.split_type != SplitType.CONTINUATION]
    
    def get_splits_for_page(self, page_num: int) -> List[SplitMarker]:
        """Get all split markers for a specific page."""
        return [m for m in self.split_markers if m.source_page == page_num or m.target_page == page_num]
