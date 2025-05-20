from typing import Dict, Any, List, Tuple
from .data_models import Element, ElementType, ElementMetadata, CoordinatesMetadata
from .text_analysis import (
    is_possible_title,
    is_possible_narrative,
    is_list_item,
    is_header_footer,
    is_footnote,
    is_contact_info,
    get_text_stats
)

def classify_element(text: str, bbox: Tuple[float, float, float, float], style_info: Dict[str, Any], page_info: Dict[str, float], page_number: int, context: Dict[str, Any]) -> Element:
    """Classify a text block into its appropriate element type.
    
    Args:
        text: The text content to classify
        bbox: Bounding box coordinates (x0, y0, x1, y1)
        style_info: Font and style information
        page_info: Page dimensions and metrics
        page_number: Current page number
        context: Additional contextual information
        
    Returns:
        Element: Classified element with metadata
    """
    # Create coordinate metadata
    coords = CoordinatesMetadata(
        x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3],
        page_width=page_info['width'],
        page_height=page_info['height']
    )
    
    # Initialize metadata
    metadata = ElementMetadata(
        style_info=style_info,
        coordinates=coords
    )
    
    # Prepare style info with median font size
    style_info['median_font_size'] = context.get('median_font_size', 12)
    
    # Start classification process
    element_type = ElementType.NARRATIVE_TEXT
    
    # 1. Check for list items (high priority)
    import re
    section_title_pat = r"^\d+(?:\.\d+)*[\s\-:]+[A-Za-z]"
    bullet_pat = r"^(\s*[\u2022\u2023\u25E6\u2043\u2219\*-]|\d+\.|[a-zA-Z]\))\s+"
    if re.match(section_title_pat, text.strip()):
        element_type = ElementType.TITLE
        metadata.confidence = 0.97
    elif is_list_item(text) and not re.match(section_title_pat, text.strip()):
        element_type = ElementType.LIST_ITEM
        metadata.confidence = 0.9
    elif not element_type == ElementType.LIST_ITEM:
        is_title, confidence = is_possible_title(text, style_info)
        if is_title:
            element_type = ElementType.TITLE
            metadata.confidence = confidence
    
    # 3. Check for page numbers (high priority)
    elif is_page_number(text):
        if coords.relative_y > 0.85:
            element_type = ElementType.PAGE_NUMBER
            metadata.confidence = 0.98

    # 4. Check for footers
    elif is_footer(text) and coords.relative_y > 0.85:
        element_type = ElementType.FOOTER
        metadata.confidence = 0.9

    # 5. Check for headers
    elif is_header_footer(text) and coords.relative_y < 0.15:
        element_type = ElementType.HEADER
        metadata.confidence = 0.8

    # 6. Check for footnotes
    elif coords.relative_y < 0.15 and is_footnote(text):
        element_type = ElementType.FOOTNOTE
        metadata.confidence = 0.85
    
    # 6. Check for contact information
    elif is_contact_info(text):
        element_type = ElementType.CONTACT_INFO
        metadata.confidence = 0.9
    
    # 7. Verify if narrative text
    elif element_type == ElementType.NARRATIVE_TEXT:
        if is_possible_narrative(text):
            metadata.confidence = 0.7
        else:
            element_type = ElementType.UNKNOWN
            metadata.confidence = 0.3
    
    return Element(
        text=text,
        element_type=element_type,
        bbox=bbox,
        page_number=page_number,
        metadata=metadata
    )

def _is_likely_title(text: str, style_info: Dict[str, Any], median_font: float) -> Tuple[bool, float]:
    """Determines if a text block is likely a title using improved heuristics.
    
    Args:
        text: The text content to analyze
        style_info: Font and style information
        median_font: Median font size for comparison
        
    Returns:
        Tuple[bool, float]: (is_title, confidence_score)
    """
    # Clean and normalize text
    text = text.strip()
    
    # Very short texts can't be titles
    if len(text) < 2:
        return False, 0.0
        
    # Initialize confidence score
    confidence = 0.0
    
    # Check text case properties
    is_all_uppercase = text.isupper()
    is_title_case = text == text.title() and not text.isupper()
    
    # Font size comparison (weighted heavily)
    font_size = style_info.get('font_size', median_font)
    if font_size > median_font * 1.2:
        confidence += 0.3
    
    # Font style (bold/italic)
    if style_info.get('is_bold', False):
        confidence += 0.2
    if style_info.get('is_italic', False):
        confidence += 0.1
        
    # Text case analysis
    if is_all_uppercase:
        confidence += 0.15
    elif is_title_case:
        confidence += 0.1
        
    # Length checks
    word_count = len(text.split())
    if word_count <= 15:  # Titles are usually short
        confidence += 0.1
    elif word_count > 25:  # Too long for a title
        return False, 0.0
        
    # Sentence structure
    sentence_count = len(re.split(r'[.!?]+', text))
    if sentence_count > 2:  # Titles rarely have multiple sentences
        return False, 0.0
    
    # Punctuation analysis
    if text.endswith(('.', ':', ';')):  # Titles usually don't end with these
        confidence -= 0.2
    
    # Section numbering patterns
    if re.match(r'^\d+\.?\d*\s+\w+', text):  # "1.2 Section Title"
        confidence += 0.3
    elif re.match(r'^[A-Z]\.\s+\w+', text):   # "A. Section Title"
        confidence += 0.25
        
    # Common title keywords
    title_keywords = r'^(chapter|section|appendix|figure|table|introduction|conclusion|abstract|summary)\s+\d*'
    if re.match(title_keywords, text.lower()):
        confidence += 0.25
        
    # Content analysis
    if contains_verb(text):  # Titles often don't contain verbs
        confidence -= 0.1
    
    # Position on page (if available)
    y_pos = style_info.get('y_position', 0.5)  # Normalized 0-1
    if y_pos > 0.8 or y_pos < 0.1:  # Titles are rarely at very top/bottom
        confidence -= 0.15
        
    # Final decision
    is_title = confidence > 0.4  # Threshold for title classification
    
    return is_title, min(1.0, max(0.0, confidence))

def _is_header_footer(text: str, coords: CoordinatesMetadata, page_info: Dict[str, float]) -> bool:
    """Determine if text block is a header or footer."""
    # Position check
    margin_threshold = 0.1  # 10% of page height
    if coords.relative_y > margin_threshold and coords.relative_y < (1 - margin_threshold):
        return False
        
    return is_header_footer(text)

def _is_page_number(text: str, coords: CoordinatesMetadata, page_info: Dict[str, float]) -> bool:
    """Determine if text block is a page number."""
    # Must be short
    if len(text) > 10:
        return False
        
    # Try to convert to number
    try:
        int(text.strip())
        return True
    except ValueError:
        pass
        
    # Check for common page number formats
    import re
    page_patterns = [
        r'^\d+$',
        r'^-\s*\d+\s*-$',
        r'^\[\d+\]$',
        r'^Page\s+\d+$'
    ]
    
    return any(re.match(pattern, text.strip()) for pattern in page_patterns)

def _is_footnote(text: str, coords: CoordinatesMetadata, page_info: Dict[str, float]) -> bool:
    """Determine if text block is a footnote."""
    # Check if text starts with a footnote marker
    if re.match(r'^\d+\.\s+|^\*+\s+', text):
        # Check if text is at the bottom of the page
        if coords.y0 < page_info['height'] * 0.2:
            return True
    return False
