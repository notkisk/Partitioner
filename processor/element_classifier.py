from typing import Dict, Any, List, Tuple
import re
from .data_models import Element, ElementType, ElementMetadata, CoordinatesMetadata
from .patterns import (
    is_list_item_start,
    is_title_pattern,
    is_header_footer,
    is_footnote
)
from .list_handler import identify_potential_list_items
from .nlp_utils import (
    sentence_count,
    under_non_alpha_ratio,
    exceeds_cap_ratio,
    contains_verb,
    contains_english_word
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
    from utils.text_analysis import is_title, is_list_item
    
    # Initialize context if needed
    if 'prev_item' not in context:
        context['prev_item'] = None
    coords = CoordinatesMetadata(
        x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3],
        page_width=page_info['width'],
        page_height=page_info['height']
    )
    
    metadata = ElementMetadata(
        style_info=style_info,
        coordinates=coords
    )
    
    # Get median font size for comparison
    median_font = context.get('median_font_size', 12)
    
    # Start classification process with default type
    element_type = ElementType.NARRATIVE_TEXT
    
    # 1. Check for list items first (high priority)
    if identify_potential_list_items(text, style_info, {
        'indent_ratio': coords.indent_ratio,
        'x0': bbox[0],
        'y0': bbox[1]
    }):
        element_type = ElementType.LIST_ITEM
        metadata.confidence = 0.9
        
    # 2. Check for titles if not a list item
    elif _is_likely_title(text, style_info, median_font):
        element_type = ElementType.TITLE
        metadata.confidence = 0.85
        
    # 3. Check for headers/footers
    elif _is_header_footer(text, coords, page_info):
        element_type = ElementType.HEADER if coords.y0 < page_info['height'] / 2 else ElementType.FOOTER
        metadata.confidence = 0.8
        
    # 4. Check for page numbers
    elif _is_page_number(text, coords, page_info):
        element_type = ElementType.PAGE_NUMBER
        metadata.confidence = 0.95
        
    # 5. Check for footnotes
    elif _is_footnote(text, coords, page_info):
        element_type = ElementType.FOOTNOTE
        metadata.confidence = 0.85
        
    # 6. Default to narrative text
    else:
        metadata.confidence = 0.7
    
    return Element(
        text=text,
        element_type=element_type,
        bbox=bbox,
        page_number=page_number,
        metadata=metadata
    )

def _is_likely_title(text: str, style_info: Dict[str, Any], median_font: float) -> bool:
    """Determine if text block is likely a title based on style and content."""
    # Early exit if it's a list item
    if is_list_item_start(text):
        return False

    # Content-based checks first
    text = text.strip()
    
    # Multi-sentence check
    if sentence_count(text, min_word_length=5) > 1:
        return False
    
    # Word length check
    words = text.split()
    if len(words) > 12:  # Title shouldn't be too wordy
        return False
    
    # Non-alpha ratio check
    if under_non_alpha_ratio(text, threshold=0.5):
        return False
    
    # Language check
    if not contains_english_word(text):
        return False
    
    # Capitalization checks
    if text.isupper() and text.strip().endswith('.'):
        return False
    
    if text.strip().endswith(','):
        return False
    
    if text.replace('.', '').replace(',', '').strip().isdigit():
        return False
    
    # Style-based checks
    font_size = style_info.get('font_size', 0)
    if font_size <= median_font * 1.2:
        return False
    
    # Title pattern check
    if is_title_pattern(text):
        return True
    
    # Additional style checks
    if style_info.get('is_bold', False) and font_size > median_font:
        # For bold text, be more lenient with capitalization
        return True
    
    # For non-bold text, require stricter capitalization
    return exceeds_cap_ratio(text, threshold=0.7)

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
        if coords.points[1][1] < page_info['height'] * 0.2:
            return True
    return False
