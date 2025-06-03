import re
from typing import Dict, Any, List, Tuple
from .data_models import Element, ElementType, ElementMetadata, CoordinatesMetadata
from .text_analysis import (
    is_possible_title,
    is_possible_narrative,
    is_list_item,
    is_header_footer,
    is_footnote,
    is_contact_info,
    is_page_number,
    is_footer,
    get_text_stats
)
from .caption_detector import is_likely_caption

def classify_element(
    text: str, 
    bbox: Tuple[float, float, float, float], 
    style_info: Dict[str, Any], 
    page_info: Dict[str, float], 
    page_number: int, 
    context: Dict[str, Any],
    elements_before: List[Dict] = None,
    elements_after: List[Dict] = None
) -> Element:
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
    
    # Start classification process with default values
    element_type = ElementType.TEXT
    confidence = 0.7  # Default confidence for regular text
    
    # Clean and prepare text
    text = text.strip()
    if not text:
        return Element(
            text="",
            element_type=ElementType.DISCARDED,
            bbox=bbox,
            page_number=page_number,
            metadata=ElementMetadata(
                style_info=style_info,
                coordinates=coords,
                confidence=1.0
            )
        )
    
    # Initialize variables
    is_title, title_confidence = is_possible_title(text, style_info)
    is_list = False  # Initialize is_list to avoid UnboundLocalError
    
    # 1. First check for section headers (highest priority)
    if is_title and title_confidence > 0.6:  # Only consider high-confidence titles
        # Check for section number patterns (e.g., "1.2.3 Section Title")
        section_pattern = r'^(\d+(?:\.\d+)+)\s+[A-Z]'
        if re.match(section_pattern, text):
            element_type = ElementType.TITLE
            confidence = max(0.9, title_confidence)  # High confidence for section headers
    
    # 2. Check for list items (high priority, but lower than section headers)
    if element_type != ElementType.TITLE:  # Only check for lists if not already identified as title
        is_list = is_list_item(text)
        if is_list:
            element_type = ElementType.LIST_ITEM
            confidence = 0.9
    
    # 3. Check for other types of titles (lower priority than list items)
    if element_type == ElementType.TEXT and is_title and title_confidence > 0.4:
        element_type = ElementType.TITLE
        confidence = max(confidence, title_confidence)
    
    # 3. Check for block equations (medium priority)
    # This is a simplified check - you might want to enhance this
    if '=' in text and ('\\' in text or '$' in text):
        element_type = ElementType.BLOCK_EQUATION
        confidence = 0.9
    
    # 4. Check for captions (only if we have elements before to check against)
    if not is_list and element_type == ElementType.TEXT and elements_before:
        try:
            # First check if this is likely a caption based on text patterns
            is_caption, caption_confidence = is_likely_caption(
                text, 
                bbox, 
                style_info, 
                page_info, 
                elements_before, 
                elements_after or []
            )
            
            if is_caption and caption_confidence > 0.6:  # Higher threshold to reduce false positives
                is_image_caption = False
                figure_found = False
                
                # Look at previous elements to find a potential figure/table
                for i in range(1, min(5, len(elements_before) + 1)):  # Check up to 4 previous elements
                    prev_element = elements_before[-i]
                    prev_bbox = getattr(prev_element, 'bbox', None) or (prev_element.get('bbox') if isinstance(prev_element, dict) else None)
                    if not prev_bbox or len(prev_bbox) < 4:
                        continue
                    
                    # Skip text elements that are likely not figures
                    if hasattr(prev_element, 'element_type') and 'text' in str(prev_element.element_type).lower():
                        continue
                        
                    # Calculate element dimensions and position
                    width = prev_bbox[2] - prev_bbox[0]
                    height = prev_bbox[3] - prev_bbox[1]
                    
                    # Skip very small elements
                    if width < 80 or height < 60:  # Increased minimum size
                        continue
                    
                    # Check vertical gap (caption should be below figure)
                    vertical_gap = bbox[1] - prev_bbox[3]
                    if vertical_gap < 0 or vertical_gap > 60:  # Caption should be below with reasonable gap
                        continue
                    
                    # Calculate horizontal overlap
                    overlap_start = max(bbox[0], prev_bbox[0])
                    overlap_end = min(bbox[2], prev_bbox[2])
                    overlap_width = max(0, overlap_end - overlap_start)
                    
                    # Require significant horizontal overlap (60% of caption width)
                    caption_width = bbox[2] - bbox[0]
                    if caption_width > 0:
                        overlap_pct = overlap_width / caption_width
                        if overlap_pct < 0.6:  # Require at least 60% overlap
                            continue
                    
                    # Check if the element above is figure-like
                    prev_type = getattr(prev_element, 'element_type', None) or (prev_element.get('element_type') if isinstance(prev_element, dict) else None)
                    
                    # If we have a type, use it to determine if it's a figure/table
                    if prev_type:
                        prev_type_str = str(prev_type).lower()
                        if any(t in prev_type_str for t in ['image', 'figure', 'table', 'chart']):
                            is_image_caption = True
                            figure_found = True
                            break
                    
                    # If no type, check size and aspect ratio
                    aspect_ratio = width / height if height > 0 else 0
                    if 0.4 < aspect_ratio < 2.5:  # Reasonable aspect ratio for figures/tables
                        is_image_caption = True
                        figure_found = True
                        break
                
                # Only classify as caption if we found a matching figure/table
                if figure_found:
                    if is_image_caption:
                        element_type = ElementType.FIGURE_CAPTION
                        confidence = min(0.95, caption_confidence * 1.1)  # Cap confidence
                    else:
                        element_type = ElementType.TABLE_CAPTION
                        confidence = min(0.9, caption_confidence)
                    
                    # Return early if we're confident this is a caption
                    if confidence > 0.75:
                        return Element(
                            text=text,
                            element_type=element_type,
                            bbox=bbox,
                            page_number=page_number,
                            metadata=ElementMetadata(
                                style_info=style_info,
                                coordinates=coords,
                                confidence=confidence
                            )
                        )
        except Exception as e:
            import logging
            logging.debug(f"Caption detection warning: {e}", exc_info=True)
    
    # 5. Check for page numbers (high confidence when detected)
    if is_page_number(text) and coords.relative_y > 0.85:
        element_type = ElementType.TEXT  # Page numbers are just regular text
        confidence = 0.98
    
    # 6. Check for footers/headers (treated as regular text but with position info)
    elif (is_footer(text) or is_header_footer(text)) and (coords.relative_y > 0.85 or coords.relative_y < 0.15):
        element_type = ElementType.TEXT
        confidence = 0.9
    
    # 7. Check for footnotes (treated as regular text)
    elif is_footnote(text) and coords.relative_y < 0.15:
        element_type = ElementType.TEXT
        confidence = 0.9
    
    # 8. Check for contact information (treated as regular text)
    elif is_contact_info(text):
        element_type = ElementType.TEXT
        confidence = 0.9
    
    # Set the final confidence
    metadata.confidence = confidence
    
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
