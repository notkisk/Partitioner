import re
from typing import Dict, Any, Optional, Tuple
from .data_models import CoordinatesMetadata

def is_likely_caption(
    text: str, 
    bbox: Tuple[float, float, float, float], 
    style_info: Dict[str, Any], 
    page_info: Dict[str, float],
    elements_before: list = None,
    elements_after: list = None
) -> Tuple[bool, float]:
    """
    Determine if a text block is likely a caption with strict rules.
    
    Args:
        text: The text content to analyze
        bbox: Bounding box coordinates (x0, y0, x1, y1)
        style_info: Font and style information
        page_info: Page dimensions and metrics
        elements_before: List of elements before this one on the page
        elements_after: List of elements after this one on the page
        
    Returns:
        Tuple[bool, float]: (is_caption, confidence_score)
    """
    if not text.strip() or not bbox or len(bbox) < 4:
        return False, 0.0
    
    confidence = 0.0
    reasons = []
    
    # Basic text checks
    text = text.strip()
    if not text:
        return False, 0.0
        
    text_lower = text.lower()
    words = [w for w in text.split() if w.strip()]
    word_count = len(words)
    
    # Skip very long text (unlikely to be a caption)
    if word_count > 50 or len(text) > 400:
        return False, 0.0
    
    # 1. Check for caption patterns (strong indicator)
    from .text_patterns import CAPTION_RE
    has_caption_prefix = False
    
    # Check first few words for caption patterns
    first_few_words = ' '.join(words[:4]).lower()
    for pattern in CAPTION_RE:
        if pattern.search(first_few_words):
            has_caption_prefix = True
            confidence += 0.9  # Very strong indicator
            reasons.append(f"Starts with caption pattern: {pattern.pattern}")
            break
    
    # 2. Check text length (captions are typically short to medium length)
    if 3 <= word_count <= 100:  # More lenient length requirements
        confidence += 0.3
        reasons.append(f"Appropriate length ({word_count} words)")
    
    # 3. Check position on page (captions are often centered)
    page_width = page_info.get('width', 0)
    if page_width > 0:
        center_x = (bbox[0] + bbox[2]) / 2
        relative_center = center_x / page_width
        if 0.4 <= relative_center <= 0.6:  # Centered
            confidence += 0.1
            reasons.append("Centered position")
    
    # 4. Check if this is directly below a figure/table (strong indicator)
    if elements_before and len(elements_before) > 0 and bbox and len(bbox) >= 4:
        # Look at previous elements to find figures/tables/images
        for i in range(1, min(5, len(elements_before) + 1)):  # Check up to 4 previous elements
            last_element = elements_before[-i]
            last_bbox = None
            last_type = None
            
            # Get bounding box and type of the previous element
            if hasattr(last_element, 'bbox'):
                last_bbox = last_element.bbox
                last_type = getattr(last_element, 'element_type', None)
            elif isinstance(last_element, dict):
                last_bbox = last_element.get('bbox')
                last_type = last_element.get('element_type')
            
            if not last_bbox or len(last_bbox) < 4:
                continue
                
            # Skip if previous element is very small (likely not a figure/table)
            if (last_bbox[2] - last_bbox[0]) * (last_bbox[3] - last_bbox[1]) < 400:  # Less than 400 square points
                continue
                
            # Calculate vertical gap between elements
            vertical_gap = bbox[1] - last_bbox[3]
            
            # Check if this is directly below the previous element with reasonable gap
            if 0 < vertical_gap <= 50:  # Increased vertical gap tolerance to 50pt
                # Calculate horizontal overlap
                overlap_start = max(bbox[0], last_bbox[0])
                overlap_end = min(bbox[2], last_bbox[2])
                overlap_width = max(0, overlap_end - overlap_start)
                
                # Calculate overlap percentage relative to caption width
                caption_width = bbox[2] - bbox[0]
                overlap_pct = overlap_width / caption_width if caption_width > 0 else 0
                
                # Check if there's significant horizontal overlap (at least 50%)
                if overlap_pct > 0.5:
                    # Check if the element above is likely a figure/table
                    is_figure_above = False
                    if last_type and any(t in str(last_type).lower() for t in ['image', 'figure', 'table', 'chart']):
                        is_figure_above = True
                    
                    # If not sure by type, check size and aspect ratio
                    if not is_figure_above:
                        width = last_bbox[2] - last_bbox[0]
                        height = last_bbox[3] - last_bbox[1]
                        if width > 100 and height > 50:  # Reasonable minimum size
                            aspect_ratio = width / height if height > 0 else 0
                            if 0.5 < aspect_ratio < 3.0:  # Reasonable aspect ratio for figures
                                is_figure_above = True
                    
                    if is_figure_above:
                        confidence += 0.6
                        reasons.append(f"Positioned below potential figure/table (gap: {vertical_gap:.1f}pt, overlap: {overlap_pct:.1%})")
                    else:
                        confidence += 0.3
                        reasons.append(f"Positioned below element (gap: {vertical_gap:.1f}pt, overlap: {overlap_pct:.1%})")
                    
                    # If we found a good candidate, no need to check further
                    if confidence >= 0.7:
                        break
    
    # 5. Enhanced caption phrase detection
    caption_phrases = [
        # Figure/Table patterns
        (r'^(?:fig(?:ure)?|table|chart|diagram|image|photo|graph|illustration|plate|exhibit|panel|drawing|sketch|picture|map|photograph|equation|formula|algorithm|code|example|program|script|schematic|flowchart|graphic|visualization|plot|listing)\s*[0-9a-z]', 0.8),
        # Position indicators
        (r'\b(?:left|right|top|bottom|above|below|center|middle|side|corner)\b', 0.3),
        # Source/credit indicators
        (r'\b(?:source|credit|from|courtesy of|photograph by|image by|photo by|©|copyright|credit line|photo credit|image credit|courtesy|permission|reproduced from)\b', 0.5),
        # Caption indicators
        (r'\b(caption|figure|fig|table|chart|diagram|image|photo|graph|illustration|plate|exhibit|panel|drawing|sketch|picture|map|photograph|equation|formula|algorithm|code|example|program|script|schematic|flowchart|graphic|visualization|plot|listing)[:.]?\s*[0-9a-z]', 0.9)
    ]
    
    # Check for patterns in text
    for pattern, weight in caption_phrases:
        if re.search(pattern, text_lower, re.IGNORECASE):
            confidence += weight
            reasons.append(f"Matches caption pattern: {pattern}")
    
    # Check for common caption structures (e.g., "Figure 1:", "Table 2.5:")
    caption_number_patterns = [
        r'^(?:fig(?:ure)?|table|chart|diagram|image|photo|graph)\s*[0-9]+(?:\.[0-9]+)*[.:]?',
        r'^[A-Za-z]\s*[0-9]+[.:]',
        r'^\(?[A-Za-z0-9]\s*[):]'
    ]
    
    first_three_words = ' '.join(words[:3])
    for pattern in caption_number_patterns:
        if re.match(pattern, first_three_words, re.IGNORECASE):
            confidence += 0.7
            reasons.append(f"Matches numbered caption pattern: {pattern}")
            break
    
    # 6. Check formatting (captions are often in italics or smaller font)
    font_size = style_info.get('size', style_info.get('font_size', 0))
    median_font = page_info.get('median_font_size', 12)
    
    if font_size > 0 and median_font > 0:
        # Captions are often slightly smaller than body text
        size_ratio = font_size / median_font
        if size_ratio < 0.9:  # At least 10% smaller than median
            confidence += 0.3
            reasons.append(f"Smaller font size ({font_size:.1f}pt vs median {median_font:.1f}pt, {size_ratio:.1f}x)")
        elif size_ratio < 1.1:  # Within 10% of median
            confidence += 0.1  # Slight positive for normal size
        else:
            confidence -= 0.1  # Slight negative for larger than normal text
    
    # 7. Check if text is in italics (common for captions)
    if style_info.get('italic', False):
        confidence += 0.2
        reasons.append("Italicized text")
    
    # 8. Penalize elements that are too far from any figure/table
    if elements_before and len(elements_before) > 1 and bbox and len(bbox) >= 4:
        # Check distance to previous elements
        min_distance = float('inf')
        for elem in elements_before[-3:]:  # Check last 3 elements
            elem_bbox = getattr(elem, 'bbox', None) or (elem.get('bbox') if isinstance(elem, dict) else None)
            if elem_bbox and len(elem_bbox) >= 4:
                distance = bbox[1] - elem_bbox[3]  # Vertical distance
                if 0 < distance < min_distance:
                    min_distance = distance
        
        if min_distance > 30:  # More than 30pt from any previous element
            confidence -= 0.3
            reasons.append(f"Too far from previous elements ({min_distance:.1f}pt)")
    
    # 9. Check text characteristics
    if text and text[0].isupper():
        # Only add confidence if it's not a single word (which could be a title)
        if word_count > 1:
            confidence += 0.1
            reasons.append("Starts with capital letter")
    
    # Check for sentence case (captions often don't end with period)
    if text and text[-1] not in '.!?':
        confidence += 0.1
        reasons.append("No sentence-ending punctuation")
    
    # 10. Text length and structure analysis
    if word_count > 50 or len(text) > 300:  # Very long text is less likely to be a caption
        confidence -= 0.3
        reasons.append("Very long text, likely not a caption")
    elif word_count > 25:  # Moderately long text
        confidence -= 0.1
        reasons.append("Long text, might not be a caption")
    
    # 11. Check if this is a continuation of a previous caption or part of a caption block
    if elements_before and len(elements_before) > 0:
        # Check previous element
        prev_element = elements_before[-1]
        prev_text = getattr(prev_element, 'text', '') if not isinstance(prev_element, dict) else prev_element.get('text', '')
        
        # Check if previous element ends with common caption separators
        if prev_text and any(prev_text.strip().endswith(p) for p in [':', ';', ',']):
            confidence += 0.3
            reasons.append("Continues from previous caption")
        
        # Check if previous element is already a caption
        prev_type = getattr(prev_element, 'element_type', None) or (prev_element.get('element_type') if isinstance(prev_element, dict) else None)
        if prev_type == ElementType.CAPTION:
            confidence += 0.4
            reasons.append("Previous element is a caption")
    
    # 12. Check for common caption patterns in the text
    # Look for patterns like "Figure 1:" or "Table 2.5:" etc.
    caption_patterns = [
        r'(?:fig(?:ure)?|table|chart|diagram|image|photo)\s*\d+(?:\.\d+)*[.:]?',
        r'^(?:fig\.?\s*\d+|table\s+[ivxcl]+|fig\s*[a-z])\b',
        r'^\s*(?:[a-z]+\.\s*)?\(?\s*(?:fig|fig\.|figure|table|tab|chart|diag|ill|img|photo)\.?\s*\d+\s*\)?',
        r'^\s*\[?(?:fig|figure|table|tab|chart|diagram|illustration|image|photo)\.?\s*\d+\]?',
    ]
    
    for pattern in caption_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            confidence += 0.4
            reasons.append(f"Matches caption pattern: {pattern}")
            break
    
    # 13. Check for common caption endings (like credit lines)
    credit_indicators = [
        'source:', 'credit:', 'courtesy of', 'photograph by', '©', 'copyright',
        'image by', 'photo by', 'courtesy', 'from', 'after', 'modified from'
    ]
    
    if any(indicator in text_lower for indicator in credit_indicators):
        confidence += 0.3
        reasons.append("Contains credit/source information")
    
    # 14. Final confidence adjustment based on multiple factors
    # If we have multiple indicators, boost the confidence
    indicator_count = len([r for r in reasons if any(x in r.lower() for x in [
        'prefix', 'phrase', 'caption', 'figure', 'table', 'pattern', 'credit', 'source'
    ])])
    
    if indicator_count >= 2:
        confidence += min(0.3, 0.15 * indicator_count)  # Cap the boost at 0.3
        reasons.append(f"Multiple caption indicators ({indicator_count})")
    
    # 15. Check if this is a continuation of a previous caption
    if elements_before and len(elements_before) > 0:
        prev_element = elements_before[-1]
        prev_bbox = None
        prev_text = ''
        
        if hasattr(prev_element, 'bbox'):
            prev_bbox = prev_element.bbox
            prev_text = getattr(prev_element, 'text', '')
        elif isinstance(prev_element, dict):
            prev_bbox = prev_element.get('bbox')
            prev_text = prev_element.get('text', '')
        
        if prev_bbox and len(prev_bbox) >= 4:
            # Check if previous element is very close and aligned
            vertical_gap = bbox[1] - prev_bbox[3]
            if 0 < vertical_gap <= 15:  # Very close vertically
                # Check horizontal alignment
                left_aligned = abs(bbox[0] - prev_bbox[0]) < 10
                right_aligned = abs(bbox[2] - prev_bbox[2]) < 10
                
                if left_aligned or right_aligned:
                    # Check if previous text ends without punctuation
                    if prev_text and prev_text.strip() and prev_text.strip()[-1] not in '.!?':
                        confidence += 0.4
                        reasons.append("Continues from previous caption")
    
    # Cap confidence at 1.0
    confidence = min(1.0, max(0.0, confidence))
    
    # Debug logging for high-confidence captions
    if confidence > 0.7:
        import logging
        logging.debug(f"Potential caption (conf: {confidence:.2f}): {text[:100]}...")
        logging.debug(f"  Reasons: {', '.join(reasons)}")
    
    # Check for common caption phrases
    caption_phrases = [
        'figure', 'fig.', 'table', 'chart', 'diagram', 'illustration',
        'image', 'photo', 'source:', 'credit:', 'caption:'
    ]
    if any(phrase in text_lower for phrase in caption_phrases):
        confidence += 0.4
    
    # Check if this is a single line (many captions are)
    if '\n' not in text:
        confidence += 0.1
    
    # Check if this is followed by a figure/table (less likely to be a caption)
    if elements_after and len(elements_after) > 0:
        next_element = elements_after[0]
        next_bbox = None
        
        # Handle both Element objects and dictionaries
        if hasattr(next_element, 'bbox'):  # It's an Element object
            next_bbox = next_element.bbox
        elif isinstance(next_element, dict):  # It's a dictionary
            next_bbox = next_element.get('bbox')
        
        if next_bbox and bbox and len(next_bbox) >= 4 and len(bbox) >= 4:
            # If next element is very close below, this might be a heading, not a caption
            vertical_gap = next_bbox[1] - bbox[3]
            if 0 < vertical_gap < 10:  # Very small gap
                confidence -= 0.2
    
    # Normalize confidence to [0, 1]
    confidence = max(0.0, min(1.0, confidence))
    
    return confidence > 0.5, confidence

def find_caption_for_figure(figure_element, elements: list) -> Optional[int]:
    """
    Find the caption for a given figure element.
    
    Args:
        figure_element: The figure element to find a caption for (can be Element or dict)
        elements: List of all elements on the page
        
    Returns:
        int: Index of the caption element, or None if not found
    """
    if not figure_element or not elements:
        return None
    
    # Handle both Element objects and dictionaries
    def get_bbox(element):
        if hasattr(element, 'bbox'):  # It's an Element object
            return element.bbox
        elif isinstance(element, dict):  # It's a dictionary
            return element.get('bbox')
        return None
    
    def get_style_info(element):
        if hasattr(element, 'metadata') and hasattr(element.metadata, 'style_info'):
            return element.metadata.style_info
        elif isinstance(element, dict):
            return element.get('metadata', {}).get('style_info', {})
        return {}
    
    def get_text(element):
        if hasattr(element, 'text'):
            return element.text
        return element.get('text', '')
    
    figure_bbox = get_bbox(figure_element)
    if not figure_bbox or len(figure_bbox) < 4:
        return None
    
    # Find the figure in the elements list
    figure_idx = -1
    for idx, element in enumerate(elements):
        if element is figure_element or (isinstance(element, dict) and element == figure_element):
            figure_idx = idx
            break
    
    if figure_idx == -1:
        return None
    
    # Look for captions below the figure (most common position)
    for i in range(figure_idx + 1, min(figure_idx + 5, len(elements))):
        element = elements[i]
        element_bbox = get_bbox(element)
        if element_bbox and len(element_bbox) >= 4 and element_bbox[1] > figure_bbox[3]:  # Below the figure
            is_caption, _ = is_likely_caption(
                get_text(element),
                element_bbox,
                get_style_info(element),
                {'width': element_bbox[2] * 2, 'height': element_bbox[3] * 2},
                elements_before=elements[max(0, i-3):i],
                elements_after=elements[i+1:i+4]
            )
            if is_caption:
                return i
    
    # If not found below, check above
    for i in range(max(0, figure_idx - 3), figure_idx):
        element = elements[i]
        element_bbox = get_bbox(element)
        if element_bbox and len(element_bbox) >= 4 and element_bbox[3] < figure_bbox[1]:  # Above the figure
            is_caption, _ = is_likely_caption(
                get_text(element),
                element_bbox,
                get_style_info(element),
                {'width': element_bbox[2] * 2, 'height': element_bbox[3] * 2},
                elements_before=elements[max(0, i-3):i],
                elements_after=elements[i+1:i+4]
            )
            if is_caption:
                return i
    
    return None
