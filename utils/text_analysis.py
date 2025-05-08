from typing import Dict, Any, Optional
import re

def get_text_metrics(text: str) -> Dict[str, float]:
    """Calculate various text metrics."""
    words = text.split()
    total_words = len(words)
    if total_words == 0:
        return {
            'avg_word_length': 0,
            'cap_ratio': 0,
            'punct_ratio': 0
        }
    
    cap_words = sum(1 for w in words if w and w[0].isupper())
    total_chars = sum(len(w) for w in words)
    punct_chars = sum(1 for c in text if c in '.,;:!?')
    
    return {
        'avg_word_length': total_chars / total_words,
        'cap_ratio': cap_words / total_words,
        'punct_ratio': punct_chars / total_chars if total_chars > 0 else 0
    }

def is_list_item(text: str, bbox: tuple, prev_item: Optional[Dict] = None) -> bool:
    """Check if text is a list item.
    
    Args:
        text: Text to check
        bbox: Bounding box of the text
        prev_item: Previous item's information (for list continuation)
        
    Returns:
        bool: True if text is a list item
    """
    # Skip empty or whitespace text
    if not text or not text.strip():
        return False
    # Common bullet point characters and their variations
    bullet_patterns = [
        # Unicode bullets and symbols
        r'^\s*[•‣◦⁃∙]\s+',
        # Numbers with various formats
        r'^\s*\d+[.)]\s+',
        r'^\s*\(\d+\)\s+',
        r'^\s*\[\d+\]\s+',
        # Letters with various formats
        r'^\s*[a-zA-Z][.)]\s+',
        r'^\s*\([a-zA-Z]\)\s+',
        r'^\s*\[[a-zA-Z]\]\s+',
        # Roman numerals
        r'^\s*(?:i{1,3}|iv|v|vi{1,3}|ix|x)[.)]\s+',
        # Common symbols and special cases
        r'^\s*[-*+~]\s+',
        r'^\s*Step\s+\d+[:.)]\s+',
        r'^\s*\d+\.\d+(?:\.\d+)*\s+'
    ]
    
    # Check if text matches any bullet pattern
    if any(re.match(pattern, text) for pattern in bullet_patterns):
        return True
        
    # Check for continuation with previous item
    if prev_item and prev_item.get('is_list_item', False):
        # Check indentation and vertical spacing
        indent_diff = abs(bbox[0] - prev_item['bbox'][0])
        vertical_diff = abs(bbox[1] - prev_item['bbox'][3])  # Distance between this top and prev bottom
        
        # Must be similarly indented and close vertically
        if indent_diff < 5 and vertical_diff < 20:  # 5 points horizontal, 20 points vertical tolerance
            # Check for common list continuation patterns
            if any([
                text.strip().startswith(('and', 'or', 'but', 'nor', 'for', 'so', 'yet')),  # Conjunctions
                re.match(r'^[a-z]\)|^\d+\)|^[A-Z]\)|^\s+\w', text),  # Sub-items
                not text[0].isupper() and not text.strip()[0].isdigit()  # Continuation sentences
            ]):
                return True
    
    return False

def is_title(text: str, style_info: Dict[str, Any], context: Dict[str, Any]) -> float:
    """Calculate confidence score (0-1) that text is a title."""
    score = 0.0
    metrics = get_text_metrics(text)
    
    # Length check
    words = text.split()
    if 2 <= len(words) <= 20:
        score += 0.2
    
    # Font size comparison
    median_font = context.get('median_font_size', 12)
    if style_info.get('font_size', 0) > median_font * 1.2:
        score += 0.3
    
    # Capitalization check
    if metrics['cap_ratio'] > 0.7:
        score += 0.2
    
    # Style check
    if style_info.get('is_bold', False):
        score += 0.2
    
    # Position check
    y_pos = style_info.get('bbox', (0, 0, 0, 0))[1]
    page_height = context.get('page_height', 1000)
    if y_pos > page_height * 0.8:
        score += 0.1
    
    return min(score, 1.0)
