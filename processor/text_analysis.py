from typing import List, Tuple
import re
from nltk import pos_tag, sent_tokenize, word_tokenize

from .text_patterns import (
    US_PHONE_NUMBERS_RE,
    US_CITY_STATE_ZIP_RE,
    UNICODE_BULLETS_RE,
    UNICODE_BULLETS_RE_0W,
    E_BULLET_PATTERN,
    REFERENCE_PATTERN_RE,
    ENUMERATED_BULLETS_RE,
    EMAIL_HEAD_RE,
    PARAGRAPH_PATTERN_RE,
    DOUBLE_PARAGRAPH_PATTERN_RE,
    LINE_BREAK_RE,
    ONE_LINE_BREAK_PARAGRAPH_PATTERN_RE,
    IP_ADDRESS_PATTERN_RE,
    EMAIL_ADDRESS_PATTERN_RE,
    ENDS_IN_PUNCT_RE,
    NUMBERED_LIST_RE,
    TITLE_RE,
    PAGE_NUMBER_RE,
    FOOTER_RE
)

def contains_verb(text: str) -> bool:
    """Check if text contains any verbs using POS tagging."""
    if text.isupper():
        text = text.lower()
    pos_tags = pos_tag(word_tokenize(text))
    return any(tag in POS_VERB_TAGS for _, tag in pos_tags)

def sentence_count(text: str, min_length: int = None) -> int:
    """Count sentences in text, optionally filtering by minimum word length."""
    sentences = sent_tokenize(text)
    if min_length is None:
        return len(sentences)
    
    count = 0
    for sentence in sentences:
        words = word_tokenize(sentence)
        if len(words) >= min_length:
            count += 1
    return count

def get_text_stats(text: str) -> Tuple[float, float, float]:
    """Calculate various text statistics.
    
    Returns:
        Tuple containing:
        - alpha_ratio: Proportion of alpha characters
        - cap_ratio: Proportion of capitalized words
        - special_char_ratio: Proportion of special characters
    """
    if not text:
        return 0.0, 0.0, 0.0
        
    # Calculate alpha ratio
    chars = [c for c in text if not c.isspace()]
    alpha_chars = [c for c in chars if c.isalpha()]
    alpha_ratio = len(alpha_chars) / len(chars) if chars else 0
    
    # Calculate capitalization ratio
    words = [w for w in text.split() if any(c.isalpha() for c in w)]
    cap_words = [w for w in words if w[0].isupper() or w.isupper()]
    cap_ratio = len(cap_words) / len(words) if words else 0
    
    # Calculate special character ratio
    special_chars = [c for c in chars if not (c.isalnum() or c.isspace())]
    special_ratio = len(special_chars) / len(chars) if chars else 0
    
    return alpha_ratio, cap_ratio, special_ratio

def is_possible_title(text: str, style_info: dict = None) -> Tuple[bool, float]:
    """Check if text could be a title using multiple heuristics."""
    if not text or len(text) < 2:
        return False, 0.0
    
    confidence = 0.0
    text = text.strip()
    
    # Length checks
    words = text.split()
    if len(words) > 15:  # Titles shouldn't be too long
        return False, 0.0
    if len(words) < 10:  # Short text more likely to be title
        confidence += 0.1
    
    # Pattern matching
    if TITLE_RE.search(text):
        confidence += 0.3
    
    # Style checks if available
    if style_info:
        if style_info.get("is_bold"):
            confidence += 0.2
        if style_info.get("font_size", 0) > style_info.get("median_font_size", 0):
            confidence += 0.2
    
    # Text characteristics
    alpha_ratio, cap_ratio, special_ratio = get_text_stats(text)
    
    if alpha_ratio < 0.3:  # Titles should contain sufficient text
        return False, 0.0
    
    if cap_ratio > 0.7:  # High capitalization suggests title
        confidence += 0.2
    
    if special_ratio > 0.3:  # Too many special chars unlikely for title
        confidence -= 0.2
    
    # Sentence structure
    if sentence_count(text) > 1:
        return False, 0.0
    
    if text.endswith((",", ":", ";")): # Titles don't typically end with these
        confidence -= 0.2
    
    # Final decision
    is_title = confidence > 0.3
    return is_title, min(1.0, confidence)

def is_possible_narrative(
    text: str,
    cap_threshold: float = 0.5,
    alpha_threshold: float = 0.5
) -> bool:
    """Check if text could be narrative content."""
    if not text or text.isnumeric():
        return False
    
    # Get text statistics
    alpha_ratio, cap_ratio, _ = get_text_stats(text)
    
    # Check thresholds
    if alpha_ratio < alpha_threshold:
        return False
    if cap_ratio > cap_threshold:
        return False
    
    # Check for sufficient content
    if sentence_count(text, min_length=3) >= 2:
        return True
    
    return contains_verb(text)

def is_list_item(text: str) -> bool:
    text = text.strip()
    return bool(UNICODE_BULLETS_RE.match(text) or NUMBERED_LIST_RE.match(text) or ENUMERATED_BULLETS_RE.match(text))

def is_header_footer(text: str) -> bool:
    """Check if text matches header/footer patterns."""
    return bool(HEADER_FOOTER_RE.search(text.lower()))

def is_footnote(text: str) -> bool:
    return bool(REFERENCE_PATTERN_RE.search(text.strip()))

def is_page_number(text: str) -> bool:
    return bool(PAGE_NUMBER_RE.match(text.strip()))

def is_footer(text: str) -> bool:
    return bool(FOOTER_RE.search(text.strip()))

def is_contact_info(text: str) -> bool:
    text = text.strip()
    return bool(
        US_CITY_STATE_ZIP_RE.match(text) or
        EMAIL_ADDRESS_PATTERN_RE.match(text) or
        US_PHONE_NUMBERS_RE.search(text)
    )
