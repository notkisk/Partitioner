import nltk
from typing import List, Tuple, Dict, Any
import re
from nltk import pos_tag, sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from dataclasses import dataclass

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')

# POS tags that indicate a verb
POS_VERB_TAGS = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}

@dataclass
class TextStats:
    alpha_ratio: float
    cap_ratio: float
    special_ratio: float
    word_count: int
    sentence_count: int
    avg_word_length: float
    has_verb: bool
    is_title_case: bool
    is_all_caps: bool

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
    FOOTER_RE,
    re  # Add re for compiling patterns
)

# Define header patterns
HEADER_PATTERNS = [
    r'^[A-Z][A-Z0-9 _-]*$',  # All caps with optional numbers and separators
    r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$',  # Title Case
    r'^[IVX]+\.?\s+[A-Z]',  # Roman numerals followed by text
    r'^\d+\.\s+[A-Z]',  # Numbered sections
    r'^[A-Z][A-Za-z0-9 ]*:$',  # Text ending with colon
    r'^(?:introduction|abstract|acknowledgments?|references|appendix|glossary|index|contents|table of contents|list of (?:figures|tables))$',
]

# Compile header patterns
HEADER_FOOTER_RE = re.compile('|'.join(HEADER_PATTERNS), re.IGNORECASE)

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

def get_text_stats(text: str) -> TextStats:
    """Calculate comprehensive text statistics.
    
    Args:
        text: Input text to analyze
        
    Returns:
        TextStats object containing various text metrics
    """
    if not text:
        return TextStats(
            alpha_ratio=0.0,
            cap_ratio=0.0,
            special_ratio=0.0,
            word_count=0,
            sentence_count=0,
            avg_word_length=0.0,
            has_verb=False,
            is_title_case=False,
            is_all_caps=False
        )
    
    # Basic character analysis
    chars = [c for c in text if not c.isspace()]
    alpha_chars = [c for c in chars if c.isalpha()]
    special_chars = [c for c in chars if not (c.isalnum() or c.isspace())]
    
    # Word analysis
    words = [w for w in text.split() if any(c.isalpha() for c in w)]
    word_count = len(words)
    
    # Calculate metrics
    alpha_ratio = len(alpha_chars) / len(chars) if chars else 0.0
    cap_words = [w for w in words if w[0].isupper() or w.isupper()]
    cap_ratio = len(cap_words) / word_count if word_count > 0 else 0.0
    special_ratio = len(special_chars) / len(chars) if chars else 0.0
    
    # Additional metrics
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0.0
    sentence_count = len(sent_tokenize(text))
    has_verb = contains_verb(text)
    is_title_case = text.istitle()
    is_all_caps = text.isupper()
    
    return TextStats(
        alpha_ratio=alpha_ratio,
        cap_ratio=cap_ratio,
        special_ratio=special_ratio,
        word_count=word_count,
        sentence_count=sentence_count,
        avg_word_length=avg_word_length,
        has_verb=has_verb,
        is_title_case=is_title_case,
        is_all_caps=is_all_caps
    )

def is_possible_title(text: str, style_info: dict = None) -> Tuple[bool, float]:
    """Check if text could be a title using comprehensive heuristics.
    
    Args:
        text: Text to evaluate
        style_info: Optional dictionary with style information
        
    Returns:
        Tuple of (is_title, confidence_score)
    """
    if not text or len(text) < 2:
        return False, 0.0
    
    text = text.strip()
    stats = get_text_stats(text)
    confidence = 0.0
    
    # Check for section number patterns (e.g., "1.2.3 Section Title")
    section_pattern = r'^(\d+(?:\.\d+)+)\s+[A-Z]'
    if re.match(section_pattern, text):
        confidence += 0.8  # Very strong indicator of a section title
    
    # Check for common caption patterns first
    caption_indicators = [
        'figure', 'fig', 'table', 'chart', 'diagram', 'illustration',
        'image', 'photo', 'source:', 'credit:', 'caption:', 'plate', 'graph',
        'exhibit', 'panel', 'drawing', 'sketch', 'picture',
        'map', 'photograph', 'photo', 'equation', 'formula', 'algorithm',
        'code', 'example', 'program', 'script', 'schematic', 'flowchart',
        'graphic', 'visualization', 'plot', 'listing', 'diagram',
        'left:', 'right:', 'top:', 'bottom:', 'source:', 'from:', 'courtesy of',
        '©', 'copyright', 'credit', 'photograph by', 'image by', 'photo by'
    ]
    
    text_lower = text.lower()
    if any(indicator in text_lower for indicator in caption_indicators):
        return False, 0.0  # Very likely a caption, not a title
    
    # Check for caption patterns like "Figure 1:" or "Table 2.5:"
    caption_patterns = [
        r'(?:fig(?:\.?|ure)?|table|chart|diagram|image|photo)\s*\d+(?:\.\d+)*[.:]?',
        r'^[A-Za-z]\s*\d+[.:]',  # Single letter followed by number (e.g., "A.1")
        r'^\(?[A-Za-z]\s*\d+[.:]?\s*[^a-z]',  # (a) or (1) at start
    ]
    
    if any(re.search(pattern, text_lower) for pattern in caption_patterns):
        return False, 0.0
    
    # Length-based checks
    if stats.word_count > 12:  # Very long for a title
        return False, 0.0
    elif stats.word_count <= 1:  # Single word titles are less likely in documents
        confidence += 0.1
    
    # Pattern matching - only match more specific title patterns
    if TITLE_RE.search(text) and stats.word_count > 1:  # Require at least 2 words for title patterns
        confidence += 0.3
    
    # Style-based scoring - require stronger style indicators
    if style_info:
        # Font size is a strong indicator but needs to be significantly larger
        font_size = style_info.get('font_size', 0)
        median_size = style_info.get('median_font_size', 12)
        size_ratio = font_size / median_size if median_size > 0 else 1.0
        
        if size_ratio > 1.5:  # Must be 50% larger than median
            confidence += 0.4
        elif size_ratio > 1.3:  # 30% larger
            confidence += 0.2
        elif size_ratio > 1.1:  # 10% larger
            confidence += 0.1
        
        # Font weight and style - require both bold and larger size
        if style_info.get('is_bold') and size_ratio > 1.2:
            confidence += 0.2
        elif style_info.get('is_bold'):
            confidence += 0.1
            
        if style_info.get('is_italic'):
            confidence -= 0.1  # Titles are less likely to be in italics
    
    # Text characteristics - needs sufficient text content
    if stats.alpha_ratio < 0.5:  # Needs more text content
        confidence -= 0.2
    
    # Case analysis - require proper title case for higher confidence
    if stats.is_title_case and stats.word_count > 1:
        confidence += 0.3
    elif text and text[0].isupper():
        confidence += 0.1  # Only slight boost for sentence case
    
    # Penalize text that looks like a caption
    if ':' in text and stats.word_count < 4:  # Short text with colon is likely a label
        confidence -= 0.3
    
    # Ensure confidence is within bounds
    confidence = max(0.0, min(1.0, confidence))
    
    # Capitalization patterns
    if stats.cap_ratio > 0.7:
        confidence += 0.15
    
    # Special characters (penalize excessive use)
    if stats.special_ratio > 0.3:
        confidence -= 0.2
    
    # Position-based scoring (if available)
    if style_info and 'y_position' in style_info:
        y_pos = style_info['y_position']  # Normalized 0-1
        if y_pos < 0.15 or y_pos > 0.9:  # Common title positions
            confidence += 0.1
    
    # Sentence structure
    if stats.sentence_count > 1:
        confidence -= 0.2
    
    # Ending punctuation
    if text.endswith(('.', '!', '?')):  # Titles rarely end with these
        confidence -= 0.1
    
    # Final decision with adjusted threshold
    is_title = confidence > 0.35
    return is_title, min(1.0, max(0.0, confidence))

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
    """Check if text is a list item using multiple patterns.
    
    Args:
        text: Text to check
        
    Returns:
        bool: True if text matches list item patterns
    """
    if not text or not text.strip():
        return False
        
    text = text.strip()
    
    # Check for bullet points (including unicode bullets)
    if UNICODE_BULLETS_RE.match(text) or UNICODE_BULLETS_RE_0W.match(text):
        return True
        
    # Check for numbered lists (e.g., "1. Item", "a) Point", "i. Subpoint")
    if ENUMERATED_BULLETS_RE.match(text.split()[0]):
        # Additional check to avoid matching section numbers (e.g., "1.2.3")
        if len(text.split()) > 1:  # Must have text after the number
            first_word = text.split()[0]
            
            # Skip section numbers with multiple dots (e.g., "1.2.3 Section")
            if first_word.count('.') > 1:
                return False
                
            # Skip section numbers with multiple parts (e.g., "1.2 Section")
            if '.' in first_word and len(first_word) > 2:
                return False
                
            # Skip section numbers with closing parenthesis (e.g., "1.2) Section")
            if ')' in first_word or ']' in first_word:
                return False
                
            # Check if the text after the number starts with a capital letter (more likely a section)
            rest_of_text = text[len(first_word):].strip()
            if rest_of_text and rest_of_text[0].isupper():
                return False
                
            return True
    
    # Check for lettered lists (a, b, c or i, ii, iii)
    if re.match(r'^[a-zA-Z][.)]\s+', text):
        # Check if the text after the letter starts with a capital letter (more likely a section)
        rest_of_text = text[2:].strip()
        if rest_of_text and rest_of_text[0].isupper():
            return False
        return True
        
    return False

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
