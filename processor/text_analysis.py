import logging
import nltk
from typing import List, Tuple, Dict, Any, Optional
import re
from nltk import pos_tag, sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from dataclasses import dataclass

logger = logging.getLogger(__name__)

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

    if text.isupper():
        text = text.lower()
    pos_tags = pos_tag(word_tokenize(text))
    return any(tag in POS_VERB_TAGS for _, tag in pos_tags)

def sentence_count(text: str, min_length: int = None) -> int:

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

def is_possible_title(text, style_info=None, coordinates=None):
    if not text or len(text) < 2:
        return False, 0.0, None
    text = text.strip()
    from .text_patterns import TITLE_PATTERNS
    import re
    stats = get_text_stats(text)
    confidence = 0.0
    title_level = None
    
    # bonus for concise titles
    if stats.word_count <= 8:
        confidence += 0.1
    # penalty for ending in punctuation
    if text.endswith(('.', ':', ';', '!', '?')):
        confidence -= 0.1
    # noun ratio boost
    pos_tags = pos_tag(word_tokenize(text))
    noun_count = sum(1 for _, tag in pos_tags if tag.startswith('NN'))
    if stats.word_count > 0 and noun_count / stats.word_count > 0.5:
        confidence += 0.1
    
    if style_info:
        if style_info.get('is_bold'):
            confidence += 0.15
        if style_info.get('size', 0) > 12.0:
            confidence += 0.1
        if stats.is_all_caps:
            confidence += 0.1
    
    if coordinates and 'relative_x' in coordinates and abs(coordinates['relative_x'] - 0.5) < 0.2:
        confidence += 0.05
    section_match = re.match(r'^(\d+(?:\.\d+)+)\s+', text)
    if section_match:
        confidence += 0.4
        parts = section_match.group(1).split('.')
        title_level = len(parts)
    text_lower = text.lower()
    for pattern in TITLE_PATTERNS:
        if re.match(pattern, text_lower):
            confidence += 0.3
            break
    if stats.word_count > 15:
        confidence -= 0.2
    if stats.has_verb:
        confidence -= 0.1
    if stats.cap_ratio > 0.7:
        confidence += 0.1
    if stats.sentence_count > 2:
        confidence -= 0.2
    # extra boost for bold large font
    if style_info and style_info.get('is_bold') and style_info.get('size', 0) >= stats.avg_word_length * 1.2:
        confidence += 0.05
    if confidence > 0.4:
        return True, min(1.0, max(0.0, confidence)), title_level
    return False, 0.0, None

def is_possible_narrative(
    text: str,
    cap_threshold: float = 0.5,
    alpha_threshold: float = 0.5
) -> bool:

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
