import re

# Comprehensive list item patterns
LIST_PATTERNS = [
    # Bullet patterns (including various Unicode bullets)
    r'^\s*[\u2022\u2023\u25E6\u2043\u2219\u25AA\u25AB\u25B8\u2027\u2218*•·⁃-]\s+',
    
    # Numbered patterns
    r'^\s*\d+[\.:\)]\s+',  # 1. or 1) or 1:
    r'^\s*\(\d+\)\s+',     # (1)
    r'^\s*\[\d+\]\s+',     # [1]
    
    # Lettered patterns
    r'^\s*[a-zA-Z][\.:\)]\s+',  # a. or A) or a:
    r'^\s*\([a-zA-Z]\)\s+',     # (a)
    r'^\s*\[[a-zA-Z]\]\s+',     # [a]
    
    # Roman numeral patterns
    r'^\s*[ivxIVX]+[\.:\)]\s+',  # i. or IV) or ii:
    r'^\s*\([ivxIVX]+\)\s+',     # (i)
    r'^\s*\[[ivxIVX]+\]\s+',     # [i]
]

# Title patterns
TITLE_PATTERNS = [
    r'^(?:chapter|section|appendix)\s+\d+',
    r'^(?:table|figure|appendix)\s+\d+[-.:]\s*',
    r'^(?:table of contents|index|glossary|references|bibliography)$',
    r'^\d+\.\d+\s+[A-Z]'
]

# Header/Footer patterns
HEADER_FOOTER_PATTERNS = [
    r'copyright\s+\d{4}',
    r'confidential',
    r'all\s+rights\s+reserved',
    r'draft',
    r'private',
    r'internal\s+use\s+only',
    r'page\s+\d+\s+of\s+\d+'
]

# Footnote patterns
FOOTNOTE_PATTERNS = [
    r'^\s*\d+\s+',  # Numeric footnotes
    r'^\s*[*†‡§]\s+',  # Symbol footnotes
    r'^\s*[a-z]\)\s+',  # Letter footnotes
    r'^\s*\[\d+\]\s+'  # Bracketed numbers
]

# Compiled patterns for better performance
COMPILED_LIST_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in LIST_PATTERNS]
COMPILED_TITLE_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in TITLE_PATTERNS]
COMPILED_HEADER_FOOTER_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in HEADER_FOOTER_PATTERNS]
COMPILED_FOOTNOTE_PATTERNS = [re.compile(pattern) for pattern in FOOTNOTE_PATTERNS]

def matches_any_pattern(text: str, patterns: list) -> bool:
    """Check if text matches any of the compiled patterns."""
    return any(pattern.search(text) for pattern in patterns)

def is_list_item_start(text: str) -> bool:
    """Check if text starts with a list item marker."""
    return matches_any_pattern(text, COMPILED_LIST_PATTERNS)

def is_title_pattern(text: str) -> bool:
    """Check if text matches a title pattern."""
    return matches_any_pattern(text, COMPILED_TITLE_PATTERNS)

def is_header_footer(text: str) -> bool:
    """Check if text matches a header/footer pattern."""
    return matches_any_pattern(text, COMPILED_HEADER_FOOTER_PATTERNS)

def is_footnote(text: str) -> bool:
    """Check if text matches a footnote pattern."""
    return matches_any_pattern(text, COMPILED_FOOTNOTE_PATTERNS)
