import sys
from typing import List
if sys.version_info < (3, 8):
    from typing_extensions import Final
else:
    from typing import Final
import re

US_PHONE_NUMBERS_PATTERN = (r"(?:\+?(\d{1,3}))?[-. (]*(\d{3})?[-. )]*(\d{3})[-. ]*(\d{4})(?: *x(\d+))?\s*$")
US_PHONE_NUMBERS_RE = re.compile(US_PHONE_NUMBERS_PATTERN)
US_CITY_STATE_ZIP_PATTERN = (
    r"(?i)\b(?:[A-Z][a-z.-]{1,15}[ ]?){1,5},\s?"
    r"(?:{Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida"
    r"|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland"
    r"|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|"
    r"New[ ]Hampshire|New[ ]Jersey|New[ ]Mexico|New[ ]York|North[ ]Carolina|North[ ]Dakota"
    r"|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode[ ]Island|South[ ]Carolina|South[ ]Dakota"
    r"|Tennessee|Texas|Utah|Vermont|Virginia|Washington|West[ ]Virginia|Wisconsin|Wyoming}"
    r"|{AL|AK|AS|AZ|AR|CA|CO|CT|DE|DC|FM|FL|GA|GU|HI|ID|IL|IN|IA|KS|KY|LA|ME|MH|MD|MA|MI|MN"
    r"|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|MP|OH|OK|OR|PW|PA|PR|RI|SC|SD|TN|TX|UT|VT|VI|VA|"
    r"WA|WV|WI|WY})(, |\s)?(?:\b\d{5}(?:-\d{4})?\b)"
)
US_CITY_STATE_ZIP_RE = re.compile(US_CITY_STATE_ZIP_PATTERN)
UNICODE_BULLETS: Final[List[str]] = [
    "\u0095",
    "\u2022",
    "\u2023",
    "\u2043",
    "\u3164",
    "\u204C",
    "\u204D",
    "\u2219",
    "\u25CB",
    "\u25CF",
    "\u25D8",
    "\u25E6",
    "\u2619",
    "\u2765",
    "\u2767",
    "\u29BE",
    "\u29BF",
    "\u002D",
    "",
    r"\*",
    "\x95",
    "·",
]
BULLETS_PATTERN = "|".join(UNICODE_BULLETS)
UNICODE_BULLETS_RE = re.compile(f"(?:{BULLETS_PATTERN})(?!{BULLETS_PATTERN})")
UNICODE_BULLETS_RE_0W = re.compile(f"(?={BULLETS_PATTERN})(?<!{BULLETS_PATTERN})")
E_BULLET_PATTERN = re.compile(r"^e(?=\s)", re.MULTILINE)
REFERENCE_PATTERN = r"\[(?:[\d]+|[a-z]|[ivxlcdm])\]"
REFERENCE_PATTERN_RE = re.compile(REFERENCE_PATTERN)
ENUMERATED_BULLETS_RE = re.compile(r"(?:(?:\d{1,3}|[a-z][A-Z])\.?){1,3}")
EMAIL_HEAD_PATTERN = (r"(MIME-Version: 1.0(.*)?\n)?Date:.*\nMessage-ID:.*\nSubject:.*\nFrom:.*\nTo:.*")
EMAIL_HEAD_RE = re.compile(EMAIL_HEAD_PATTERN)
PARAGRAPH_PATTERN = r"\s*\n\s*"
PARAGRAPH_PATTERN_RE = re.compile(f"((?:{BULLETS_PATTERN})|{PARAGRAPH_PATTERN})(?!{BULLETS_PATTERN}|$)",)
DOUBLE_PARAGRAPH_PATTERN_RE = re.compile("(" + PARAGRAPH_PATTERN + "){2}")
LINE_BREAK = r"(?<=\n)"
LINE_BREAK_RE = re.compile(LINE_BREAK)
ONE_LINE_BREAK_PARAGRAPH_PATTERN = r"^(?:(?!\.\s*$).)*$"
ONE_LINE_BREAK_PARAGRAPH_PATTERN_RE = re.compile(ONE_LINE_BREAK_PARAGRAPH_PATTERN)
IP_ADDRESS_PATTERN = (
    r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)){3}",
    "[a-z0-9]{4}::[a-z0-9]{4}:[a-z0-9]{4}:[a-z0-9]{4}:[a-z0-9]{4}%?[0-9]*",
)
IP_ADDRESS_PATTERN_RE = re.compile(f"({'|'.join(IP_ADDRESS_PATTERN)})")
IP_ADDRESS_NAME_PATTERN = r"[a-zA-Z0-9-]*\.[a-zA-Z]*\.[a-zA-Z]*"
MAPI_ID_PATTERN = r"[0-9]*\.[0-9]*\.[0-9]*\.[0-9]*;"
EMAIL_DATETIMETZ_PATTERN = (r"[A-Za-z]{3},\s\d{1,2}\s[A-Za-z]{3}\s\d{4}\s\d{2}:\d{2}:\d{2}\s[+-]\d{4}")
EMAIL_DATETIMETZ_PATTERN_RE = re.compile(EMAIL_DATETIMETZ_PATTERN)
EMAIL_ADDRESS_PATTERN = r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+"
EMAIL_ADDRESS_PATTERN_RE = re.compile(EMAIL_ADDRESS_PATTERN)
ENDS_IN_PUNCT_PATTERN = r"[^\w\s]\Z"
ENDS_IN_PUNCT_RE = re.compile(ENDS_IN_PUNCT_PATTERN)
LIST_OF_DICTS_PATTERN = r"\A\s*\[\s*{?"
JSON_PATTERN = r"(?s)\{(?=.*:).*?(?:\}|$)|\[(?s:.*?)\](?:$|,|\])"
VALID_JSON_CHARACTERS = r"[,:{}\[\]0-9.\-+Eaeflnr-u \n\r\t]"
IMAGE_URL_PATTERN = (
    r"(?i)https?://"
    r"(?:[a-z0-9$_@.&+!*\\(\\),%-])+"
    r"(?:/[a-z0-9$_@.&+!*\\(\\),%-]*)*"
    r"\.(?:jpg|jpeg|png|gif|bmp|heic)"
)
NUMBERED_LIST_PATTERN = r"^\d+(\.|\))\s(.+)"
NUMBERED_LIST_RE = re.compile(NUMBERED_LIST_PATTERN)

# Caption patterns
CAPTION_PATTERNS = [
    r'^(?:Figure|Fig\.?|Table|Chart|Diagram|Image|Photo|Illustration|Fig\.?\s*\d+)[:.]?\s*',
    r'^\s*\(?\s*(?:Figure|Fig\.?|Table|Chart|Diagram|Image|Photo|Illustration)\s*\d+\s*\)?\s*[:.-]?\s*',
    r'^\s*\[?\s*(?:Figure|Fig\.?|Table|Chart|Diagram|Image|Photo|Illustration)\s*\d+\s*\]?\s*[:.-]?\s*',
    r'^\s*\d+\s*[:.-]?\s*[A-Z]',  # Number followed by capital letter (e.g., "1. A diagram showing...")
]

# Compiled regex for caption patterns
CAPTION_RE = [re.compile(pattern, re.IGNORECASE) for pattern in CAPTION_PATTERNS]

# Common caption suffixes that might indicate the end of a caption
CAPTION_END_PATTERNS = [
    r'\s*[.?!]?\s*$',  # Ends with optional punctuation
    r'\s*\[\d+\]\s*$',  # Ends with citation [1]
    r'\s*\([^)]*\)\s*$',  # Ends with parenthetical note
]

TITLE_PATTERNS = [
    r"^(?:chapter|section|appendix)\s+\d+",
    r"^(?:table|figure|appendix)\s+\d+[-.:]\s*",
    r"^(?:table of contents|index|glossary|references|bibliography)$",
    r"^\d+\.\d+\s+[A-Z]",
    r"^ITEM\s+\d+[A-Z]?\.",
    r"^[A-Z\s]{10,}$",
]
TITLE_RE = re.compile("|".join(TITLE_PATTERNS), re.IGNORECASE)

PAGE_NUMBER_PATTERNS = [
    r'^\d+$',
    r'^-\s*\d+\s*-$',
    r'^\[\d+\]$',
    r'^Page\s+\d+$',
    r'^Page\s+\d+\s+of\s+\d+$',
    r'^\d+\s*/\s*\d+$',
    r'^p\.?\s*\d+$',
]
PAGE_NUMBER_RE = re.compile("|".join(PAGE_NUMBER_PATTERNS), re.IGNORECASE)

FOOTER_PATTERNS = [
    r'confidential',
    r'all\s+rights\s+reserved',
    r'internal\s+use\s+only',
    r'copyright\s+\d{4}',
    r'page\s+\d+\s+of\s+\d+',
    r'^\d+$',
    r'^-\s*\d+\s*-$',
    r'^\[\d+\]$',
    r'^Page\s+\d+$',
    r'^Page\s+\d+\s+of\s+\d+$',
]
FOOTER_RE = re.compile("|".join(FOOTER_PATTERNS), re.IGNORECASE)
