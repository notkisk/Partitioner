import nltk
import re
from typing import Optional, List, Set

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# Constants
POS_VERB_TAGS = ["VB", "VBG", "VBD", "VBN", "VBP", "VBZ"]
ENGLISH_WORD_SPLIT_RE = re.compile(r"[\s\-,.!?_\/]+")
NON_LOWERCASE_ALPHA_RE = re.compile(r"[^a-z]")

def remove_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text)

def sentence_count(text: str, min_word_length: Optional[int] = None) -> int:
    sentences = nltk.sent_tokenize(text)
    count = 0
    for sentence in sentences:
        cleaned_sentence = remove_punctuation(sentence)
        words = [word for word in nltk.word_tokenize(cleaned_sentence) if word.isalnum()]
        if min_word_length is not None:
            if len(words) >= min_word_length:
                count += 1
        else:
            count += 1
    return count

def under_non_alpha_ratio(text: str, threshold: float = 0.5) -> bool:
    if not text:
        return False
    
    stripped_chars = [char for char in text if char.strip()]
    if not stripped_chars:
        return False

    alpha_count = sum(1 for char in stripped_chars if char.isalpha())
    total_stripped_count = len(stripped_chars)

    if total_stripped_count == 0:
        return False

    ratio = alpha_count / total_stripped_count
    return ratio < threshold

def exceeds_cap_ratio(text: str, sentence_min_len: int = 3, threshold: float = 0.5) -> bool:
    if sentence_count(text, min_word_length=sentence_min_len) > 1:
        return False

    if text.isupper():
        return True

    tokens = [tk for tk in nltk.word_tokenize(text) if tk.isalpha()]
    if not tokens:
        return True

    capitalized_count = sum(1 for word in tokens if word.istitle() or word.isupper())
    ratio = capitalized_count / len(tokens)
    return ratio > threshold

def contains_verb(text: str) -> bool:
    if not text.strip():
        return False
    
    text_to_tag = text.lower() if text.isupper() else text
    tokens = nltk.word_tokenize(text_to_tag)
    if not tokens:
        return False
    
    pos_tags = nltk.pos_tag(tokens)
    return any(tag in POS_VERB_TAGS for _, tag in pos_tags)

# Load a basic set of common English words
# In practice, this should be loaded from a comprehensive word list file
ENGLISH_WORDS: Set[str] = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    # Add more common words as needed
}

def contains_english_word(text: str) -> bool:
    if not text.strip():
        return False
    
    lower_text = text.lower()
    potential_words = ENGLISH_WORD_SPLIT_RE.split(lower_text)
    
    for word in potential_words:
        cleaned_word = NON_LOWERCASE_ALPHA_RE.sub("", word)
        if len(cleaned_word) > 1 and cleaned_word in ENGLISH_WORDS:
            return True
    return False
