from typing import List, Dict, Any, Tuple, Optional
import re
import numpy as np
from dataclasses import dataclass
from text_cleaners import TextCleaner

@dataclass
class BlockAnalysis:
    is_title: bool = False
    is_list_item: bool = False
    is_header_footer: bool = False
    is_page_number: bool = False
    is_footnote: bool = False
    is_annotation: bool = False
    confidence: float = 0.0
    metadata: Dict[str, Any] = None
    cross_page_refs: List[int] = None

class TextBlockAnalyzer:
    TITLE_PATTERNS = [
        r"^(?:chapter|section|appendix)\s+\d+",
        r"^(?:table|figure|appendix)\s+\d+[-.:]\s*",
        r"^(?:table of contents|index|glossary|references|bibliography)$",
        r"^\d+\.\d+\s+[A-Z]"
    ]
    
    LIST_MARKERS = [
        r"^\s*(?:\d+\.|\(\d+\)|\[?\d+\]|\w+\.|\(\w+\)|\[?\w+\])\s+",
        r"^\s*[•●○◆▪-]\s+",
        r"^\s*[A-Z]\.\s+",
        r"^\s*[IVXLCDMivxlcdm]+\.\s+"
    ]
    
    HEADER_FOOTER_MARKERS = [
        "copyright", "confidential", "all rights reserved",
        "draft", "private", "internal use", "page"
    ]

    FOOTNOTE_PATTERNS = [
        r"^\s*\d+\s+",  # Numeric footnotes
        r"^\s*[*†‡§]\s+",  # Symbol footnotes
        r"^\s*[a-z]\)\s+",  # Letter footnotes
        r"^\s*\[\d+\]\s+"  # Bracketed numbers
    ]

    ANNOTATION_MARKERS = [
        r"^\s*\[\w+:\s",  # [Note: ...
        r"^\s*\{\w+:\s",  # {Comment: ...
        r"^\s*<\w+>\s",   # <Annotation> ...
        r"^\s*\(\w+:\s"   # (Edit: ...
    ]

    def __init__(self):
        self.page_stats = {}
        self.document_stats = {}
        self.cross_page_elements = {
            "headers": [],
            "footers": [],
            "footnotes": [],
            "annotations": []
        }
        self.similarity_threshold = 0.85

    def _should_merge_blocks(self, block1: Dict[str, Any], block2: Dict[str, Any]) -> bool:
        """Check if two blocks should be merged based on their properties"""
        # Check vertical distance (should be close)
        y_diff = abs(block1["bbox"][3] - block2["bbox"][1])
        if y_diff > 15:  # Maximum 15 points gap
            return False
            
        # Check horizontal alignment
        x_overlap = (
            min(block1["bbox"][2], block2["bbox"][2]) -
            max(block1["bbox"][0], block2["bbox"][0])
        )
        
        # For list items, allow continuation with indentation
        if block1["element_type"] == "ListItem":
            if block2["bbox"][0] > block1["bbox"][0] + 10:  # Indented continuation
                return True
            
        # For other blocks, require horizontal overlap
        elif x_overlap < 0:
            return False
            
        # Check font consistency
        if block1["style_info"]["dominant_font"] != block2["style_info"]["dominant_font"]:
            return False
            
        # Check if blocks have same element type
        if block1["element_type"] != block2["element_type"] and \
           not (block1["element_type"] == "ListItem" and block2["element_type"] == "NarrativeText"):
            return False
            
        # Check if second block starts with lowercase (likely continuation)
        if block2["text"].strip() and not block2["text"].strip()[0].isupper():
            return True
            
        # Check if first block ends with hyphen or incomplete sentence
        if block1["text"].strip().endswith(("-", ",", ";", ":")) or \
           not block1["text"].strip().endswith("."):
            return True
            
        # Check if blocks are list items with sequential numbers
        if block1["element_type"] == "ListItem":
            num1 = re.search(r'^\d+', block1["text"])
            num2 = re.search(r'^\d+', block2["text"])
            if num1 and num2 and int(num2.group()) == int(num1.group()) + 1:
                return True
            
        return False

    def _merge_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge text blocks that are likely part of the same content"""
        if not blocks:
            return blocks
            
        merged = []
        current = blocks[0]
        
        for next_block in blocks[1:]:
            if self._should_merge_blocks(current, next_block):
                # Merge text
                if current["text"].strip().endswith("-"):
                    current["text"] = current["text"].strip()[:-1] + next_block["text"].strip()
                else:
                    current["text"] = current["text"].strip() + " " + next_block["text"].strip()
                    
                # Update bbox to encompass both blocks
                current["bbox"] = (
                    min(current["bbox"][0], next_block["bbox"][0]),
                    min(current["bbox"][1], next_block["bbox"][1]),
                    max(current["bbox"][2], next_block["bbox"][2]),
                    max(current["bbox"][3], next_block["bbox"][3])
                )
            else:
                merged.append(current)
                current = next_block
        
        merged.append(current)
        return merged

    def _clean_text(self, text: str) -> str:
        """Clean text by removing unwanted characters and normalizing whitespace"""
        # Remove bullet points and other list markers
        text = re.sub(r'^[•●○◆▪-]\s*', '', text)
        
        # Remove newlines and extra whitespace
        text = re.sub(r'\\n|\s*\n\s*', ' ', text)
        
        # Remove hyphenation at line breaks
        text = re.sub(r'-\s+(?=[a-z])', '', text)
        
        # Remove dots used for table of contents
        text = re.sub(r'\.\s*\.\s*\.+\s*$', '', text)
        
        # Clean up brackets and references
        text = re.sub(r'\[\d+\]\s*', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text

    def analyze_block_structure(self, text: str, font_info: Dict[str, Any], position_info: Dict[str, float], page_num: int) -> BlockAnalysis:
        """Analyze the structure and type of a text block"""
        analysis = BlockAnalysis()
        text = text.strip()
        
        # Check for page number
        if (len(text) <= 3 and text.isdigit() and 
            (position_info["relative_y"] < 0.1 or position_info["relative_y"] > 0.9)):
            analysis.is_page_number = True
            analysis.confidence = 0.9
            return analysis
        
        # Check for title using multiple factors
        title_score = 0
        # Font size significantly larger
        if font_info["avg_font_size"] > self.document_stats.get("median_font_size", 0) * 1.2:
            title_score += 2
        # Near top of page
        if position_info["relative_y"] > 0.8:
            title_score += 1
        # Short text (likely a heading)
        if len(text.split()) <= 10:
            title_score += 1
        # No ending punctuation (typical for titles)
        if not text.rstrip().endswith(('.', ':', ';', '?', '!')):
            title_score += 1
        # Bold text
        if font_info.get("is_bold", False):
            title_score += 1
        # Title pattern match
        if any(re.match(pattern, text.lower()) for pattern in self.TITLE_PATTERNS):
            title_score += 2
            
        if title_score >= 3:
            analysis.is_title = True
            analysis.confidence = min(0.5 + (title_score * 0.1), 1.0)
            return analysis
            
        # Check for list items using multiple patterns
        list_score = 0
        # Pattern match
        if any(re.match(pattern, text) for pattern in self.LIST_MARKERS):
            list_score += 2
        # Indentation
        if position_info["indent_ratio"] > 0.05:
            list_score += 1
        # Short text
        if len(text) < 200:
            list_score += 1
        # Few lines
        if font_info.get("line_count", 1) <= 3:
            list_score += 1
            
        if list_score >= 3:
            analysis.is_list_item = True
            analysis.confidence = min(0.5 + (list_score * 0.1), 1.0)
            return analysis
            
        # Check for footnotes
        footnote_score = 0
        # Pattern match
        if any(re.match(pattern, text) for pattern in self.FOOTNOTE_PATTERNS):
            footnote_score += 2
        # Position near bottom
        if position_info["relative_y"] < 0.2:
            footnote_score += 1
        # Smaller font
        if font_info["avg_font_size"] < self.document_stats.get("median_font_size", 0):
            footnote_score += 1
            
        if footnote_score >= 3:
            analysis.is_footnote = True
            analysis.confidence = min(0.5 + (footnote_score * 0.1), 1.0)
            return analysis
        
        # Check for header/footer
        header_score = 0
        # Position at top/bottom
        if position_info["relative_y"] < 0.1 or position_info["relative_y"] > 0.9:
            header_score += 2
        # Contains header/footer keywords
        if any(marker in text.lower() for marker in self.HEADER_FOOTER_MARKERS):
            header_score += 2
        # Smaller font
        if font_info["avg_font_size"] < self.document_stats.get("median_font_size", 0):
            header_score += 1
        # Short text
        if len(text) < 100:
            header_score += 1
            
        if header_score >= 3:
            analysis.is_header_footer = True
            analysis.confidence = min(0.5 + (header_score * 0.1), 1.0)
            # Check for recurring elements
            similar_blocks = self._find_similar_blocks(text, page_num)
            if similar_blocks:
                analysis.cross_page_refs = similar_blocks
            return analysis
            
        return analysis

    def _analyze_title_likelihood(self, text: str, style_info: Dict[str, Any], 
                                metrics: Dict[str, Any]) -> float:
        """Analyze how likely a text block is to be a title"""
        score = 0.0
        text_lower = text.strip().lower()

        # Pattern matching
        if any(re.match(pattern, text_lower) for pattern in self.TITLE_PATTERNS):
            score = max(score, 0.8)

        # Style analysis
        if style_info.get("is_bold", False):
            score += 0.2
        if style_info.get("avg_font_size", 0) > style_info.get("body_font_size", 12) * 1.2:
            score += 0.3

        # Content analysis
        if metrics["line_count"] <= 2 and metrics["word_count"] <= 20:
            score += 0.1
        if metrics["capitalization_ratio"] > 0.6:
            score += 0.1

        return min(score, 1.0)

    def _analyze_list_likelihood(self, text: str, style_info: Dict[str, Any],
                               position_info: Dict[str, Any]) -> float:
        """Analyze how likely a text block is to be a list item"""
        score = 0.0
        text = text.strip()

        # Pattern matching
        if any(re.match(pattern, text) for pattern in self.LIST_MARKERS):
            score = max(score, 0.8)

        # Extract potential ordered bullets
        a, b, c = TextCleaner.extract_ordered_bullets(text)
        if a is not None:
            score = max(score, 0.7)

        # Position analysis
        indent_ratio = position_info.get("indent_ratio", 0)
        if 0.05 <= indent_ratio <= 0.15:  # Typical list indentation
            score += 0.2

        # Content analysis
        lines = text.split("\n")
        if len(lines) == 1:  # List items are typically single lines
            score += 0.1

        return min(score, 1.0)

    def _analyze_header_footer_likelihood(self, text: str, position_info: Dict[str, Any],
                                        metrics: Dict[str, Any], page_number: int) -> Tuple[float, bool]:
        """Analyze how likely a text block is to be a header/footer"""
        score = 0.0
        text_lower = text.lower()

        # Marker matching
        if any(marker in text_lower for marker in self.HEADER_FOOTER_MARKERS):
            score = max(score, 0.7)

        # Position analysis
        relative_y = position_info.get("relative_y", 0.5)
        if relative_y < 0.1 or relative_y > 0.9:
            score += 0.2

        # Content analysis
        if metrics["line_count"] == 1 and metrics["word_count"] < 10:
            score += 0.1
        if metrics["avg_line_length"] < 50:  # Headers/footers typically short
            score += 0.1

        # Check if this text appears in similar positions across pages
        is_recurring = False
        if position_info.get("relative_y", 0.5) < 0.1:  # Header zone
            similar_blocks = [b for b in self.cross_page_elements["headers"]
                           if self._text_similarity(b["text"], text) > self.similarity_threshold]
            if similar_blocks:
                score += 0.2
                is_recurring = True
            self.cross_page_elements["headers"].append({
                "text": text,
                "page": page_number,
                "position": position_info
            })
        elif position_info.get("relative_y", 0.5) > 0.9:  # Footer zone
            similar_blocks = [b for b in self.cross_page_elements["footers"]
                           if self._text_similarity(b["text"], text) > self.similarity_threshold]
            if similar_blocks:
                score += 0.2
                is_recurring = True
            self.cross_page_elements["footers"].append({
                "text": text,
                "page": page_number,
                "position": position_info
            })
            
        return min(score, 1.0), is_recurring

    def _analyze_footnote_likelihood(self, text: str, position_info: Dict[str, Any],
                                  style_info: Dict[str, Any]) -> float:
        """Analyze how likely a text block is to be a footnote"""
        score = 0.0
        text = text.strip()

        # Pattern matching
        if any(re.match(pattern, text) for pattern in self.FOOTNOTE_PATTERNS):
            score = max(score, 0.7)

        # Position analysis
        relative_y = position_info.get("relative_y", 0.5)
        if relative_y > 0.8:  # Footnotes typically at bottom
            score += 0.2

        # Style analysis
        if style_info.get("avg_font_size", 12) < style_info.get("body_font_size", 12):
            score += 0.1

        # Content analysis
        if len(text.split()) > 3:  # Must be more than just a reference number
            score += 0.1

        return min(score, 1.0)

    def _analyze_annotation_likelihood(self, text: str, style_info: Dict[str, Any]) -> float:
        """Analyze how likely a text block is to be an annotation"""
        score = 0.0
        text = text.strip()

        # Pattern matching
        if any(re.match(pattern, text) for pattern in self.ANNOTATION_MARKERS):
            score = max(score, 0.7)

        # Style analysis
        if style_info.get("is_italic", False):
            score += 0.2
        if style_info.get("avg_font_size", 12) < style_info.get("body_font_size", 12):
            score += 0.1

        # Content analysis
        if text.startswith("[") and text.endswith("]"):
            score += 0.1
        if any(marker in text.lower() for marker in ["note", "comment", "edit", "annotation"]):
            score += 0.2

        return min(score, 1.0)

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text blocks using multiple metrics"""
        if not text1 or not text2:
            return 0.0

        # Normalize texts
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()

        # 1. Exact match shortcut
        if t1 == t2:
            return 1.0

        # 2. Length-based pre-filtering
        len_ratio = min(len(t1), len(t2)) / max(len(t1), len(t2))
        if len_ratio < 0.5:  # Too different in length
            return 0.0

        # 3. Token-based similarity (improved Jaccard)
        tokens1 = set(t1.split())
        tokens2 = set(t2.split())
        
        # Remove common stop words that might inflate similarity
        stop_words = {"a", "an", "the", "in", "on", "at", "of", "to", "for", "by"}
        tokens1 = tokens1 - stop_words
        tokens2 = tokens2 - stop_words
        
        if not tokens1 or not tokens2:  # After stop word removal
            return 0.0

        jaccard = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))

        # 4. Character n-gram similarity
        def get_ngrams(text: str, n: int) -> set:
            return {text[i:i+n] for i in range(len(text)-n+1)}

        # Use both trigrams and bigrams
        trigrams1 = get_ngrams(t1, 3)
        trigrams2 = get_ngrams(t2, 3)
        bigrams1 = get_ngrams(t1, 2)
        bigrams2 = get_ngrams(t2, 2)

        if trigrams1 and trigrams2:
            trigram_sim = len(trigrams1.intersection(trigrams2)) / len(trigrams1.union(trigrams2))
        else:
            trigram_sim = 0.0

        if bigrams1 and bigrams2:
            bigram_sim = len(bigrams1.intersection(bigrams2)) / len(bigrams1.union(bigrams2))
        else:
            bigram_sim = 0.0

        # 5. Sequence similarity using longest common subsequence
        def lcs_length(s1: str, s2: str) -> int:
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m):
                for j in range(n):
                    if s1[i] == s2[j]:
                        dp[i + 1][j + 1] = dp[i][j] + 1
                    else:
                        dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
            return dp[m][n]

        sequence_sim = 2 * lcs_length(t1, t2) / (len(t1) + len(t2)) if t1 and t2 else 0.0

        # 6. Position-aware similarity for headers/footers
        position_bonus = 0.0
        if all(x.isdigit() or x.isspace() for x in t1) and all(x.isdigit() or x.isspace() for x in t2):
            # For page numbers, be more lenient
            position_bonus = 0.3

        # Combine all metrics with weights
        similarity = (
            0.35 * jaccard +          # Token-based similarity
            0.25 * trigram_sim +      # Character trigram similarity
            0.15 * bigram_sim +       # Character bigram similarity
            0.25 * sequence_sim +     # Sequence similarity
            position_bonus            # Position-based bonus
        )

    def _find_similar_blocks(self, text: str, current_page: int) -> List[int]:
        """Find pages with similar text blocks"""
        similar_pages = []
        
        # Get all blocks from cross-page elements
        all_blocks = (
            self.cross_page_elements["headers"] +
            self.cross_page_elements["footers"] +
            self.cross_page_elements["footnotes"]
        )
        
        # Look for similar blocks on other pages
        for block in all_blocks:
            if block["page"] != current_page:
                similarity = self._text_similarity(text, block["text"])
                if similarity > self.similarity_threshold:
                    similar_pages.append(block["page"])
        
        return similar_pages
        
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text blocks using multiple metrics"""
        # Normalize texts
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()
        
        # 1. Exact match shortcut
        if t1 == t2:
            return 1.0
            
        # 2. Length-based pre-filtering
        len_ratio = min(len(t1), len(t2)) / max(len(t1), len(t2))
        if len_ratio < 0.5:  # Too different in length
            return 0.0
            
        # 3. Token-based similarity (improved Jaccard)
        tokens1 = set(t1.split())
        tokens2 = set(t2.split())
        
        # Remove common stop words that might inflate similarity
        stop_words = {"a", "an", "the", "in", "on", "at", "of", "to", "for", "by"}
        tokens1 = tokens1 - stop_words
        tokens2 = tokens2 - stop_words
        
        if not tokens1 or not tokens2:  # After stop word removal
            return 0.0
            
        jaccard = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))
        return jaccard
