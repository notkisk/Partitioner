import logging
import re
from dataclasses import asdict
from typing import Dict, Any, List, Tuple
import math

logger = logging.getLogger(__name__)
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
    text,
    bbox,
    style_info,
    page_info,
    page_number,
    context,
    elements_before=None,
    elements_after=None
):
    from .data_models import CoordinatesMetadata, ElementMetadata, ElementType, Element
    from dataclasses import asdict
    coords = CoordinatesMetadata(
        x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3],
        page_width=page_info['width'],
        page_height=page_info['height']
    )
    metadata = ElementMetadata(
        style_info=style_info,
        coordinates=coords
    )
    style_info['median_font_size'] = context.get('median_font_size', 12)
    element_type = ElementType.TEXT
    confidence = 0.7
    text = text.strip()
    text = ' '.join(text.splitlines())
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
    # early caption detection override
    if elements_before:
        try:
            is_caption, caption_confidence = is_likely_caption(
                text,
                bbox,
                style_info,
                page_info,
                elements_before,
                elements_after or []
            )
            if is_caption and caption_confidence > 0.7:
                lower_text = text.lower()
                if lower_text.startswith('figure') or lower_text.startswith('table'):
                    try:
                        is_caption, cap_conf = is_likely_caption(
                            text, bbox, style_info, page_info,
                            elements_before or [], elements_after or []
                        )
                        if is_caption and cap_conf > 0.7:
                            if lower_text.startswith('figure'):
                                element_type = ElementType.FIGURE_CAPTION
                                confidence = min(0.95, cap_conf * 1.1)
                            else:
                                element_type = ElementType.TABLE_CAPTION
                                confidence = min(0.9, cap_conf)
                            metadata.confidence = confidence
                            return Element(
                                text=text,
                                element_type=element_type,
                                bbox=bbox,
                                page_number=page_number,
                                metadata=metadata
                            )
                    except:
                        pass
                if lower_text.startswith('figure'):
                    element_type = ElementType.FIGURE_CAPTION
                    confidence = min(0.95, caption_confidence * 1.1)
                elif lower_text.startswith('table'):
                    element_type = ElementType.TABLE_CAPTION
                    confidence = min(0.9, caption_confidence)
                else:
                    element_type = ElementType.CAPTION
                    confidence = caption_confidence
                metadata.confidence = confidence
                
                # ensure caption is near an image or table
                max_gap = style_info.get('median_font_size', 12) * 2
                for el in (elements_before or []) + (elements_after or []):
                    if el.page_number == page_number and el.element_type in (ElementType.FIGURE, ElementType.TABLE_BODY):
                        top, bottom = el.bbox[3], el.bbox[1]
                        dist = min(abs(top - bbox[1]), abs(bbox[3] - bottom))
                        if dist <= max_gap and bbox_overlap(el.bbox, bbox) > 0:
                            return Element(
                                text=text,
                                element_type=element_type,
                                bbox=bbox,
                                page_number=page_number,
                                metadata=metadata
                            )
        except Exception:
            pass
    is_title, title_confidence, title_level = is_possible_title(text, style_info, asdict(coords))
    if is_title:
        element_type = ElementType.TITLE
        confidence = max(0.7, title_confidence)
        if title_level is not None:
            confidence = min(1.0, confidence + (0.1 * (4 - title_level)))
        if confidence > 0.85:
            metadata.title_level = title_level
        metadata.confidence = confidence
        return Element(
            text=text,
            element_type=element_type,
            bbox=bbox,
            page_number=page_number,
            metadata=metadata
        )
    # refined fallback for missing titles: concise, single sentence, no verb, noun-rich, uppercase heavy, no trailing punctuation, larger font
    stats = get_text_stats(text)
    median_fs = style_info.get('median_font_size', 12)
    font_size = style_info.get('font_size', median_fs)
    # spatial isolation: compute gaps above and below
    page_h = coords.page_height
    # find highest bottom of blocks above current
    above_y = max((el.bbox[3] for el in (elements_before or []) if el.page_number == page_number and el.bbox[3] < bbox[1]), default=0)
    below_y = min((el.bbox[1] for el in (elements_after or []) if el.page_number == page_number and el.bbox[1] > bbox[3]), default=page_h)
    gap_above = bbox[1] - above_y
    gap_below = below_y - bbox[3]
    # relaxed spatial isolation and noun-rich check for refined title fallback
    is_isolated = (gap_above > median_fs * 1.5 and gap_below > median_fs * 1.5) or (gap_above > median_fs * 2 or gap_below > median_fs * 2)
    if (
        stats.sentence_count == 1 and
        not stats.has_verb and
        stats.word_count <= 8 and
        font_size > median_fs * 1.1 and
        stats.cap_ratio >= 0.6 and
        stats.noun_ratio > 0.5 and
        not text.endswith(('.', ':', ';', '!', '?')) and
        is_isolated
    ):
        element_type = ElementType.TITLE
        confidence = 0.85
        metadata.confidence = confidence
        return Element(
            text=text,
            element_type=element_type,
            bbox=bbox,
            page_number=page_number,
            metadata=metadata
        )
    # skip list detection for bold or large-font (likely titles)
    if element_type == ElementType.TEXT:
        median_fs = style_info.get('median_font_size', style_info.get('font_size', 12))
        fs = style_info.get('font_size', median_fs)
        if fs <= median_fs * 1.1 and not style_info.get('is_bold', False):
            if is_list_item(text):
                element_type = ElementType.LIST_ITEM
                confidence = 0.9
                metadata.confidence = confidence
                return Element(text=text, element_type=element_type, bbox=bbox, page_number=page_number, metadata=metadata)
    if element_type == ElementType.TEXT:
        if '=' in text and ('\\' in text or '$' in text):
            element_type = ElementType.BLOCK_EQUATION
            confidence = 0.9
            metadata.confidence = confidence
            return Element(
                text=text,
                element_type=element_type,
                bbox=bbox,
                page_number=page_number,
                metadata=metadata
            )
    # additional title fallback for bold/large/isolated single-line text
    stats = get_text_stats(text)
    median_fs = style_info.get('median_font_size', style_info.get('font_size', 12))
    if (
        element_type == ElementType.TEXT and
        stats.sentence_count == 1 and
        not stats.has_verb and
        stats.word_count <= 8 and
        style_info.get('font_size', median_fs) > median_fs * 1.1 and
        style_info.get('is_bold', False)
    ):
        element_type = ElementType.TITLE
        metadata.confidence = max(metadata.confidence, 0.75)
        return Element(text=text, element_type=element_type, bbox=bbox, page_number=page_number, metadata=metadata)
    metadata.confidence = confidence
    return Element(text=text, element_type=element_type, bbox=bbox, page_number=page_number, metadata=metadata)

def bbox_overlap(bbox1, bbox2):
    x0 = max(bbox1[0], bbox2[0])
    y0 = max(bbox1[1], bbox2[1])
    x1 = min(bbox1[2], bbox2[2])
    y1 = min(bbox1[3], bbox2[3])
    if x0 >= x1 or y0 >= y1:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    return inter / (area1 + area2 - inter)

def _is_likely_title(text: str, style_info: Dict[str, Any], median_font: float) -> Tuple[bool, float]:

    text = text.strip()
    
    if len(text) < 2:
        return False, 0.0
        
    confidence = 0.0
    
    is_all_uppercase = text.isupper()
    is_title_case = text == text.title() and not text.isupper()
    
    font_size = style_info.get('font_size', median_font)
    if font_size > median_font * 1.2:
        confidence += 0.3
    
    if style_info.get('is_bold', False):
        confidence += 0.2
    if style_info.get('is_italic', False):
        confidence += 0.1
        
    if is_all_uppercase:
        confidence += 0.15
    elif is_title_case:
        confidence += 0.1
        
    word_count = len(text.split())
    if word_count <= 15:
        confidence += 0.1
    elif word_count > 25:
        return False, 0.0
        
    sentence_count = len(re.split(r'[.!?]+', text))
    if sentence_count > 2:
        return False, 0.0
    
    if text.endswith(('.', ':', ';')):
        confidence -= 0.2
    
    if re.match(r'^\d+\.?\d*\s+\w+', text):
        confidence += 0.3
    elif re.match(r'^[A-Z]\.\s+\w+', text):
        confidence += 0.25
        
    title_keywords = r'^(chapter|section|appendix|introduction|conclusion|abstract|summary)\s+\d*'
    if re.match(title_keywords, text.lower()):
        confidence += 0.25
        
    if contains_verb(text):
        confidence -= 0.1
    
    y_pos = style_info.get('y_position', 0.5)
    if y_pos > 0.8 or y_pos < 0.1:
        confidence -= 0.15
        
    is_title = confidence > 0.4
    
    return is_title, min(1.0, max(0.0, confidence))

def _is_header_footer(text: str, coords: CoordinatesMetadata, page_info: Dict[str, float]) -> bool:

    margin_threshold = 0.1
    if coords.relative_y > margin_threshold and coords.relative_y < (1 - margin_threshold):
        return False
        
    return is_header_footer(text)

def _is_page_number(text: str, coords: CoordinatesMetadata, page_info: Dict[str, float]) -> bool:

    if len(text) > 10:
        return False
        
    try:
        int(text.strip())
        return True
    except ValueError:
        pass
        
    import re
    page_patterns = [
        r'^\d+$',
        r'^-\s*\d+\s*-$',
        r'^\[\d+\]$',
        r'^Page\s+\d+$'
    ]
    
    return any(re.match(pattern, text.strip()) for pattern in page_patterns)

def _is_footnote(text: str, coords: CoordinatesMetadata, page_info: Dict[str, float]) -> bool:

    if re.match(r'^\d+\.\s+|^\*+\s+', text):
        if coords.y0 < page_info['height'] * 0.2:
            return True
    return False
