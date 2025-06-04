import re
import logging
from typing import Dict, Any, Optional, Tuple, List
from .data_models import CoordinatesMetadata

logger = logging.getLogger(__name__)

def is_likely_caption(
    text,
    bbox,
    style_info,
    page_info,
    elements_before=None,
    elements_after=None
):
    if not text or not bbox or len(bbox) < 4:
        return False, 0.0
    try:
        bbox = [float(c) for c in bbox]
    except:
        return False, 0.0
    text = text.strip()
    if not text:
        return False, 0.0
    words = [w for w in text.split() if w.strip()]
    word_count = len(words)
    if word_count > 40 or len(text) > 300:
        return False, 0.0
    from .text_patterns import CAPTION_RE, CAPTION_END_PATTERNS
    for pattern in CAPTION_RE:
        if pattern.match(text):
            for end_pat in CAPTION_END_PATTERNS:
                m = re.search(end_pat, text)
                if m:
                    text = text[:m.end()]
                    break
            return True, 0.95
    if 3 <= word_count <= 30:
        short_score = 0.15
    else:
        short_score = 0.0
    raw_width = page_info.get('width', 0)
    try:
        page_width = float(raw_width)
    except (TypeError, ValueError):
        page_width = 0.0
    pos_score = 0.0
    if page_width > 0:
        center_x = (bbox[0] + bbox[2]) / 2
        relative_center = center_x / page_width
        if 0.4 <= relative_center <= 0.6:
            pos_score = 0.05
    fig_score = 0.0
    if elements_before and bbox and len(bbox) >= 4:
        for i in range(1, min(4, len(elements_before) + 1)):
            last_element = elements_before[-i]
            last_bbox = getattr(last_element, 'bbox', None) or (last_element.get('bbox') if isinstance(last_element, dict) else None)
            last_type = getattr(last_element, 'element_type', None) or (last_element.get('element_type') if isinstance(last_element, dict) else None)
            if not last_bbox or len(last_bbox) < 4:
                continue
            if (last_bbox[2] - last_bbox[0]) * (last_bbox[3] - last_bbox[1]) < 400:
                continue
            vertical_gap = bbox[1] - last_bbox[3]
            if 0 < vertical_gap <= 40:
                overlap_start = max(bbox[0], last_bbox[0])
                overlap_end = min(bbox[2], last_bbox[2])
                overlap_width = max(0, overlap_end - overlap_start)
                caption_width = bbox[2] - bbox[0]
                overlap_pct = overlap_width / caption_width if caption_width > 0 else 0
                if overlap_pct > 0.5:
                    if last_type and any(t in str(last_type).lower() for t in ['image', 'figure', 'table', 'chart']):
                        fig_score = 0.4
                        break
    font_size = style_info.get('size', style_info.get('font_size', 0))
    median_font = page_info.get('median_font_size', 12)
    font_score = 0.0
    if font_size > 0 and median_font > 0:
        size_ratio = font_size / median_font
        if size_ratio < 0.92:
            font_score = 0.05
    italic_score = 0.05 if style_info.get('italic', False) else 0.0
    text_lower = text.lower()
    phrases = ['figure', 'fig.', 'table', 'tbl.', 'tab.', 'chart', 'diagram', 'image', 'photo', 'source:', 'caption:']
    phrase_score = 0.05 if any(p in text_lower for p in phrases) else 0.0
    single_line_score = 0.05 if '\n' not in text else 0.0
    conf = short_score + pos_score + fig_score + font_score + italic_score + phrase_score + single_line_score
    if conf >= 0.7:
        return True, min(conf, 1.0)
    return False, conf

def find_caption_for_figure(figure_element, elements: list) -> Optional[int]:

    if not figure_element or not elements:
        return None
    
    def get_bbox(element):
        raw = element.bbox if hasattr(element, 'bbox') else element.get('bbox') if isinstance(element, dict) else None
        if not raw or len(raw) < 4:
            return None
        try:
            return [float(c) for c in raw]
        except:
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
    try:
        figure_bbox = [float(c) for c in figure_bbox]
    except:
        return None
    figure_idx = -1
    for idx, element in enumerate(elements):
        if element is figure_element or (isinstance(element, dict) and element == figure_element):
            figure_idx = idx
            break
    
    if figure_idx == -1:
        return None
    
    for i in range(figure_idx + 1, min(figure_idx + 5, len(elements))):
        element = elements[i]
        element_bbox = get_bbox(element)
        if element_bbox and len(element_bbox) >= 4 and element_bbox[1] > figure_bbox[3]:  
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
    
    page_num = figure_element.get('page_number') if isinstance(figure_element, dict) else getattr(figure_element, 'page_number', None)
    spatial_below = []
    for idx, el in enumerate(elements):
        bbox = get_bbox(el)
        if not bbox or len(bbox) < 4:
            continue
        el_page = el.get('page_number') if isinstance(el, dict) else getattr(el, 'page_number', None)
        if el_page != page_num or idx == figure_idx:
            continue
        y0 = bbox[1]
        if y0 >= figure_bbox[3]:
            spatial_below.append((idx, bbox))
    spatial_below.sort(key=lambda x: x[1][1] - figure_bbox[3])
    for idx, bbox in spatial_below[:10]:
        el = elements[idx]
        text = get_text(el)
        style = get_style_info(el)
        is_cap, conf = is_likely_caption(text, bbox, style, {'width': bbox[2]*2, 'height': bbox[3]*2})
        logger.debug(f"Spatial below test for idx {idx}, text={text[:30]!r}, conf={conf}")
        if is_cap:
            return idx
    spatial_above = []
    for idx, el in enumerate(elements):
        bbox = get_bbox(el)
        if not bbox or len(bbox) < 4:
            continue
        el_page = el.get('page_number') if isinstance(el, dict) else getattr(el, 'page_number', None)
        if el_page != page_num or idx == figure_idx:
            continue
        y1 = bbox[3]
        if y1 <= figure_bbox[1]:
            spatial_above.append((idx, bbox))
    spatial_above.sort(key=lambda x: figure_bbox[1] - x[1][3])
    for idx, bbox in spatial_above[:10]:
        el = elements[idx]
        text = get_text(el)
        style = get_style_info(el)
        is_cap, conf = is_likely_caption(text, bbox, style, {'width': bbox[2]*2, 'height': bbox[3]*2})
        logger.debug(f"Spatial above test for idx {idx}, text={text[:30]!r}, conf={conf}")
        if is_cap:
            return idx
    return None
