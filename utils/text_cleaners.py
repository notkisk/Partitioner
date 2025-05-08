import re
import sys
import unicodedata
import quopri
import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import Counter

class TextCleaner:
    # Core patterns
    UNICODE_BULLETS_RE = re.compile(r'^[\s\t]*(?:[\u2022\u2023\u25E6\u2043\u2219*•·\⁃]\s*)+', re.MULTILINE)
    UNICODE_BULLETS_RE_0W = re.compile(r'[\u2022\u2023\u25E6\u2043\u2219*•·\⁃]')
    PARAGRAPH_PATTERN = re.compile(r'\n')
    DOUBLE_PARAGRAPH_PATTERN = re.compile(r'\n\s*\n')
    LINE_BREAK_RE = re.compile(r'\n')
    E_BULLET_PATTERN = re.compile(r'^e[\s\n]')
    
    # Extraction patterns
    EMAIL_PATTERN = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
    IP_PATTERN = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
    IP_NAME_PATTERN = re.compile(r'(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?:$|\n)')
    US_PHONE_PATTERN = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
    EMAIL_DATETIMETZ_PATTERN = r'(?:[A-Za-z]{3},\s+\d{1,2}\s+[A-Za-z]{3}\s+\d{4}\s+\d{2}:\d{2}:\d{2}\s+[+-]\d{4})'
    IMAGE_URL_PATTERN = r'<img[^>]+src=[\"\']([^\"\'>]+)[\"\'][^>]*>'
    MAPI_ID_PATTERN = r'[0-9A-F]{8}[-]?(?:[0-9A-F]{4}[-]?){3}[0-9A-F]{12};'

    @staticmethod
    def clean_extra_whitespace(text: str, max_newlines: int = 2) -> str:
        lines = text.strip().split('\n')
        cleaned_lines = [' '.join(line.split()) for line in lines]
        text = '\n'.join(line for line in cleaned_lines if line)
        
        while '\n' * (max_newlines + 1) in text:
            text = text.replace('\n' * (max_newlines + 1), '\n' * max_newlines)
        return text

    @staticmethod
    def clean_extra_whitespace_with_index_run(text: str) -> Tuple[str, np.ndarray]:
        if not text:
            return '', np.array([])

        cleaned_text = TextCleaner.clean_extra_whitespace(text)
        if not cleaned_text:
            return '', np.zeros(0)

        moved_indices = np.zeros(len(cleaned_text), dtype=int)
        distance = 0
        orig_idx = 0
        clean_idx = 0

        while clean_idx < len(cleaned_text):
            if orig_idx >= len(text):
                moved_indices[clean_idx:] = distance
                break

            is_match = text[orig_idx] == cleaned_text[clean_idx]
            is_mapped_whitespace = text[orig_idx].isspace() and cleaned_text[clean_idx].isspace()

            if is_match or is_mapped_whitespace:
                moved_indices[clean_idx] = distance
                orig_idx += 1
                clean_idx += 1
            else:
                distance += 1
                orig_idx += 1

        return cleaned_text, moved_indices

    @staticmethod
    def normalize_ligatures(text: str) -> str:
        ligature_map = {
            'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
            'æ': 'ae', 'Æ': 'AE', 'ﬅ': 'ft', 'ʪ': 'ls', 'œ': 'oe',
            'Œ': 'OE', 'ȹ': 'qp', 'ﬆ': 'st', 'ʦ': 'ts'
        }
        for ligature, replacement in ligature_map.items():
            text = text.replace(ligature, replacement)
        return unicodedata.normalize('NFKC', text)

    @staticmethod
    def dehyphenate_text(text: str) -> str:
        pattern = r'(\w+)-\n(\w+)'
        return re.sub(pattern, r'\1\2', text)

    @staticmethod
    def remove_control_characters(text: str, keep_common_whitespace: bool = True) -> str:
        if keep_common_whitespace:
            allowed = {'\n', '\r', '\t'}
            return ''.join(char for char in text if char.isprintable() or char in allowed)
        return ''.join(char for char in text if char.isprintable())

    @staticmethod
    def clean_text(text: str, max_newlines: int = 2, clean_bullets: bool = True,
                   clean_dashes: bool = True, clean_trailing_punct: bool = True,
                   extra_whitespace: bool = True, lowercase: bool = False) -> str:
        text = TextCleaner.remove_control_characters(text)
        text = TextCleaner.normalize_ligatures(text)
        text = TextCleaner.dehyphenate_text(text)
        text = TextCleaner.replace_mime_encodings(text)
        text = text.lower() if lowercase else text
        
        if clean_bullets:
            text = TextCleaner.clean_bullets(text)
            text = TextCleaner.clean_ordered_bullets(text)
        
        if clean_dashes:
            text = TextCleaner.clean_dashes(text)
            
        if clean_trailing_punct:
            text = TextCleaner.clean_trailing_punctuation(text)
            
        if extra_whitespace:
            text = TextCleaner.clean_extra_whitespace(text, max_newlines)
            
        return text.strip()

    @staticmethod
    def clean_bullets(text: str) -> str:
        search = TextCleaner.UNICODE_BULLETS_RE.match(text)
        if search is None:
            return text
        cleaned_text = TextCleaner.UNICODE_BULLETS_RE.sub('', text, 1)
        return cleaned_text.strip()

    @staticmethod
    def clean_ordered_bullets(text: str) -> str:
        text_sp = text.split()
        if len(text_sp) < 2:
            return text
            
        if any(['.' not in text_sp[0], '..' in text_sp[0]]):
            return text

        bullet = re.split(pattern=r'[\.]+', string=text_sp[0])
        if not bullet[-1]:
            bullet.pop()

        if not bullet or len(bullet[0]) > 2:
            return text

        is_valid_bullet = all(
            part.isalnum() or (len(part)==1 and part in 'ivxlcdmIVXLCDM') 
            for part in bullet if part
        )

        return ' '.join(text_sp[1:]) if is_valid_bullet else text

    @staticmethod
    def clean_dashes(text: str) -> str:
        return re.sub(r'[-–—]', ' ', text).strip()

    @staticmethod
    def clean_trailing_punctuation(text: str) -> str:
        return text.strip().rstrip('.,:;')

    @staticmethod
    def replace_mime_encodings(text: str, encoding: str = 'utf-8') -> str:
        try:
            text_bytes = text.encode(encoding)
            decoded_bytes = quopri.decodestring(text_bytes)
            return decoded_bytes.decode(encoding)
        except Exception:
            return text

    @staticmethod
    def clean_prefix(text: str, pattern: str, ignore_case: bool = False, strip_leading: bool = True) -> str:
        flags = re.IGNORECASE if ignore_case else 0
        cleaned_text = re.sub(rf'^{pattern}', '', text, count=1, flags=flags)
        return cleaned_text.lstrip() if strip_leading else cleaned_text

    @staticmethod
    def clean_postfix(text: str, pattern: str, ignore_case: bool = False, strip_trailing: bool = True) -> str:
        flags = re.IGNORECASE if ignore_case else 0
        cleaned_text = re.sub(rf'{pattern}$', '', text, count=1, flags=flags)
        return cleaned_text.rstrip() if strip_trailing else cleaned_text

    @staticmethod
    def group_bullet_paragraph(paragraph: str) -> list:
        clean_paragraphs = []
        paragraph = (re.sub(TextCleaner.E_BULLET_PATTERN, '·', paragraph)).strip()

        bullet_paras = re.split(TextCleaner.UNICODE_BULLETS_RE_0W, paragraph)
        for bullet in bullet_paras:
            if bullet:
                clean_paragraphs.append(re.sub(TextCleaner.PARAGRAPH_PATTERN, ' ', bullet))
        return clean_paragraphs

    @staticmethod
    def group_broken_paragraphs(text: str) -> str:
        paragraphs = TextCleaner.DOUBLE_PARAGRAPH_PATTERN.split(text)
        clean_paragraphs = []
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
                
            para_split = TextCleaner.PARAGRAPH_PATTERN.split(paragraph)
            all_lines_short = all(len(line.strip().split()) < 5 for line in para_split)
            
            if TextCleaner.UNICODE_BULLETS_RE.match(paragraph.strip()) or \
               TextCleaner.E_BULLET_PATTERN.match(paragraph.strip()):
                clean_paragraphs.extend(TextCleaner.group_bullet_paragraph(paragraph))
            elif all_lines_short:
                clean_paragraphs.extend([line for line in para_split if line.strip()])
            else:
                clean_paragraphs.append(re.sub(TextCleaner.PARAGRAPH_PATTERN, ' ', paragraph))

        return '\n\n'.join(clean_paragraphs)

    @staticmethod
    def extract_email_addresses(text: str) -> List[str]:
        return TextCleaner.EMAIL_PATTERN.findall(text)

    @staticmethod
    def extract_ip_addresses(text: str) -> List[str]:
        return TextCleaner.IP_PATTERN.findall(text)

    @staticmethod
    def extract_phone_numbers(text: str) -> List[str]:
        regex_match = TextCleaner.US_PHONE_PATTERN.search(text)
        if regex_match is None:
            return []
        start, end = regex_match.span()
        return [text[start:end].strip()]
        
    @staticmethod
    def extract_ordered_bullets(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        a, b, c = None, None, None
        text_sp = text.split()
        if any(['.' not in text_sp[0], '..' in text_sp[0]]):
            return a, b, c

        bullet = re.split(pattern=r'[\.]+', string=text_sp[0])
        if not bullet[-1]:
            bullet.pop()

        if len(bullet[0]) > 2:
            return a, b, c

        a, *temp = bullet
        if temp:
            try:
                b, c, *_ = temp
            except ValueError:
                b, = temp
                c = None
            b = ''.join(b)
            c = ''.join(c) if c else None
        return a, b, c
        
    @staticmethod
    def extract_image_urls(text: str) -> List[str]:
        return re.findall(TextCleaner.IMAGE_URL_PATTERN, text)
        
    @staticmethod
    def extract_mapi_id(text: str) -> List[str]:
        mapi_ids = re.findall(TextCleaner.MAPI_ID_PATTERN, text)
        return [mid.replace(';', '') for mid in mapi_ids]
        
    @staticmethod
    def extract_datetimetz(text: str) -> Optional[datetime.datetime]:
        date_extractions = re.findall(TextCleaner.EMAIL_DATETIMETZ_PATTERN, text)
        if date_extractions:
            return datetime.datetime.strptime(
                date_extractions[0],
                '%a, %d %b %Y %H:%M:%S %z'
            )
        return None
        
    @staticmethod
    def extract_text_before(text: str, pattern: str, index: int = 0, strip: bool = True) -> str:
        matches = list(re.finditer(pattern, text))
        if not matches or index < 0 or index >= len(matches):
            raise ValueError(f'Pattern not found or invalid index: {index}')
            
        start, _ = matches[index].span()
        before_text = text[:start]
        return before_text.rstrip() if strip else before_text
        
    @staticmethod
    def extract_text_after(text: str, pattern: str, index: int = 0, strip: bool = True) -> str:
        matches = list(re.finditer(pattern, text))
        if not matches or index < 0 or index >= len(matches):
            raise ValueError(f'Pattern not found or invalid index: {index}')
            
        _, end = matches[index].span()
        after_text = text[end:]
        return after_text.lstrip() if strip else after_text
        
    @staticmethod
    def new_line_grouper(text: str) -> str:
        paragraphs = TextCleaner.LINE_BREAK_RE.split(text)
        clean_paragraphs = [para for para in paragraphs if para.strip()]
        return '\n\n'.join(clean_paragraphs)
        
    @staticmethod
    def auto_paragraph_grouper(text: str, max_line_count: int = 2000, threshold: float = 0.1) -> str:
        lines = TextCleaner.LINE_BREAK_RE.split(text)
        max_line_count = min(len(lines), max_line_count)
        line_count = empty_line_count = 0
        
        for line in lines[:max_line_count]:
            line_count += 1
            if not line.strip():
                empty_line_count += 1
                
        ratio = empty_line_count / line_count if line_count else 0
        return TextCleaner.new_line_grouper(text) if ratio < threshold else TextCleaner.group_broken_paragraphs(text)
