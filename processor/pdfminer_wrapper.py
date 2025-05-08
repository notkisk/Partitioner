from typing import List, Dict, Any, Generator, Optional
from pdfminer.high_level import extract_pages
from pdfminer.layout import (
    LAParams,
    LTPage,
    LTTextBoxHorizontal,
    LTTextLineHorizontal,
    LTChar,
    LTAnno
)

class PDFMinerWrapper:
    def __init__(self, la_params: Optional[Dict[str, Any]] = None):
        """
        Initialize PDFMiner wrapper with custom layout parameters.
        
        Args:
            la_params: Optional custom LAParams configuration
        """
        self.la_params = la_params or {
            'line_margin': 0.3,      # Reduced to keep list items together
            'word_margin': 0.1,
            'char_margin': 0.5,
            'boxes_flow': 0.5,       # Default flow for reading order
            'detect_vertical': False,
            'all_texts': False
        }
        
        self.la_params = LAParams(**self.la_params)
        
    def extract_pages(self, pdf_path: str) -> Generator[LTPage, None, None]:
        """
        Extract pages from PDF using PDFMiner.
        
        Args:
            pdf_path: Path to PDF file
            
        Yields:
            LTPage: PDFMiner page layout object
        """
        yield from extract_pages(pdf_path, laparams=self.la_params)
        
    def extract_text_blocks(self, page: LTPage) -> List[Dict[str, Any]]:
        """
        Extract text blocks with style information from a page.
        
        Args:
            page: PDFMiner page layout object
            
        Returns:
            List[Dict[str, Any]]: List of text blocks with style info
        """
        blocks = []
        
        for element in page:
            if isinstance(element, LTTextBoxHorizontal):
                block = self._process_text_box(element)
                if block:
                    blocks.append(block)
                    
        return blocks
        
    def _process_text_box(self, text_box: LTTextBoxHorizontal) -> Optional[Dict[str, Any]]:
        """
        Process a text box to extract text, style, and layout information.
        
        Args:
            text_box: PDFMiner text box object
            
        Returns:
            Optional[Dict[str, Any]]: Processed text block or None if empty
        """
        text = text_box.get_text().strip()
        if not text:
            return None
            
        # Get bounding box
        bbox = text_box.bbox
        
        # Extract style information
        style_info = self._extract_style_info(text_box)
        
        # Get line information
        lines = []
        for line in text_box:
            if isinstance(line, LTTextLineHorizontal):
                line_text = line.get_text().strip()
                if line_text:
                    lines.append({
                        'text': line_text,
                        'bbox': line.bbox,
                        'style': self._extract_style_info(line)
                    })
        
        return {
            'text': text,
            'bbox': bbox,
            'style_info': style_info,
            'lines': lines
        }
        
    def _extract_style_info(self, element) -> Dict[str, Any]:
        """
        Extract style information from a PDFMiner element.
        
        Args:
            element: PDFMiner layout element
            
        Returns:
            Dict[str, Any]: Style information
        """
        style_info = {
            'font_name': None,
            'font_size': 0,
            'is_bold': False,
            'is_italic': False,
            'is_monospace': False,
            'char_count': 0,
            'fonts': {}
        }
        
        # Process all characters to get dominant style
        for obj in element:
            if isinstance(obj, LTChar):
                style_info['char_count'] += 1
                font_name = obj.fontname
                style_info['fonts'][font_name] = style_info['fonts'].get(font_name, 0) + 1
                
                # Update running size average
                style_info['font_size'] = (
                    (style_info['font_size'] * (style_info['char_count'] - 1) + obj.size)
                    / style_info['char_count']
                )
                
        # Determine dominant font
        if style_info['fonts']:
            dominant_font = max(style_info['fonts'].items(), key=lambda x: x[1])[0]
            style_info['font_name'] = dominant_font
            style_info['is_bold'] = 'Bold' in dominant_font or 'bold' in dominant_font
            style_info['is_italic'] = 'Italic' in dominant_font or 'italic' in dominant_font
            style_info['is_monospace'] = 'Mono' in dominant_font or 'Courier' in dominant_font
            
        return style_info
        
    def collect_page_statistics(self, page: LTPage) -> Dict[str, Any]:
        """
        Collect statistics about text styles and layout from a page.
        
        Args:
            page: PDFMiner page layout object
            
        Returns:
            Dict[str, Any]: Page statistics
        """
        stats = {
            'font_sizes': [],
            'line_heights': [],
            'font_counts': {},
            'x_positions': [],
            'y_positions': []
        }
        
        for element in page:
            if isinstance(element, LTTextBoxHorizontal):
                for line in element:
                    if isinstance(line, LTTextLineHorizontal):
                        stats['line_heights'].append(line.height)
                        stats['x_positions'].append(line.x0)
                        stats['y_positions'].append(line.y0)
                        
                        for char in line:
                            if isinstance(char, LTChar):
                                stats['font_sizes'].append(char.size)
                                stats['font_counts'][char.fontname] = \
                                    stats['font_counts'].get(char.fontname, 0) + 1
                                    
        return stats
