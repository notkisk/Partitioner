import camelot
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import fitz  # PyMuPDF
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def extract_tables_as_html(pdf_path: str, pages: str = 'all', flavor: str = 'lattice') -> List[Dict[str, Any]]:
    """
    Extract tables from a PDF using Camelot and return them as HTML strings with metadata.
    
    Args:
        pdf_path: Path to the PDF file
        pages: Pages to process (default: 'all')
        flavor: Camelot flavor ('lattice' or 'stream')
        
    Returns:
        List of table dictionaries with HTML content and metadata
    """
    try:
        tables = camelot.read_pdf(
            str(pdf_path),
            pages=pages,
            flavor=flavor,
            strip_text='\n',
            suppress_stdout=True,
            backend='poppler'  # or 'ghostscript'
        )
        
        if not tables:
            logger.debug("No tables found in the PDF")
            return []
            
        # Get page dimensions for accurate bbox calculations
        doc = fitz.open(pdf_path)
        page_dims = {}
        for page in doc:
            page_dims[page.number + 1] = {
                'width': page.rect.width,
                'height': page.rect.height
            }
        doc.close()
        
        table_blocks = []
        for i, table in enumerate(tables):
            try:
                page_num = table.page
                # Clean and process the table data
                df = table.df
                df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
                df.replace(["", "nan", "NaN", "NULL"], pd.NA, inplace=True)
                df.dropna(how="all", inplace=True)
                df.dropna(how="all", axis=1, inplace=True)
                df.fillna("", inplace=True)

                # Check if the table DataFrame contains any meaningful text content
                has_content = False
                for col in df.columns:
                    if df[col].astype(str).str.strip().any():
                        has_content = True
                        break

                if not has_content:
                    logger.info(f"Skipping table {i+1} on page {page_num} as it contains no meaningful text content after cleaning.")
                    continue # Skip this table block
                
                # Generate clean HTML without extra newlines or classes
                def df_to_clean_html(df):
                    parts = ['<table>']
                    
                    # Add headers if they exist
                    if not df.empty:
                        parts.append('<thead><tr>')
                        for col in df.columns:
                            parts.append(f'<th>{col}</th>' if pd.notna(col) else '<th></th>')
                        parts.append('</tr></thead>')
                    
                    # Add rows
                    parts.append('<tbody>')
                    for _, row in df.iterrows():
                        parts.append('<tr>')
                        for item in row:
                            cell = str(item) if pd.notna(item) else ''
                            parts.append(f'<td>{cell}</td>')
                        parts.append('</tr>')
                    parts.append('</tbody></table>')
                    
                    # Join without newlines between tags
                    return ''.join(parts)
                
                html = df_to_clean_html(df)
                
                # Get page dimensions for bbox scaling
                page_num = table.parsing_report.get('page', 1)
                page_width = page_dims.get(page_num, {}).get('width', 612)  # Default to letter size
                page_height = page_dims.get(page_num, {}).get('height', 792)
                
                raw_camelot_bbox = table._bbox
                logger.debug(f"Table {i+1}, Page {page_num}: Raw Camelot _bbox: {raw_camelot_bbox}")
                
                # raw_camelot_bbox is (camelot_original_x1, camelot_original_y2, camelot_original_x2, camelot_original_y1)
                # camelot_original_x1: Left x-coordinate
                # camelot_original_y2: Top y-coordinate
                # camelot_original_x2: Right x-coordinate
                # camelot_original_y1: Bottom y-coordinate
                
                c_left_x   = float(raw_camelot_bbox[0])
                c_top_y    = float(raw_camelot_bbox[1])
                c_right_x  = float(raw_camelot_bbox[2])
                c_bottom_y = float(raw_camelot_bbox[3])
                
                camelot_x_left_pdf = c_left_x
                camelot_y_top_edge_from_page_top = c_top_y
                camelot_x_right_pdf = c_right_x
                camelot_y_bottom_edge_from_page_top = c_bottom_y

                logger.debug(f"Table {i+1}, Page {page_num}: Page height for y-conversion: {page_height}")
                logger.debug(f"Table {i+1}, Page {page_num}: Parsed Camelot Coords (y from page top): x_left={camelot_x_left_pdf}, y_top_edge_from_pg_top={camelot_y_top_edge_from_page_top}, x_right={camelot_x_right_pdf}, y_bottom_edge_from_pg_top={camelot_y_bottom_edge_from_page_top}")

                system_x0_left   = camelot_x_left_pdf
                system_y0_bottom = page_height - camelot_y_bottom_edge_from_page_top
                system_x1_right  = camelot_x_right_pdf
                system_y1_top    = page_height - camelot_y_top_edge_from_page_top
                
                bbox = [
                    system_x0_left,
                    system_y0_bottom,
                    system_x1_right,
                    system_y1_top
                ]
                logger.debug(f"Table {i+1}, Page {page_num}: Constructed bbox for system: {bbox} (height = {bbox[3]-bbox[1]})")
                
                table_block = {
                    "type": "table",
                    "page": page_num,
                    "bbox": bbox,
                    "html": html,
                    "accuracy": table.parsing_report.get('accuracy', 0),
                    "whitespace": table.parsing_report.get('whitespace', 0),
                    "order": table.order,
                    "flavor": flavor,
                    "page_width": page_width,
                    "page_height": page_height
                }
                
                table_blocks.append(table_block)
                
            except Exception as e:
                logger.error(f"Error processing table {i+1}: {str(e)}")
                continue
                
        return table_blocks
        
    except Exception as e:
        logger.error(f"Error extracting tables: {str(e)}")
        return []

def process_pdf_tables(pdf_path: str, flavor: str = 'lattice'):
    """
    Process tables from a PDF using the specified extraction method.
    
    Args:
        pdf_path: Path to the PDF file
        flavor: Table extraction method ('lattice' or 'stream')
        
    Returns:
        List of processed tables with metadata
    """
    return extract_tables_as_html(pdf_path, flavor=flavor)
