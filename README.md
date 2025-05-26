# Partitioner


A Python library for extracting and classifying text elements from PDF documents. The processor uses advanced heuristics and machine learning techniques to identify different types of text elements such as titles, list items, headers, footers, and more.

## Features

- Enhanced list item detection with support for:
  - Unicode bullets and symbols
  - Numbered lists (decimal, roman, alphabetical)
  - Multi-line items and nested lists
- Intelligent text block relationships using:
  - Font and style analysis
  - Spatial positioning
  - Visual hierarchy detection
- Advanced document structure analysis:
  - Title and heading classification
  - Header and footer detection
  - Page number extraction
  - Footnote recognition
- Robust coordinate system handling
- Statistical analysis of document styles

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/pdf-processor.git
cd Partitioner
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Process a single PDF file with default settings:
```bash
python main.py path/to/your/document.pdf
```

Customize layout analysis parameters:
```bash
python main.py path/to/your/document.pdf --line-margin 0.5 --char-margin 2.0 --word-margin 0.1 --detect-vertical
```

### Python API

```python
from processor.pipeline import process_pdf

# Process with default parameters
elements = process_pdf('path/to/your/document.pdf')

# Customize layout analysis
la_params = {
    'line_margin': 0.5,
    'char_margin': 2.0,
    'word_margin': 0.1,
    'boxes_flow': 0.5,
    'detect_vertical': True,
    'all_texts': True
}
elements = process_pdf('path/to/your/document.pdf', la_params=la_params)
```

Process all PDFs in a directory:
```bash
python main.py path/to/pdf/directory
```

Optional arguments:
- `-o, --output`: Specify output directory
- `--config`: Path to custom LAParams configuration
- `--log-file`: Path to log file
- `-v, --verbose`: Increase output verbosity

### Python API

```python
from processor import process_pdf

# Process a PDF file
elements = process_pdf('path/to/document.pdf')

# Each element has:
for element in elements:
    print(f"Type: {element.element_type}")
    print(f"Text: {element.text}")
    print(f"Page: {element.page_number}")
    print(f"Confidence: {element.metadata.confidence}")
```

## Configuration

You can customize the PDFMiner LAParams by providing a JSON configuration file:

```json
{
    "line_margin": 0.3,
    "word_margin": 0.1,
    "char_margin": 0.5,
    "boxes_flow": 0.5,
    "detect_vertical": false,
    "all_texts": false
}
```

## Output Format

The processor generates a JSON file with the following structure:

```json
{
    "source_file": "path/to/input.pdf",
    "elements": [
        {
            "text": "Example text",
            "element_type": "LIST_ITEM",
            "bbox": [x0, y0, x1, y1],
            "page_number": 1,
            "metadata": {
                "confidence": 0.95,
                "style_info": {
                    "font_name": "Arial",
                    "font_size": 12,
                    "is_bold": false
                },
                "list_group_id": "abc123"
            }
        }
    ]
}
```

## Project Structure

```
pdf_processor/
├── main.py                     # CLI interface
├── processor/
│   ├── __init__.py
│   ├── pipeline.py             # Main processing pipeline
│   ├── pdfminer_wrapper.py     # PDFMiner integration
│   ├── spatial_grouper.py      # Text block merging
│   ├── element_classifier.py   # Element classification
│   ├── list_handler.py         # List processing
│   ├── patterns.py            # Regex patterns
│   └── data_models.py         # Data structures
├── utils/
│   ├── __init__.py
│   ├── coordinates.py         # Geometry utilities
│   ├── file_io.py            # File operations
│   └── logger.py             # Logging setup
└── configs/                   # Configuration files
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
