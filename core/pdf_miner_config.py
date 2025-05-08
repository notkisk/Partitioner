from dataclasses import dataclass
from pdfminer.layout import LAParams

@dataclass
class PDFMinerConfig:
    char_margin: float = 2.0  # Space between chars to be considered separate
    line_margin: float = 0.5  # Space between lines within a text box
    word_margin: float = 0.1  # Space between words
    line_overlap: float = 0.5  # Ratio of overlap for lines to be combined
    boxes_flow: float = 0.5   # Text flow direction (0.5 = mixed)
    detect_vertical: bool = False  # Whether to detect vertical text
    all_texts: bool = True    # Force all text to be extracted

    def to_laparams(self) -> LAParams:
        return LAParams(
            char_margin=self.char_margin,
            line_margin=self.line_margin,
            word_margin=self.word_margin,
            line_overlap=self.line_overlap,
            boxes_flow=self.boxes_flow,
            detect_vertical=self.detect_vertical,
            all_texts=self.all_texts
        )

    @classmethod
    def strict_mode(cls) -> 'PDFMinerConfig':
        """Strict mode: Less likely to merge text blocks"""
        return cls(
            char_margin=1.0,
            line_margin=0.3,
            word_margin=0.1,
            line_overlap=0.3
        )

    @classmethod
    def loose_mode(cls) -> 'PDFMinerConfig':
        """Loose mode: More likely to merge text blocks"""
        return cls(
            char_margin=3.0,
            line_margin=0.7,
            word_margin=0.2,
            line_overlap=0.7
        )

    @classmethod
    def adaptive_mode(cls, avg_char_width: float, avg_line_height: float) -> 'PDFMinerConfig':
        """Adaptive mode: Adjusts parameters based on document metrics"""
        return cls(
            char_margin=max(1.5, avg_char_width * 0.3),
            line_margin=max(0.3, avg_line_height * 0.2),
            word_margin=max(0.1, avg_char_width * 0.1),
            line_overlap=0.5
        )
