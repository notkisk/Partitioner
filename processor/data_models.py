from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, List, Tuple, Optional
import uuid

class ElementType(Enum):
    """Types of text elements that can be identified."""
    # Text elements
    TITLE = "TITLE"
    TEXT = "TEXT"
    BLOCK_EQUATION = "BLOCK_EQUATION"
    
    # List elements
    LIST_ITEM = "LIST_ITEM"
    LIST = "LIST"
    INDEX = "INDEX"
    
    # Table elements
    TABLE_BODY = "TABLE_BODY"
    TABLE_CAPTION = "TABLE_CAPTION"
    TABLE_FOOTNOTE = "TABLE_FOOTNOTE"
    
    # Figure elements (formerly Image)
    FIGURE = "FIGURE"
    FIGURE_CAPTION = "FIGURE_CAPTION"
    FIGURE_FOOTNOTE = "FIGURE_FOOTNOTE"
    
    # Other
    DISCARDED = "DISCARDED"
    
    # Legacy types (for backward compatibility)
    NARRATIVE_TEXT = "TEXT"
    HEADER = "TITLE"
    FOOTER = "TEXT"
    PAGE_NUMBER = "TEXT"
    FOOTNOTE = "TEXT"
    CONTACT_INFO = "TEXT"
    TABLE_CELL = "TABLE_BODY"
    CAPTION = "TEXT"
    UNKNOWN = "DISCARDED"
    ANNOTATION = "TEXT"

@dataclass
class CoordinatesMetadata:
    x0: float
    y0: float
    x1: float
    y1: float
    page_width: float
    page_height: float
    relative_x: float = 0.0
    relative_y: float = 0.0
    indent_ratio: float = 0.0

    def __post_init__(self):
        if self.page_width and self.page_height:
            self.relative_x = self.x0 / self.page_width
            self.relative_y = self.y0 / self.page_height
            self.indent_ratio = self.x0 / self.page_width
            
    @property
    def points(self) -> Tuple[float, float, float, float]:
        """Get the coordinates as a tuple of points."""
        return (self.x0, self.y0, self.x1, self.y1)

@dataclass
class ElementMetadata:
    style_info: Dict[str, Any] = field(default_factory=dict)
    coordinates: Optional[CoordinatesMetadata] = None
    list_group_id: Optional[str] = None
    confidence: float = 0.0
    cross_page_refs: List[int] = field(default_factory=list)
    page_number: Optional[int] = None
    table_data: Optional[Dict[str, Any]] = None

@dataclass
class Element:
    text: str
    element_type: ElementType
    bbox: Tuple[float, float, float, float]
    page_number: int
    metadata: ElementMetadata = field(default_factory=ElementMetadata)

    @property
    def id(self) -> str:
        if not hasattr(self, '_id'):
            self._id = uuid.uuid4().hex
        return self._id
