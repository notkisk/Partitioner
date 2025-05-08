from typing import Tuple, Dict, Any

class CoordinateSystem:
    def __init__(self, width: float, height: float):
        """Initialize coordinate system with page dimensions.
        
        Args:
            width: Page width in points
            height: Page height in points
        """
        self.width = width
        self.height = height
        
    def normalize_bbox(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """Convert coordinates to normalized [0-1] space."""
        x0, y0, x1, y1 = bbox
        return (
            x0 / self.width,
            y0 / self.height,
            x1 / self.width,
            y1 / self.height
        )
        
    def denormalize_bbox(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """Convert from normalized to absolute coordinates."""
        x0, y0, x1, y1 = bbox
        return (
            x0 * self.width,
            y0 * self.height,
            x1 * self.width,
            y1 * self.height
        )
    
    def get_relative_position(self, bbox: Tuple[float, float, float, float]) -> Dict[str, float]:
        """Get relative position metrics for a bbox."""
        x0, y0, x1, y1 = self.normalize_bbox(bbox)
        return {
            'top_margin': y0,
            'bottom_margin': 1 - y1,
            'left_margin': x0,
            'right_margin': 1 - x1,
            'width': x1 - x0,
            'height': y1 - y0,
            'center_x': (x0 + x1) / 2,
            'center_y': (y0 + y1) / 2
        }
