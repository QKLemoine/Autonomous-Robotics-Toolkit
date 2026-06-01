"""
Core data structures for 3D perception and segmentation pipelines.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np

@dataclass
class CameraIntrinsics:
    """Standard camera intrinsics matrix components."""
    fx: float
    fy: float
    cx: float
    cy: float

@dataclass
class SegmentationResult:
    """
    Container for the results of an instance segmentation method.
    """
    labels: np.ndarray
    num_objects: int
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensures labels are correctly typed and counts match."""
        if self.labels.dtype != np.int32:
            self.labels = self.labels.astype(np.int32)
            
        actual_num = int(self.labels.max())
        if actual_num != self.num_objects:
            self.num_objects = actual_num