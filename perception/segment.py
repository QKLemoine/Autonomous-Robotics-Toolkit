"""
3D Point Cloud Instance Segmentation

This module provides segmentation algorithms for cluttered tabletop environments,
including basic Connected Components Labeling (CCL) and advanced Depth-Aware Region Growing.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np
import scipy.ndimage

# Import your custom RANSAC implementation (adjust path as needed)
# from perception.ransac import table_mask_from_ransac


@dataclass
class CameraIntrinsics:
    """Standard camera intrinsics matrix components."""
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class SegmentationResult:
    """Container for segmentation pipeline outputs."""
    labels: np.ndarray
    num_objects: int
    diagnostics: Optional[Dict[str, Any]] = None


class TabletopSegmenter:
    """
    A perception pipeline for isolating and segmenting objects on flat surfaces.
    """
    def __init__(self, min_object_size: int = 800, max_range_m: float = 2.0):
        self.min_area = min_object_size
        self.max_range = max_range_m

    def extract_foreground(self, depth_m: np.ndarray, plane_model: Any, K: CameraIntrinsics, margin_m: float = 0.01) -> np.ndarray:
        """
        Isolates points sitting above the estimated table plane.
        (Note: Ensure your ransac.py handles the actual mathematical projection).
        """
        # Placeholder for foreground extraction logic. 
        # Typically involves calculating point-to-plane distance and thresholding.
        pass

    def cleanup_mask(self, mask: np.ndarray, open_iters: int = 1, close_iters: int = 2) -> np.ndarray:
        """Removes small artifacts using morphological operations."""
        cleaned = scipy.ndimage.binary_opening(mask, iterations=open_iters)
        cleaned = scipy.ndimage.binary_closing(cleaned, iterations=close_iters)
        return cleaned

    def segment_ccl(self, fg_mask: np.ndarray) -> SegmentationResult:
        """
        Baseline Method: Standard Connected Components Labeling (CCL).
        Groups spatially connected pixels without considering depth boundaries.
        """
        # Use SciPy's optimized CCL algorithm
        structure = scipy.ndimage.generate_binary_structure(2, 2) # 8-connectivity
        labels, num_features = scipy.ndimage.label(fg_mask, structure=structure)
        
        # Filter by minimum area
        component_sizes = np.bincount(labels.ravel())
        too_small = component_sizes < self.min_area
        too_small_mask = too_small[labels]
        labels[too_small_mask] = 0
        
        # Re-index remaining labels sequentially
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels > 0]
        
        final_labels = np.zeros_like(labels)
        for new_id, old_id in enumerate(unique_labels, start=1):
            final_labels[labels == old_id] = new_id
            
        return SegmentationResult(labels=final_labels, num_objects=len(unique_labels))

    def segment_depth_aware(self, fg_clean: np.ndarray, depth_m: np.ndarray, depth_thresh: float = 0.02) -> SegmentationResult:
        """
        Advanced Method: Depth-Aware Region Growing.
        Grows regions using DFS, strictly bounding growth by depth discontinuities.
        Highly effective for touching objects with distinct depth profiles.
        """
        H, W = depth_m.shape
        labels = np.zeros((H, W), dtype=np.int32)
        visited = np.zeros((H, W), dtype=bool)
        
        instance_id = 1

        for y in range(H):
            for x in range(W):
                # Skip if background or already assigned
                if not fg_clean[y, x] or visited[y, x]:
                    continue

                # Initialize region growing
                stack = [(y, x)]
                visited[y, x] = True
                pixels = [(y, x)]

                # Perform DFS region growing
                while stack:
                    cy, cx = stack.pop()

                    # Explore 4-connected neighbors
                    for ny, nx in [(cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)]:
                        if (0 <= ny < H and 0 <= nx < W and not visited[ny, nx] and fg_clean[ny, nx]):
                            
                            # Check depth consistency
                            if abs(depth_m[ny, nx] - depth_m[cy, cx]) < depth_thresh:
                                visited[ny, nx] = True
                                stack.append((ny, nx))
                                pixels.append((ny, nx))

                # Filter out small noisy regions
                if len(pixels) >= self.min_area:
                    for py, px in pixels:
                        labels[py, px] = instance_id
                    instance_id += 1

        return SegmentationResult(labels=labels, num_objects=instance_id - 1)

    def process_scene(self, depth_m: np.ndarray, K: CameraIntrinsics, use_depth_aware: bool = True) -> SegmentationResult:
        """
        Master pipeline: Runs RANSAC, extracts foreground, cleans it, and segments.
        """
        H, W = depth_m.shape
        
        # 1. Estimate Table Plane (requires ransac module integration)
        # table_mask, plane = table_mask_from_ransac(depth_m, K, self.max_range)
        # if plane is None:
        #     return SegmentationResult(labels=np.zeros((H, W), dtype=np.int32), num_objects=0)
        
        # 2. Extract and Clean Foreground
        # fg_mask = self.extract_foreground(depth_m, plane, K)
        # fg_clean = self.cleanup_mask(fg_mask)
        
        # For now, assuming fg_clean is generated:
        fg_clean = np.ones((H, W), dtype=bool) # Placeholder until RANSAC is linked
        
        # 3. Segment
        if use_depth_aware:
            return self.segment_depth_aware(fg_clean, depth_m)
        else:
            return self.segment_ccl(fg_clean)