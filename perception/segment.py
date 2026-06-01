"""
Inputs:
- rgb:     (H, W, 3) uint8   RGB image in **RGB** channel order
- depth_m: (H, W)    float32 depth in **meters**
- K:       Intrinsics object with fields (fx, fy, cx, cy)

Output:
- SegmentationResult with fields:
  - labels:      (H, W) int32, 0=background, 1..N=instances
  - num_objects: int, number of instances N
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np


from utils.segmentation_types import SegmentationResult
from utils.ocid_io import Intrinsics
from baseline import baseline_solve
from utils.segmentation import table_mask_from_ransac, foreground_from_plane, cleanup_mask

def solve(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    K: Intrinsics,
    params: Optional[Dict[str, Any]] = None,
) -> SegmentationResult:
    """
    Implement your instance segmentation method here.

    Args:
        rgb: (H, W, 3) uint8 RGB image (RGB order)
        depth_m: (H, W) float32 depth in meters
        K: camera intrinsics (fx, fy, cx, cy)
        params: optional config dict (you can define your own keys)

    Returns:
        SegmentationResult:
            labels: (H, W) int32, 0=background, 1..N=instances
            num_objects: int, number of instances
    """

    DEFAULT_PARAMS = {
        'max_range_m': 2.0,
        'ransac': {
            'iters': 2000,
            'inlier_thresh_m': 0.008,
            'min_inliers': 5000,
            'seed': 0
        },
        'min_object_size': 800
    }

    # Use default parameters if none provided
    if params is None:
        params = DEFAULT_PARAMS

    H, W = depth_m.shape

    # Extract configuration values
    max_range = float(params.get('max_range_m', 2.0))
    ransac_cfg = params.get('ransac', {})
    min_area = int(params.get('min_object_size', 800))

    # ====================================================================
    # Table Plane Estimation (RANSAC)
    # ====================================================================
    table_mask, plane = table_mask_from_ransac(
        depth_m,
        K,
        max_range,
        ransac_cfg
    )

    # If no plane found, return empty segmentation
    if plane is None:
        return SegmentationResult(
            labels=np.zeros((H, W), dtype=np.int32),
            num_objects=0,
        )

    # ====================================================================
    # Foreground Extraction
    #
    # Foreground consists of points above the detected table plane.
    # ====================================================================
    fg_mask = foreground_from_plane(
        depth_m,
        K,
        plane,
        max_range,
        margin_m=0.01
    )

    # Clean small artifacts using morphological operations
    fg_clean = cleanup_mask(
        fg_mask,
        open_iters=1,
        close_iters=2
    )

    # ====================================================================
    # Depth-Aware Region Growing
    #
    # Instead of standard connected components, we grow regions
    # only if neighboring pixels have similar depth values.
    #
    # This helps separate touching objects with depth discontinuities.
    # ====================================================================

    labels = np.zeros((H, W), dtype=np.int32)
    visited = np.zeros((H, W), dtype=bool)

    depth_thresh = 0.02  # Maximum allowed depth difference (meters)
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
                for ny, nx in [
                    (cy - 1, cx),
                    (cy + 1, cx),
                    (cy, cx - 1),
                    (cy, cx + 1)
                ]:
                    if (
                        0 <= ny < H and
                        0 <= nx < W and
                        not visited[ny, nx] and
                        fg_clean[ny, nx]
                    ):
                        # Check depth consistency
                        if abs(depth_m[ny, nx] - depth_m[cy, cx]) < depth_thresh:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                            pixels.append((ny, nx))

            # Filter out small noisy regions
            if len(pixels) >= min_area:
                for py, px in pixels:
                    labels[py, px] = instance_id
                instance_id += 1

    # ====================================================================
    # Return final segmentation result
    # ====================================================================
    return SegmentationResult(
        labels=labels,
        num_objects=instance_id - 1,
    )