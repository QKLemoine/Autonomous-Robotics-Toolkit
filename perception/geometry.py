"""
3D Geometry and Projection utilities for point cloud processing.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy import ndimage as ndi
from .datatypes import CameraIntrinsics
from .ransac import ransac_plane, PlaneModel # Ensure ransac.py is in the same folder

def backproject(depth_m: np.ndarray, K: CameraIntrinsics) -> Tuple[np.ndarray, np.ndarray]:
    """
    Back-project a 2D depth image into a 3D point cloud.
    """
    valid = np.isfinite(depth_m)
    vs, us = np.nonzero(valid)
    z = depth_m[vs, us].astype(np.float32)
    
    x = (us.astype(np.float32) - float(K.cx)) * z / float(K.fx)
    y = (vs.astype(np.float32) - float(K.cy)) * z / float(K.fy)
    
    pts = np.stack((x, y, z), axis=1)
    idxs = np.stack((vs, us), axis=1)
    return pts, idxs


def extract_foreground_from_plane(
    depth_m: np.ndarray,
    K: CameraIntrinsics,
    plane: PlaneModel,
    max_range_m: float,
    margin_m: float = 0.01,
) -> np.ndarray:
    """
    Isolates foreground points sitting on the camera side of the estimated plane.
    """
    H, W = depth_m.shape
    valid = np.isfinite(depth_m) & (depth_m < max_range_m)
    fg = np.zeros((H, W), dtype=bool)
    
    if not np.any(valid):
        return fg

    pts, idxs = backproject(depth_m, K)
    within_range = pts[:, 2] < max_range_m
    pts = pts[within_range]
    idxs = idxs[within_range]

    s0 = float(plane.d) 
    sd = plane.signed_distance(pts) 

    camera_side = (sd * s0) > float(margin_m)
    fg_pixels = idxs[camera_side]
    fg[fg_pixels[:, 0], fg_pixels[:, 1]] = True
    return fg