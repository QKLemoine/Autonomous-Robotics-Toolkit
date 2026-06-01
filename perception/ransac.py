from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class PlaneModel:
    """
    Represents a plane in 3D space using the implicit equation: n · X + d = 0.
    
    The plane is defined by:
    - n: unit normal vector (shape (3,), ||n|| = 1)
    - d: scalar offset
    
    For any point X on the plane: n · X + d = 0
    """
    n: np.ndarray  # (3,) unit normal vector
    d: float       # scalar offset

    def __post_init__(self):
        """Validate that the normal vector has correct shape and is normalized."""
        if self.n.shape != (3,):
            raise ValueError(f"Normal vector must have shape (3,), got {self.n.shape}")
        norm = np.linalg.norm(self.n)
        if abs(norm - 1.0) > 1e-6:
            raise ValueError(f"Normal vector must be unit length (||n||=1), got ||n||={norm:.6f}")

    def signed_distance(self, pts: np.ndarray) -> np.ndarray:
        """
        Compute signed distance from points to the plane.
        
        Positive values: point is on the side of the plane in the direction of n.
        Negative values: point is on the opposite side.
        Zero: point lies on the plane.
        
        Args:
            pts: Array of shape (N, 3) containing N 3D points.
        
        Returns:
            Array of shape (N,) containing signed distances.
        """
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Points must have shape (N, 3), got {pts.shape}")
        return pts @ self.n + self.d

    def distance(self, pts: np.ndarray) -> np.ndarray:
        """
        Compute absolute (unsigned) distance from points to the plane.
        
        This is the perpendicular distance from each point to the plane surface.
        
        Args:
            pts: Array of shape (N, 3) containing N 3D points.
        
        Returns:
            Array of shape (N,) containing non-negative distances.
        """
        return np.abs(self.signed_distance(pts))


def fit_plane_from_3pts(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Optional[PlaneModel]:
    """
    Fit a plane through 3 non-collinear points.
    
    Args:
        p1, p2, p3: Three 3D points, each with shape (3,).
    
    Returns:
        PlaneModel if the points are non-collinear, None otherwise.
    """
    # Validate input shapes
    for i, pt in enumerate([p1, p2, p3], 1):
        if pt.shape != (3,):
            raise ValueError(f"Point p{i} must have shape (3,), got {pt.shape}")

    #create two vectors on the plane
    v1 = p2 - p1
    v2 = p3 - p1

    #compute normal via cross product
    n = np.cross(v1, v2)

    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return None #Points are collinear or degenerate
    
    #normalize normal vecotr
    n = n / norm

    #compute offset
    d = -float(np.dot(n, p1))

    return PlaneModel(n=n.astype(np.float32), d=d)

    # ====================================================================




def refine_plane_svd(pts: np.ndarray) -> Optional[PlaneModel]:
    """
    Least-squares plane fit via SVD (Singular Value Decomposition).
    
    Args:
        pts: Array of shape (N, 3) containing N 3D points (at least 3 required).
    
    Returns:
        PlaneModel if successful, None if insufficient points or degenerate case.
    """
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Points must have shape (N, 3), got {pts.shape}")
    
    if pts.shape[0] < 3:
        return None
    
    # Center the points at the origin
    centroid = pts.mean(axis=0)
    X = pts - centroid
    
    # SVD: X = U @ S @ Vh
    # The normal is the right singular vector (vh) corresponding to smallest singular value
    # This is the direction of least variance, i.e., perpendicular to the best-fit plane
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    n = vh[-1]  # Last row of Vh corresponds to smallest singular value
    
    # Normalize (should already be unit, but ensure for numerical stability)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return None
    n = n / norm
    
    # Compute offset: d = -n · centroid
    d = -float(np.dot(n, centroid))
    
    return PlaneModel(n=n.astype(np.float32), d=float(d))


def ransac_plane(pts: np.ndarray,
                 iters: int = 2000,
                 inlier_thresh: float = 0.008,
                 min_inliers: int = 5000,
                 seed: int = 0) -> Tuple[Optional[PlaneModel], Optional[np.ndarray]]:
    """
    RANSAC (RANdom SAmple Consensus) algorithm for robust plane fitting.

    1. Randomly sample 3 points and fit a plane using fit_plane_from_3pts
    2. Count inliers (points within threshold distance of the plane)
    3. Keep the plane with the most inliers
    4. Refine the best plane using SVD on all inliers (refine_plane_svd)
    5. Return None if fewer than min_inliers are found
    
    Args:
        pts: Array of shape (N, 3) containing N 3D points (float32 recommended).
        iters: Maximum number of RANSAC iterations.
        inlier_thresh: Distance threshold for considering a point an inlier.
        min_inliers: Minimum number of inliers required for success.
        seed: Random seed for reproducibility.
    
    Returns:
        Tuple of (best_plane, inlier_mask) where:
        - best_plane: PlaneModel if successful, None otherwise
        - inlier_mask: Boolean array of shape (N,) indicating inliers, None if failed
    """
    # Validate input
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Points must have shape (N, 3), got {pts.shape}")
    
    if pts.shape[0] < 3:
        return None, None
    
    if inlier_thresh <= 0:
        raise ValueError(f"inlier_thresh must be positive, got {inlier_thresh}")
    
    if min_inliers < 3:
        raise ValueError(f"min_inliers must be at least 3, got {min_inliers}")

    # Create a random number generator with a fixed seed (ensures reproducability).
    rng = np.random.default_rng(seed)

    best_plane = None
    best_inlier_mask = None
    max_inliers = 0

    #how many total 3D points we have
    N = pts.shape[0]

    for _ in range(iters):
        #Randomly sample 3 distinct points; replace = False ensures distinctness
        idx = rng.choice(N, size=3, replace=False)
        p1, p2, p3 = pts[idx]

        #fit the points to plane from earlier function
        plane = fit_plane_from_3pts(p1, p2, p3)
        if plane is None:
            continue

        #Compute distances to all points
        distances = plane.distance(pts)

        #Determine inliers; if distance < threshold
        inlier_mask = distances < inlier_thresh
        num_inliers = np.sum(inlier_mask)

        #Update best model
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_plane = plane
            best_inlier_mask = inlier_mask

    # Check minimum inliers requirement
    if best_plane is None or max_inliers < min_inliers:
        return None, None

    # Refine using SVD on inliers
    refined_plane = refine_plane_svd(pts[best_inlier_mask])
    if refined_plane is None:
        return None, None

    # Recompute final inlier mask using refined plane
    final_distances = refined_plane.distance(pts)
    final_inlier_mask = final_distances < inlier_thresh

    return refined_plane, final_inlier_mask
    
    # ====================================================================