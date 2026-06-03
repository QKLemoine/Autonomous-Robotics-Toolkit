"""
Utility functions for 2D geometry, spatial math, and Webots data parsing.
"""

import math
import csv
import numpy as np
from typing import List, Tuple, Optional
from localization.geometry import SE2, Point

# ==============================================================================
# Geometrical Math Helpers
# ==============================================================================

def line_intersection(p1: Point, p2: Point, p3: Point, p4: Point) -> Optional[Point]:
    """Computes the exact intersection point of two line segments (p1-p2 and p3-p4)."""
    denominator = ((p4.y - p3.y) * (p2.x - p1.x)) - ((p4.x - p3.x) * (p2.y - p1.y))

    if denominator == 0:
        return None  # Lines are parallel

    ua = (((p4.x - p3.x) * (p1.y - p3.y)) - ((p4.y - p3.y) * (p1.x - p3.x))) / denominator
    ub = (((p2.x - p1.x) * (p1.y - p3.y)) - ((p2.y - p1.y) * (p1.x - p3.x))) / denominator

    if 0 <= ua <= 1 and 0 <= ub <= 1:
        intersection_x = p1.x + ua * (p2.x - p1.x)
        intersection_y = p1.y + ua * (p2.y - p1.y)
        return Point(intersection_x, intersection_y)
        
    return None

def distance_between_points(p1: Point, p2: Point) -> float:
    """Computes the Euclidean distance between two points."""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def point_on_segment(p1: Point, p2: Point, p3: Point) -> bool:
    """Checks if point (p3) lies on the line segment defined by p1-p2."""
    return (min(p1.x, p2.x) <= p3.x <= max(p1.x, p2.x)) and \
           (min(p1.y, p2.y) <= p3.y <= max(p1.y, p2.y))

def line_segment_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """Boolean check for whether two line segments intersect."""
    p_intersect = line_intersection(p1, p2, p3, p4)
    if p_intersect is None:
        return False
    return point_on_segment(p1, p2, p_intersect) and point_on_segment(p3, p4, p_intersect)

def line_rectangle_intersect(p1: Point, p2: Point, rect_pose: SE2, rect_dim: List[float]) -> bool:
    """Checks if a line segment intersects with any boundary of a rotated rectangle."""
    rect_width, rect_height = rect_dim[0], rect_dim[1]
    a, b = rect_width / 2, rect_height / 2
    
    # Local bounding box corners
    corners_local = [Point(-a, -b), Point(a, -b), Point(a, b), Point(-a, b)]
    
    # Transform to world coordinates
    corners_w = [rect_pose.transform_point(pt) for pt in corners_local]
    
    # Check all four edges
    for i in range(4):
        if line_segment_intersect(p1, p2, corners_w[i], corners_w[(i + 1) % 4]):
            return True
    return False

def point_in_rectangle(point: Point, rect_pose: SE2, rect_dim: List[float]) -> bool:
    """Checks if a point is strictly inside a rotated rectangle."""
    rect_width, rect_height = rect_dim[0], rect_dim[1]
    a, b = rect_width / 2, rect_height / 2
    
    point_local = rect_pose.inverse().transform_point(point)
    return (-a < point_local.x < a) and (-b < point_local.y < b)

def diff_heading_rad(heading1: float, heading2: float) -> float:
    """Computes angular difference normalized to (-pi, pi]."""
    dh = heading1 - heading2
    while dh > math.pi: dh -= 2 * math.pi
    while dh <= -math.pi: dh += 2 * math.pi
    return dh

def pose_distance(pose1: SE2, pose2: SE2) -> float:
    """Computes a unified distance metric incorporating both translation and rotation."""
    diff_x = pose1.x - pose2.x
    diff_y = pose1.y - pose2.y
    diff_h = diff_heading_rad(pose1.h, pose2.h)
    return math.sqrt(diff_x**2 + diff_y**2 + diff_h**2)

def poses_within_dist(ref_pose: SE2, poses: List[SE2], distance: float) -> List[SE2]:
    """Filters a list of poses, returning only those within a set distance threshold."""
    return [pose for pose in poses if pose_distance(ref_pose, pose) < distance]

# ==============================================================================
# Webots Simulator Data Parsers
# ==============================================================================

def axis_angle_to_rotation_matrix(axis: List[float], angle: float) -> np.ndarray:
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    return np.array([
        [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c]
    ])

def rotation_matrix_to_euler_angles(R: np.ndarray) -> np.ndarray:
    sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    if sy > 1e-6:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

def read_marker_positions(wbt_file_path: str) -> List[Point]:
    """Parses visual marker translations from a .wbt file."""
    marker_positions = []
    reading_panel = False
    with open(wbt_file_path, "r") as file:
        for line in file:
            if "DirectionPanel" in line:
                reading_panel = True
                continue
            if reading_panel and "translation" in line:
                x, y = line.strip().split()[1:3]
                marker_positions.append(Point(float(x), float(y)))
                reading_panel = False
    return marker_positions

def read_walls(wbt_file_path: str) -> Tuple[List[SE2], List[List[float]]]:
    """Parses environmental boundaries (Walls) from a .wbt file."""
    wall_poses, wall_dimensions = [], []
    reading_wall = False
    
    with open(wbt_file_path, "r") as file:
        for line in file:
            if "Wall" in line:
                reading_wall = True
                continue
            if reading_wall:
                if "translation" in line:
                    x, y = map(float, line.strip().split()[1:3])
                    wall_poses.append(SE2(x, y, 0))
                elif "rotation" in line:
                    rotation = list(map(float, line.strip().split()[1:5]))
                    R = axis_angle_to_rotation_matrix(rotation[0:3], rotation[3])
                    h = rotation_matrix_to_euler_angles(R)[2]
                    wall_poses[-1].h = h
                elif "size" in line:
                    dim = list(map(float, line.strip().split()[1:4]))
                    wall_dimensions.append(dim)
                    reading_wall = False
                    
    return wall_poses, wall_dimensions