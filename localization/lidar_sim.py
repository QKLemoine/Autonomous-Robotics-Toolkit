"""
LiDAR measurement simulation for 2D rigid bodies.
Computes ray-casting intersections against environmental obstacles.
"""

import math
import numpy as np
from typing import List

from localization.geometry import SE2, Point
# Assuming these will be in your refactored utils/wall files
from localization.wall import Wall
from localization.utils import line_rectangle_intersect, line_intersection, distance_between_points

class LidarSimulator:
    """
    Simulates a 2D LiDAR sensor by casting rays and computing intersections
    with rectangular boundaries in the environment.
    """
    def __init__(self, walls: List[Wall], max_range: float, n_rays: int):
        self.walls = walls
        self.max_range = max_range
        self.n_rays = n_rays
        self.resolution = int(360 / n_rays)
        self.measurements = math.inf * np.ones(self.n_rays)

    def read(self, pose: SE2) -> np.ndarray:
        """
        Simulates LiDAR readings from a specific SE(2) pose.
        
        Args:
            pose: The current SE(2) pose of the sensor.
            
        Returns:
            An array of distance measurements. Returns math.inf for rays 
            that reach max_range without detecting an obstacle.
        """
        # Reset measurements (Webots returns inf for no detection)
        self.measurements = math.inf * np.ones(self.n_rays)
        start_p = pose.position()
        
        for i in range(self.n_rays):
            # Webots LiDAR measurement angles start offset by resolution/2
            angle_offset_deg = (i + 0.5) * self.resolution
            ray_angle = pose.h + math.radians(angle_offset_deg)
            
            # Endpoint of the lidar ray at maximum range
            end_p = Point(
                start_p.x + self.max_range * math.cos(ray_angle),
                start_p.y + self.max_range * math.sin(ray_angle)
            )
            
            min_dist = math.inf
            
            for wall in self.walls:
                # Fast AABB/Bounding-box check before performing detailed line intersection
                if line_rectangle_intersect(start_p, end_p, wall.pose, wall.dimensions):
                    
                    corners = [
                        Point(wall.top_right[0], wall.top_right[1]),
                        Point(wall.bottom_right[0], wall.bottom_right[1]),
                        Point(wall.bottom_left[0], wall.bottom_left[1]),
                        Point(wall.top_left[0], wall.top_left[1])
                    ]
                    
                    # Check intersection against all 4 edges of the wall
                    for j in range(4):
                        p1 = corners[j]
                        p2 = corners[(j + 1) % 4]
                        
                        intersect_p = line_intersection(start_p, end_p, p1, p2)
                        
                        if intersect_p is not None:
                            dist = distance_between_points(start_p, intersect_p)
                            if dist < min_dist:
                                min_dist = dist
                                
            self.measurements[i] = min_dist

        return self.measurements