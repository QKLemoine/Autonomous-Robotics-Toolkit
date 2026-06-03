"""
Core SE(2) geometry components for 2D rigid body transformations.
"""

import math
import numpy as np

class Point:
    """A simple 2D point representation."""
    
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return f"[{self.x:.3f}, {self.y:.3f}]"

    def __repr__(self) -> str:
        return self.__str__()


class SE2:
    """
    Represents a 2D pose or transformation in SE(2).
    Handles translations and rotations in a 2D plane using homogeneous matrices.
    """
    
    def __init__(self, x: float, y: float, h: float):
        self.x = x
        self.y = y
        self.h = h
        self.c = math.cos(self.h)
        self.s = math.sin(self.h)

        # Homogeneous transformation matrix
        self.T = np.array([
            [self.c, -self.s, self.x],
            [self.s,  self.c, self.y],
            [0.0,     0.0,    1.0   ]
        ])

    def position(self) -> Point:
        """Returns the translation component as a Point object."""
        return Point(self.x, self.y)

    def transform_point(self, point: Point) -> Point:
        """Applies the SE(2) transformation to a 2D point."""
        new_x = self.c * point.x - self.s * point.y + self.x
        new_y = self.s * point.x + self.c * point.y + self.y
        return Point(new_x, new_y)

    def compose(self, other: 'SE2') -> 'SE2':
        """Composes this transformation with another SE(2) transformation."""
        new_x = self.c * other.x - self.s * other.y + self.x
        new_y = self.s * other.x + self.c * other.y + self.y
        new_h = self.h + other.h
        
        # Normalize angle to [-pi, pi]
        new_h = math.atan2(math.sin(new_h), math.cos(new_h))
        return SE2(new_x, new_y, new_h)

    def inverse(self) -> 'SE2':
        """Returns the inverse of the transformation."""
        new_x = -self.c * self.x - self.s * self.y
        new_y =  self.s * self.x - self.c * self.y
        new_h = -self.h
        return SE2(new_x, new_y, new_h)

    def add_noise(self, x_sigma: float, y_sigma: float, h_sigma: float) -> 'SE2':
        """Applies Gaussian noise to the transformation components."""
        new_x = self.x + np.random.normal(0, x_sigma)
        new_y = self.y + np.random.normal(0, y_sigma)
        new_h = self.h + np.random.normal(0, h_sigma)
        return SE2(new_x, new_y, new_h)

    @staticmethod
    def mean(pose_list: list['SE2']) -> 'SE2':
        """Computes the mean of multiple poses, using circular mean for orientation."""
        x_mean = np.mean([pose.x for pose in pose_list])
        y_mean = np.mean([pose.y for pose in pose_list])
        
        sin_mean = np.mean([math.sin(pose.h) for pose in pose_list])
        cos_mean = np.mean([math.cos(pose.h) for pose in pose_list])
        h_mean = math.atan2(sin_mean, cos_mean)
        
        return SE2(x_mean, y_mean, h_mean)

    def __str__(self) -> str:
        return f"[{self.x:.3f}, {self.y:.3f}, {math.degrees(self.h):.1f}°]"

    def __repr__(self) -> str:
        return self.__str__()