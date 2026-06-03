"""
Environmental Boundary Data Structures.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from localization.geometry import SE2

class Wall:
    """
    Represents a rectangular wall boundary in a 2D environment. 
    Computes and stores coordinate corners for rapid intersection testing.
    """
    def __init__(self, pose: SE2, dimensions: list[float]):
        self.pose = pose
        self.dimensions = dimensions
        
        w, h = dimensions[0] / 2.0, dimensions[1] / 2.0
        
        # Compute 4 corner points of the wall (local coordinates)
        self.top_right = [pose.x + w, pose.y + h]
        self.bottom_right = [pose.x + w, pose.y - h]
        self.bottom_left = [pose.x - w, pose.y - h]
        self.top_left = [pose.x - w, pose.y + h]

        # Line equation constants: Ax + By + C = 0
        self.top = []
        self.right = []
        self.bottom = []
        self.left = []
        
        # Automatically apply the orientation transform upon instantiation
        self.rotate()
        self.compute_line_equations()

    def rotate(self):
        """Applies the SE(2) orientation to the corner points."""
        angle = self.pose.h
        c, s = math.cos(angle), math.sin(angle)
        rotation_matrix = np.array([[c, -s], [s, c]])
        
        origin = np.array([self.pose.x, self.pose.y])
        
        def transform(point):
            return np.matmul(rotation_matrix, np.array(point) - origin) + origin

        self.top_right = transform(self.top_right)
        self.bottom_right = transform(self.bottom_right)
        self.bottom_left = transform(self.bottom_left)
        self.top_left = transform(self.top_left)

    def compute_line_equations(self):
        """Computes Standard Form line equations for the four edges."""
        def get_abc(p1, p2):
            return [p1[1] - p2[1], p2[0] - p1[0], -p2[0]*p1[1] + p2[1]*p1[0]]
            
        self.top = get_abc(self.top_right, self.top_left)
        self.right = get_abc(self.bottom_right, self.top_right)
        self.bottom = get_abc(self.bottom_left, self.bottom_right)
        self.left = get_abc(self.top_left, self.bottom_left)

    def plot(self, ax=None):
        """Visualizes the bounding box using Matplotlib."""
        if ax is None:
            ax = plt.gca()
            
        def plot_line(A, B, C, x_range):
            x = np.linspace(x_range[0], x_range[1], 100)
            y = (-A * x - C) / B
            ax.plot(x, y, color='black', linewidth=2)

        x_vals = [self.top_left[0], self.top_right[0], self.bottom_left[0], self.bottom_right[0]]
        x_min, x_max = min(x_vals), max(x_vals)

        plot_line(*self.top, (x_min, x_max))
        plot_line(*self.right, (x_min, x_max))
        plot_line(*self.bottom, (x_min, x_max))
        plot_line(*self.left, (x_min, x_max))