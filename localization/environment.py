"""
Simulation environment manager and differential drive kinematics.
"""

import os
import json
import math
import numpy as np

from localization.geometry import SE2
# Assuming these will be refactored into a standardized utils module shortly
from localization.utils import read_marker_positions, read_walls, point_in_rectangle

class RobotEnvironment:
    """
    Manages the 2D simulation boundaries, obstacle configurations, 
    and handles differential drive kinematic calculations.
    """
    def __init__(self, config_file_path: str, world_path: str = "worlds"):
        with open(config_file_path, "r") as file:
            configs = json.load(file)
            
        self.axle_length = configs["axle_length"]
        self.robot_radius = configs["robot_radius"]
        self.wheel_radius = configs["wheel_radius"]
        
        self.fov = configs["fov"]
        self.baseline = configs["camera_baseline"]
        self.T_r_c = SE2(*configs["camera_pose"])
        self.T_r_l = SE2(*configs["lidar_pose"])
        
        self.world_file = os.path.join(world_path, configs["world_file"])
        self.markers = read_marker_positions(self.world_file)
        self.wall_poses, self.wall_dimensions = read_walls(self.world_file)
        
        x_limits = configs["x_range"]
        y_limits = configs["y_range"]
        self.x_min = x_limits[0] + self.robot_radius
        self.x_max = x_limits[1] - self.robot_radius
        self.y_min = y_limits[0] + self.robot_radius
        self.y_max = y_limits[1] - self.robot_radius

    def random_free_pose(self) -> SE2:
        """Generates a random pose guaranteed to be free of obstacle collisions."""
        while True:
            x = np.random.uniform(self.x_min, self.x_max)
            y = np.random.uniform(self.y_min, self.y_max)
            h = np.random.uniform(-np.pi, np.pi)
            pose = SE2(x, y, h)
            if self.is_free(pose):
                return pose

    def is_free(self, pose: SE2) -> bool:
        """Checks if a given SE(2) pose intersects with any map boundaries or walls."""
        if pose.x < self.x_min or pose.x > self.x_max:
            return False
        if pose.y < self.y_min or pose.y > self.y_max:
            return False
            
        for wall_pose, wall_dim in zip(self.wall_poses, self.wall_dimensions):
            if point_in_rectangle(pose.position(), wall_pose, wall_dim):
                return False
        return True

    def diff_drive_kinematics(self, omega_l: float, omega_r: float) -> tuple[float, float]:
        """
        Converts wheel rotational speeds (rad/s) to robot linear (m/s) and angular velocities (rad/s).
        """
        v_l = omega_l * self.wheel_radius
        v_r = omega_r * self.wheel_radius
        
        v_x = (v_r + v_l) / 2.0
        omega = (v_r - v_l) / self.axle_length
        
        return v_x, omega

    def diff_drive_odometry(self, omega_l: float, omega_r: float, dt: float) -> SE2:
        """
        Computes the exact SE(2) odometry transform for a given time step.
        Utilizes exact integration for circular arcs.
        """
        v_x, omega = self.diff_drive_kinematics(omega_l, omega_r)
        
        # Handle straight line motion to avoid division by zero
        if math.fabs(omega) < 1e-5:
            return SE2(v_x * dt, 0, omega * dt)
            
        curve_radius = v_x / omega
        curve_angle = omega * dt
        
        dx = curve_radius * math.sin(curve_angle)
        dy = curve_radius * (1 - math.cos(curve_angle))
        dh = curve_angle
        
        return SE2(dx, dy, dh)