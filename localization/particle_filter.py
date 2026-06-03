"""
Monte Carlo Localization (MCL) Particle Filter.
Estimates the pose of a robot in a known 2D environment using odometry and LiDAR.
"""

import math
import numpy as np
from typing import List

from localization.geometry import SE2
from localization.environment import RobotEnvironment
from localization.lidar_sim import LidarSimulator
# Assuming this will be in your refactored utils.py
from localization.utils import poses_within_dist

class ParticleFilter:
    """
    A robust 2D Particle Filter for state estimation.
    Uses differential drive kinematics for the prediction step and 
    ray-casted LiDAR simulation for the update step.
    """
    def __init__(
        self, 
        env: RobotEnvironment, 
        lidar_sim: LidarSimulator,
        num_particles: int = 200,
        motion_trans_sigma: float = 0.05,
        motion_head_sigma: float = 0.05,
        lidar_range_sigma: float = 0.2
    ):
        self.env = env
        self.lidar_sim = lidar_sim
        self.num_particles = num_particles
        
        # Noise parameters
        self.motion_trans_sigma = motion_trans_sigma
        self.motion_head_sigma = motion_head_sigma
        self.lidar_range_sigma = lidar_range_sigma

        # Initialize particles randomly across the free space
        self.particles = [self.env.random_free_pose() for _ in range(self.num_particles)]

    def _motion_update(self, odometry: SE2) -> List[SE2]:
        """Applies odometry with Gaussian noise to all particles (Prediction Step)."""
        new_particles = []
        for particle in self.particles:
            noisy_odo = odometry.add_noise(
                self.motion_trans_sigma, 
                self.motion_trans_sigma, 
                self.motion_head_sigma
            )
            new_particles.append(particle.compose(noisy_odo))
        return new_particles

    def _calculate_likelihood(self, robot_lidar: List[float], particle_lidar: np.ndarray) -> float:
        """
        Calculates the likelihood of a particle using a multivariate Gaussian 
        over the LiDAR ray distances.
        """
        max_range = self.lidar_sim.max_range
        variance = self.lidar_range_sigma ** 2
        sum_sq_diff = 0.0

        for r_val, p_val in zip(robot_lidar, particle_lidar):
            # Sanitize inputs: replace infinity with max_range
            if math.isinf(r_val):
                r_val = max_range
            if math.isinf(p_val):
                p_val = max_range

            diff = r_val - p_val
            sum_sq_diff += (diff ** 2)

        return math.exp(-sum_sq_diff / (2.0 * variance))

    def _compute_weights(self, motion_particles: List[SE2], robot_lidar: List[float]) -> List[float]:
        """Computes the importance weight for each particle based on sensor readings."""
        weights = []
        for particle in motion_particles:
            # Filter out impossible states
            if not self.env.is_free(particle):
                weights.append(1e-8) 
                continue
                
            # Simulate LiDAR from this particle's perspective
            particle_lidar = self.lidar_sim.read(particle)
            
            # Weight based on how closely simulation matches reality
            weight = self._calculate_likelihood(robot_lidar, particle_lidar)
            weights.append(weight)
            
        return weights

    def _resample(self, motion_particles: List[SE2], weights: List[float]) -> List[SE2]:
        """Stochastic Universal Sampling based on normalized particle weights."""
        weight_sum = float(sum(weights))
        if weight_sum == 0:
            # Fallback if all particles have 0 weight (e.g., completely lost)
            norm_weights = [1.0 / len(weights)] * len(weights)
        else:
            norm_weights = [w / weight_sum for w in weights]

        # Resample with replacement
        measured_particles = np.random.choice(
            motion_particles, 
            size=self.num_particles, 
            p=norm_weights
        ).tolist()
        
        return measured_particles

    def update(self, odometry: SE2, robot_lidar_measures: List[float]) -> None:
        """
        Executes one full iteration of the Particle Filter.
        1. Predict: Move particles according to odometry.
        2. Update: Weight particles against sensor readings.
        3. Resample: Draw a new set of particles based on weights.
        """
        motion_particles = self._motion_update(odometry)
        weights = self._compute_weights(motion_particles, robot_lidar_measures)
        self.particles = self._resample(motion_particles, weights)

    def compute_best_estimate(self) -> SE2:
        """
        Computes the most likely robot pose. 
        Iteratively expands a search radius to find the densest cluster of particles, 
        ignoring uniform outliers.
        """
        mean_pose = SE2.mean(self.particles)
        neighbor_distance = 0.1
        neighbor_poses = []
        
        # Expand search until we capture at least 5% of the particles
        while len(neighbor_poses) < self.num_particles * 0.05:
            neighbor_distance *= 2
            neighbor_poses = poses_within_dist(mean_pose, self.particles, neighbor_distance)
            
        return SE2.mean(neighbor_poses)