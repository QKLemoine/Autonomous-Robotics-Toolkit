"""
Quickstart Demo: Monte Carlo Localization (Particle Filter)

This script creates a synthetic 2D room, simulates a differential drive 
robot moving through it, and visualizes the particle filter converging 
on the true robot pose using simulated LiDAR rays.
"""

import math
import os
import numpy as np
import matplotlib.pyplot as plt

from localization.geometry import SE2
from localization.wall import Wall
from localization.lidar_sim import LidarSimulator
from localization.particle_filter import ParticleFilter

class MockEnvironment:
    """A synthetic environment that bypasses the need for Webots world files."""
    def __init__(self):
        self.x_min, self.x_max = 0.0, 10.0
        self.y_min, self.y_max = 0.0, 10.0

    def random_free_pose(self) -> SE2:
        x = np.random.uniform(self.x_min + 0.5, self.x_max - 0.5)
        y = np.random.uniform(self.y_min + 0.5, self.y_max - 0.5)
        h = np.random.uniform(-math.pi, math.pi)
        return SE2(x, y, h)

    def is_free(self, pose: SE2) -> bool:
        return (self.x_min < pose.x < self.x_max) and (self.y_min < pose.y < self.y_max)


def run_demo():
    print("Initializing Synthetic Localization Environment...")
    env = MockEnvironment()

    # Create a 10x10 enclosed room using the Wall class
    walls = [
        Wall(SE2(5, 10, 0), [10, 0.1]),          # Top wall
        Wall(SE2(5, 0, 0), [10, 0.1]),           # Bottom wall
        Wall(SE2(0, 5, math.pi/2), [10, 0.1]),   # Left wall
        Wall(SE2(10, 5, math.pi/2), [10, 0.1])   # Right wall
    ]

    # Initialize 10-ray LiDAR and 300 Particles
    lidar = LidarSimulator(walls, max_range=6.0, n_rays=10)
    pf = ParticleFilter(env, lidar, num_particles=300)

    # The ground truth pose of our "robot" (starts at x=2, y=2, facing right)
    true_pose = SE2(2.0, 2.0, 0.0)

    print("Simulating Robot Motion and Particle Filter Updates...")
    
    # We will step the simulation forward 8 times
    for step in range(8):
        # 1. Move the robot forward by 0.5 meters
        odometry = SE2(0.8, 0.0, 0.0)
        true_pose = true_pose.compose(odometry)

        # 2. Get exact LiDAR readings from the true pose
        true_lidar = lidar.read(true_pose)

        # 3. Pass odometry and sensor data into the Particle Filter
        pf.update(odometry, true_lidar)
        best_estimate = pf.compute_best_estimate()

        print(f"Step {step+1} | True: {true_pose} | Est: {best_estimate}")

    # Generate the final visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)
    ax.set_title("Monte Carlo Localization: Final Convergence")

    # Draw the walls
    for w in walls:
        w.plot(ax)

    # Plot the particle cloud
    px = [p.x for p in pf.particles]
    py = [p.y for p in pf.particles]
    ax.scatter(px, py, s=15, c='blue', alpha=0.3, label="Particle Cloud")

    # Plot True Pose vs Estimated Pose
    ax.scatter(true_pose.x, true_pose.y, s=150, c='green', marker='^', label="True Robot Pose")
    ax.scatter(best_estimate.x, best_estimate.y, s=150, c='red', marker='x', label="PF Estimate")

    ax.legend()
    os.makedirs("output", exist_ok=True)
    out_path = "output/demo_localization.png"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"\nSuccess! Convergence visualization saved to: {out_path}")

if __name__ == "__main__":
    run_demo()