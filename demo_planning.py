"""
Quickstart Demo: Kinematic Path Optimization (RRT*)

This script generates a 2D configuration space with obstacles, 
runs the RRT* algorithm to find an optimal path, and visualizes 
both the search tree expansion and the recursively smoothed trajectory.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from planning.environment import KinematicEnvironment
from planning.rrt_star import RRTStarPlanner

def run_demo():
    print("Initializing Kinematic Environment...")
    # 10x10 coordinate space (from -5 to 5 on both axes)
    width, height = 10.0, 10.0
    
    # Create two walls to form an 'S' shaped maze
    obstacles = [
        [(-2.0, -5.0), (-1.0, -5.0), (-1.0, 2.0), (-2.0, 2.0)],  # Bottom-left wall
        [(1.0, -2.0), (2.0, -2.0), (2.0, 5.0), (1.0, 5.0)]       # Top-right wall
    ]
    
    env = KinematicEnvironment(width, height, obstacles)
    
    print("Initializing RRT* Planner...")
    start = (-4.0, -4.0)
    goal = (4.0, 4.0)
    
    # We use a smaller max_nodes for a fast demo, but enough to find a path
    planner = RRTStarPlanner(env, start, goal, max_nodes=1500)
    
    print("Computing asymptotically optimal path (this may take a moment)...")
    smoothed_path = planner.plan(min_nodes=500)

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_title("Kinematic Path Optimization (RRT*)")

    # Plot the obstacles
    for obs in obstacles:
        poly = Polygon(obs, closed=True, facecolor='black', alpha=0.8)
        ax.add_patch(poly)

    # Plot the RRT* Search Tree
    for node in planner.nodes:
        if node.parent is not None:
            ax.plot([node.x, node.parent.x], [node.y, node.parent.y], 
                    color='red', alpha=0.2, linewidth=0.5)

    # Plot the smoothed path
    if smoothed_path:
        px = [p[0] for p in smoothed_path]
        py = [p[1] for p in smoothed_path]
        ax.plot(px, py, color='gold', linewidth=3, label="Smoothed Optimal Path", zorder=5)
    
    # Plot Start and Goal markers
    ax.scatter(start[0], start[1], s=150, c='green', marker='o', label="Start", zorder=10)
    ax.scatter(goal[0], goal[1], s=150, c='blue', marker='*', label="Goal", zorder=10)

    ax.legend()
    os.makedirs("output", exist_ok=True)
    out_path = "output/demo_planning.png"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"\nSuccess! Search tree and optimal path saved to: {out_path}")

if __name__ == "__main__":
    run_demo()