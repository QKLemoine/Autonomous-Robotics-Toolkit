# Autonomous Robotics Toolkit

While modern robotics heavily relies on high-level APIs and black-box middleware, developing robust machine intelligence requires a deep understanding of algorithmic theory. This repository serves as a unified sandbox for implementing, testing, and benchmarking foundational robotics algorithms from first principles. 

By building custom differential drive kinematics, LiDAR simulation, and optimization logic purely in Python, this toolkit provides a transparent, highly customizable environment for evaluating perception and planning strategies.

## Core Modules

### 1. 3D Perception & Segmentation (`/perception`)
A tabletop instance segmentation pipeline designed for handling cluttered environments.
* **RANSAC Plane Fitting:** Robust background subtraction to identify and isolate supporting surfaces from raw RGB-D point cloud data.
* **Connected Components Labeling (CCL):** Spatial grouping of unorganized point clusters into distinct, trackable instance IDs.

### 2. Kinematic Path Planning (`/planning`)
A lightweight 2D path-planning library focused on optimization and obstacle avoidance.
* **RRT* (Rapidly-exploring Random Tree Star):** An asymptotically optimal sampling-based algorithm for navigating complex geometric spaces.
* **Recursive Trajectory Smoothing:** Post-processing algorithms to prune redundant nodes and reduce Euclidean distance for smooth, continuous robot motion.

### 3. Monte Carlo Localization (`/localization`)
A 2D state-estimation engine for navigating highly ambiguous and symmetrical environments.
* **Particle Filter Implementation:** Solves the kidnapped robot problem using an SE(2) state-space representation.
* **Sensor & Kinematic Simulation:** Custom forward kinematics for differential-drive configurations and ray-casting simulations for 10-beam LiDAR distance modeling.

### 4. Imitation Learning for Locomotion (`/learning`)
A continuous-control machine learning module built and evaluated in MuJoCo.
* **Behavioral Cloning:** Supervised learning implementation to train neural network policies on expert trajectory data.
* **Multi-Pedal Control:** Benchmarked on high-dimensional ant-locomotion tasks to evaluate the limits of behavioral cloning in stabilizing complex body dynamics.

## Installation & Setup

This toolkit uses Conda for strict dependency management to avoid system-level package conflicts.

1. Clone the repository:
   ```bash
   git clone [https://github.com/YourUsername/Autonomous-Robotics-Toolkit.git](https://github.com/YourUsername/Autonomous-Robotics-Toolkit.git)
   cd Autonomous-Robotics-Toolkit
   ```

2. Create and activate the virtual environment:
   ```bash
   conda create -n robotics-toolkit python=3.12 -y
   conda activate robotics-toolkit
   ```

3. Install the required mathematical and visualization libraries:
   ```bash
   conda install numpy scipy matplotlib -y
   ```

## Quickstart: Perception Demo

You can immediately test the 3D perception pipeline without needing to download massive external RGB-D datasets. The toolkit includes a synthetic data generator that builds a mathematical model of a cluttered tabletop and runs it through the segmentation engine.

```bash
python demo.py
```

*This will generate a synthetic depth map, isolate the tabletop using RANSAC, run depth-aware region growing, and save a color-coded instance mask to `output/demo_segmentation.png`.*

## Quickstart: Localization Demo

This toolkit includes a 2D Monte Carlo Localization (MCL) engine. To see the particle filter in action without needing to configure the Webots physics simulator, run the standalone synthetic demo:

```bash
python demo_localization.py
```

*This script generates a synthetic enclosed environment, simulates a differential drive robot moving through it, calculates ray-casted LiDAR distances, and applies Stochastic Universal Sampling. It outputs a visualization of the particles converging on the true pose to `output/demo_localization.png`.*