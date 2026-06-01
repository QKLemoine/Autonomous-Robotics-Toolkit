"""
Quickstart Demo: 3D Point Cloud Segmentation

This script generates a synthetic depth map of a cluttered tabletop,
runs it through the perception pipeline, and outputs a visualization.
"""

import numpy as np
import os
from perception.datatypes import CameraIntrinsics
from perception.ransac import PlaneModel
from perception.segment import TabletopSegmenter
from perception.visualization import save_side_by_side

def generate_synthetic_scene():
    """Generates a fake depth map and RGB image of a table with objects."""
    H, W = 480, 640
    
    # 1. Background / Floor (2.5 meters away)
    depth_m = np.full((H, W), 2.5, dtype=np.float32)
    rgb = np.full((H, W, 3), 50, dtype=np.uint8) # Dark gray background
    
    # 2. Tabletop (1.0 meter away)
    depth_m[200:480, 100:540] = 1.0
    rgb[200:480, 100:540] = [139, 69, 19] # Brown table
    
    # 3. Object 1: Isolated Box (0.8 meters away)
    depth_m[250:320, 200:270] = 0.8
    rgb[250:320, 200:270] = [200, 50, 50] # Red box
    
    # 4. Object 2: Touching Box A (0.75 meters away)
    depth_m[350:420, 300:380] = 0.75
    rgb[350:420, 300:380] = [50, 200, 50] # Green box
    
    # 5. Object 3: Touching Box B (0.70 meters away - distinct depth discontinuity)
    depth_m[350:420, 380:450] = 0.70
    rgb[350:420, 380:450] = [50, 50, 200] # Blue box

    # Standard VGA camera intrinsics
    K = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    
    return rgb, depth_m, K

def run_demo():
    print("Generating synthetic tabletop scene...")
    rgb, depth_m, K = generate_synthetic_scene()
    
    print("Initializing Tabletop Segmenter...")
    segmenter = TabletopSegmenter(min_object_size=500)
    
    # Note: In a full run, RANSAC would generate this plane dynamically.
    # For this demo, we mock the ideal table plane (facing the camera directly).
    mock_table_plane = PlaneModel(n=np.array([0.0, 0.0, 1.0]), d=-1.0)
    
    print("Extracting foreground and segmenting instances...")
    # 1. Extract points above the table
    fg_mask = depth_m < 0.95 # Simple mock for extract_foreground
    
    # 2. Run the depth-aware region growing
    result = segmenter.segment_depth_aware(fg_mask, depth_m, depth_thresh=0.03)
    
    print(f"Success! Detected {result.num_objects} unique objects.")
    
    # 3. Visualize and save
    os.makedirs("output", exist_ok=True)
    out_path = "output/demo_segmentation.png"
    save_side_by_side(rgb, result.labels, out_path, title_left="Raw Synthetic RGB")
    print(f"Visualization saved to: {out_path}")

if __name__ == "__main__":
    run_demo()