"""
Visualization utilities for instance segmentation masks and RGB-D overlays.
"""

from pathlib import Path
from typing import Union
import numpy as np
import matplotlib.pyplot as plt

def label_to_color_image(labels: np.ndarray) -> np.ndarray:
    """Converts a 2D label map to a deterministic RGB color image."""
    H, W = labels.shape
    out = np.zeros((H, W, 3), dtype=np.float32)
    ids = np.unique(labels)
    ids = ids[ids != 0]   
    
    if ids.size == 0:
        return out

    for i in ids:
        r = ((int(i) * 123457) % 256) / 255.0
        g = ((int(i) * 234569) % 256) / 255.0
        b = ((int(i) * 345679) % 256) / 255.0
        out[labels == i, :] = (r, g, b)
    return out

def overlay(rgb: np.ndarray, labels: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Overlays color-coded instance labels over an RGB image."""
    rgb_f = rgb.astype(np.float32)
    if rgb_f.max() > 1.5:
        rgb_f /= 255.0
        
    color = label_to_color_image(labels)
    mask = labels > 0
    out = rgb_f.copy()
    out[mask] = (1.0 - float(alpha)) * rgb_f[mask] + float(alpha) * color[mask]
    return np.clip(out, 0.0, 1.0)

def save_side_by_side(
    rgb: np.ndarray, labels: np.ndarray, out_path: Union[str, Path],
    title_left: str = "Original", title_right: str = "Segmentation", alpha: float = 0.55
) -> None:
    """Saves a side-by-side comparison of the raw RGB frame and the segmentation mask."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ov = overlay(rgb, labels, alpha=alpha)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    
    axes[0].imshow(np.clip(rgb, 0, 255).astype(np.uint8) if rgb.max() > 1.5 else np.clip(rgb, 0, 1))
    axes[0].set_title(title_left)
    axes[0].axis('off')

    axes[1].imshow(ov)
    axes[1].set_title(f"{title_right} (instances={int(np.max(labels))})")
    axes[1].axis('off')

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)