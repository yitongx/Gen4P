"""Visualization utilities for D4RT-query."""

import torch
import numpy as np
from typing import Optional
import matplotlib.cm as cm


def visualize_depth(
    depth: torch.Tensor,
    colormap: str = 'magma',
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
) -> np.ndarray:
    """Visualize depth map with colormap.

    Args:
        depth: (H, W) depth map
        colormap: Matplotlib colormap name
        min_depth: Minimum depth for normalization
        max_depth: Maximum depth for normalization

    Returns:
        Colored depth visualization (H, W, 3) in [0, 255] uint8
    """
    depth = depth.detach().float().cpu().numpy() if isinstance(depth, torch.Tensor) else depth

    # Handle invalid values
    valid = np.isfinite(depth) & (depth > 0)

    if min_depth is None:
        min_depth = depth[valid].min() if valid.any() else 0

    if max_depth is None:
        max_depth = depth[valid].max() if valid.any() else 1

    # Normalize
    depth_norm = (depth - min_depth) / (max_depth - min_depth + 1e-6)
    depth_norm = np.clip(depth_norm, 0, 1)

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colored = cmap(depth_norm)[..., :3]  # (H, W, 3)

    # Mark invalid regions
    colored[~valid] = 0

    # Convert to uint8
    colored = (colored * 255).astype(np.uint8)

    return colored
