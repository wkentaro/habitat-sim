import numpy as np


def pointcloud_from_depth(
    depth: np.ndarray, fx: float, fy: float, cx: float, cy: float,
) -> np.ndarray:
    assert depth.dtype.kind == "f", "depth must be float and have meter values"

    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = ~np.isnan(depth)
    z = np.where(valid, depth, np.nan)
    x = np.where(valid, z * (c - cx) / fx, np.nan)
    y = np.where(valid, z * (r - cy) / fy, np.nan)
    pc = np.dstack((x, y, z))

    return pc
