#!/usr/bin/env python

import numpy as np
import trimesh

from camera_transforms import opencv_to_opengl
from pointcloud_from_depth import pointcloud_from_depth


data = np.load("logs/scan_room/0000.npz")

intrinsic_matrix = data["intrinsic_matrix"]
rgb = data["rgb"]
depth = data["depth"]

fx, fy, cx, cy = intrinsic_matrix[[0, 1, 0, 1], [0, 1, 2, 2]]
pcd = pointcloud_from_depth(depth, fx=fx, fy=fy, cx=cx, cy=cy)

scene = trimesh.Scene(
    camera=trimesh.scene.Camera(resolution=(640, 480), focal=(fx, fy)),
    camera_transform=opencv_to_opengl(),
)
geometry = trimesh.PointCloud(
    vertices=pcd.reshape(-1, 3), colors=rgb.reshape(-1, 3)
)
scene.add_geometry(geometry)
scene.show()
