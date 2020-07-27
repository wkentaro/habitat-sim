import numpy as np
import trimesh.transformations as ttf


def opencv_to_opengl(transform=None):
    if transform is None:
        transform = np.eye(4)
    return transform @ ttf.rotation_matrix(np.deg2rad(-180), [1, 0, 0])


def coppeliasim_to_opengl(transform=None):
    if transform is None:
        transform = np.eye(4)
    return opencv_to_opengl(transform) @ ttf.rotation_matrix(
        np.deg2rad(180), [0, 0, 1]
    )


def opengl_to_opencv(transform=None):
    if transform is None:
        transform = np.eye(4)
    return transform @ ttf.rotation_matrix(np.deg2rad(180), [1, 0, 0])
