#!/usr/bin/env python

import cv2
import habitat_sim
import imgviz
import magnum
import numpy as np
import path


def keyboard_interface(sim):
    while True:
        obs = sim.get_sensor_observations()

        cv2.imshow("rgb_sensor", obs["rgb_sensor"])
        key = chr(cv2.waitKey())
        if key == "q":
            break
        elif key == "1":
            sim.agents[0].scene_node.translate(magnum.Vector3([0.1, 0, 0]))
        elif key == "2":
            sim.agents[0].scene_node.translate(magnum.Vector3([-0.1, 0, 0]))
        elif key == "3":
            sim.agents[0].scene_node.translate(magnum.Vector3([0, 0, 0.1]))
        elif key == "4":
            sim.agents[0].scene_node.translate(magnum.Vector3([0, 0, -0.1]))
        elif key == "5":
            sim.agents[0].scene_node.translate(magnum.Vector3([0, 0.1, 0]))
        elif key == "6":
            sim.agents[0].scene_node.translate(magnum.Vector3([0, -0.1, 0]))
        elif key == "7":
            sim.agents[0].scene_node.rotate_x(magnum.Deg(15))
        elif key == "8":
            sim.agents[0].scene_node.rotate_x(magnum.Deg(-15))
        elif key == "9":
            sim.agents[0].scene_node.rotate_y(magnum.Deg(15))
        elif key == "0":
            sim.agents[0].scene_node.rotate_y(magnum.Deg(-15))
        elif key == "a":
            sim.agents[0].scene_node.rotate_z(magnum.Deg(15))
        elif key == "b":
            sim.agents[0].scene_node.rotate_z(magnum.Deg(-15))
        else:
            print(key)

        print(sim.agents[0].scene_node.translation)


def get_intrinsic_matrix(fovy, height, width):
    aspect_ratio = width / height
    fovx = 2 * np.arctan(np.tan(fovy * 0.5) * aspect_ratio)

    resolution = np.array([width, height])
    fov = np.array([fovx, fovy])
    fx, fy = resolution / (2 * np.tan(fov / 2))
    cx, cy = width / 2, height / 2

    K = np.eye(3, dtype=float)
    K[0, 0] = fx
    K[1, 1] = fy
    K[:2, 2] = cx, cy
    return K


def main():
    out_dir = path.Path("logs/scan_room")
    out_dir.makedirs_p()

    sim_config = habitat_sim.SimulatorConfiguration()
    sim_config.scene.id = (
        "/Users/wkentaro/data/replica/apartment_0/habitat/mesh_semantic.ply"
    )

    rgb_sensor = habitat_sim.SensorSpec()
    rgb_sensor.uuid = "rgb_sensor"
    rgb_sensor.resolution = [480, 640]

    depth_sensor = habitat_sim.SensorSpec()
    depth_sensor.uuid = "depth_sensor"
    depth_sensor.resolution = [480, 640]
    depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH

    semantic_sensor = habitat_sim.SensorSpec()
    semantic_sensor.uuid = "semantic_sensor"
    semantic_sensor.resolution = [480, 640]
    semantic_sensor.sensor_type = habitat_sim.SensorType.SEMANTIC

    fovy = np.radians(int(rgb_sensor.parameters["hfov"]))
    height, width = rgb_sensor.resolution
    intrinsic_matrix = get_intrinsic_matrix(fovy, height, width)

    agent_config = habitat_sim.AgentConfiguration()
    agent_config.sensor_specifications = [
        rgb_sensor,
        depth_sensor,
        semantic_sensor,
    ]

    sim = habitat_sim.Simulator(
        habitat_sim.Configuration(sim_config, [agent_config])
    )

    # keyboard_interface(sim)

    sim.agents[0].scene_node.translation = magnum.Vector3(
        0.214587, 1.12523, 5.56012
    )

    for i in range(90):
        sim.agents[0].scene_node.rotate_y(magnum.Deg(1))

        obs = sim.get_sensor_observations()
        rgb = obs["rgb_sensor"][:, :, :3][:, :, ::-1]
        depth = obs["depth_sensor"]
        semantic = obs["semantic_sensor"]
        extrinsic_matrix = np.asarray(sim.agents[0].scene_node.transformation)

        data = dict(
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=extrinsic_matrix,
            rgb=rgb,
            depth=depth,
            semantic=semantic,
        )
        npz_file = out_dir / f"{i:04d}.npz"
        np.savez_compressed(npz_file, **data)
        print(f"==> Saved to: {npz_file}")


if __name__ == "__main__":
    main()
