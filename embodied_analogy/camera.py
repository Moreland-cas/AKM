"""Camera.

Concepts:
    - Create and mount cameras
    - Render RGB images, point clouds, segmentation masks
"""

import sapien
import numpy as np
from PIL import Image, ImageColor

import trimesh
from sapien.utils.viewer import Viewer
from transforms3d.euler import mat2euler


def main():
    scene = sapien.Scene()
    scene.set_timestep(1 / 100.0)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    urdf_path = "/home/zby/Programs/Sapien/assets/100015/mobility.urdf"
    asset = loader.load(urdf_path)
    assert asset, "failed to load URDF."

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
    scene.add_point_light([1, 2, 2], [1, 1, 1])
    scene.add_point_light([1, -2, 2], [1, 1, 1])
    scene.add_point_light([-1, 0, 1], [1, 1, 1])

    # ---------------------------------------------------------------------------- #
    # Camera
    # ---------------------------------------------------------------------------- #
    near, far = 0.1, 100
    # width, height = 640, 480
    width, height = 1200, 1200

    # sapien中的相机坐标系为： forward(x), left(y) and up(z)
    # Camera to World 也就是相机在世界坐标系下的位姿
    # C2W 中的 t 是相机在世界坐标系下的坐标
    # C2W 中的 R 的三列从左到右依次是相机坐标系的 x,y,z 轴在世界坐标系下的坐标向量
    cam_pos = np.array([-2, -2, 3])
    forward = -cam_pos / np.linalg.norm(cam_pos)
    left = np.cross([0, 0, 1], forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    mat44[:3, 3] = cam_pos

    camera = scene.add_camera(
        name="camera",
        width=width,
        height=height,
        fovy=np.deg2rad(35),
        # fovy=np.deg2rad(60),
        near=near,
        far=far,
    )
    camera.entity.set_pose(sapien.Pose(mat44)) # C2W
    print("Intrinsic matrix\n", camera.get_intrinsic_matrix())

    scene.step()  # run a physical step
    scene.update_render()  # sync pose from SAPIEN to renderer
    camera.take_picture()  # submit rendering jobs to the GPU

    # 渲染过程
    rgba = camera.get_picture("Color")  # 获取RGBA图像，格式为[H, W, 4]
    rgba = rgba[..., :3]
    rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
    rgba_pil = Image.fromarray(rgba_img)
    rgba_pil.save("color.png")

    # 点云数据的大小为 H * W * 4，格式为(x, y, z, render_depth)，其中 render_depth < 1 的点是有效的
    position = camera.get_picture("Position")  # [H, W, 4]
    # OpenGL/Blender: y up and -z forward
    points_opengl = position[..., :3][position[..., 3] < 1]
    points_color = rgba[position[..., 3] < 1]
    # Model matrix is the transformation from OpenGL camera space to SAPIEN world space
    # camera.get_model_matrix() must be called after scene.update_render()!
    model_matrix = camera.get_model_matrix() # opengl camera 2 world
    points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]
    points_color = (np.clip(points_color, 0, 1) * 255).astype(np.uint8)
    trimesh.PointCloud(points_world, points_color).show()

    # Depth
    depth = -position[..., 2]
    depth_image = (depth * 1000.0).astype(np.uint16)
    depth_pil = Image.fromarray(depth_image)
    depth_pil.save("depth.png")
    
    depth_valid_mask = position[..., 3] < 1
    depth_valid_mask_pil = Image.fromarray(depth_valid_mask)
    depth_valid_mask_pil.show()

if __name__ == "__main__":
    main()
