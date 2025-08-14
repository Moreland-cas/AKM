import cv2
import torch
import random
import numpy as np
import open3d as o3d
import graspnetAPI
import matplotlib.pyplot as plt
from typing import List
from PIL import Image
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import SpectralClustering
from pytorch_lightning import seed_everything

from akm.utility.constants import *

def initialize_napari():
    global ACTIVATE_NAPARI
    if ACTIVATE_NAPARI:
        import napari
        from qtpy.QtCore import QTimer

        with napari.gui_qt() as app:
            viewer = napari.Viewer()
            time_in_msec = 100
            QTimer().singleShot(time_in_msec, app.quit)
        viewer.close()

def clean_pc_np(
    points: np.ndarray,
    voxel_size=0.01,
    sor_k=20, sor_std=2.0,
    # The threshold value of the proportion of categories with fewer points to the total points
    clustering_threshold=0.1, 
    # Number of clustering iterations
    num_iterations=5,  
) -> np.ndarray:
    """
    Input: (N,3) np.ndarray
    Output: (M,3) np.ndarray (M <= N)
    """
    # 1) np.ndarray → o3d.PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 2) downsample
    pcd = pcd.voxel_down_sample(voxel_size)
    
    # 3) Statistical outlier removal
    _, ind_sor = pcd.remove_statistical_outlier(
        nb_neighbors=sor_k,
        std_ratio=sor_std
    )
    pcd = pcd.select_by_index(ind_sor)

    # 4) Multiple binary clustering filtering
    points_after_sor = np.asarray(pcd.points)
    for _ in range(num_iterations):
        if len(points_after_sor) < 2:
            # If the number of points is less than 2, clustering cannot continue
            break  

        # Use SpectralClustering for bipartite clustering
        clustering = SpectralClustering(n_clusters=2, affinity='nearest_neighbors')
        labels = clustering.fit_predict(points_after_sor)

        # Count the number of points in each cluster
        unique_labels, label_counts = np.unique(labels, return_counts=True)

        # Check the number of points in the two clusters
        if len(label_counts) == 2:
            if label_counts[0] / len(points_after_sor) < clustering_threshold:
                # If the first cluster has too few points, keep the second cluster.
                points_after_sor = points_after_sor[labels == 1]
            elif label_counts[1] / len(points_after_sor) < clustering_threshold:
                # If the second cluster has too few points, keep the first cluster.
                points_after_sor = points_after_sor[labels == 0]
            else:
                # If both clusters have enough points, keep all the points
                break
        else:
            # If there is only one cluster, exit the loop directly
            break
    return points_after_sor


def add_text_to_image(
    image: Image.Image, 
    text: str, 
    font_scale: float = 0.05, 
    text_color: tuple = (255, 255, 255), 
    position_ratio: tuple = (0.05, 0.05)
) -> Image.Image:
    """
    Adds text to an image. The font size and position are dynamically determined based on the image size, using the Times New Roman font.

    Args:
        image: PIL Image object
        text: The text to be added
        font_scale: The ratio of the font size to the image height (default 0.05, i.e. 5%)
        text_color: The text color, in RGB tuple format (R, G, B)
        position_ratio: The ratio of the text position to the image width and height (x_ratio, y_ratio), default (0.05, 0.05)

    Returns:
        PIL Image object containing the added text
    """
    new_image = image.copy()
    new_image = new_image.resize((3200, 2400), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(new_image)
    
    img_width, img_height = new_image.size
    
    # Dynamically calculate font size (based on the ratio of image height)
    font_size = int(img_height * font_scale)
    
    try:
        font_path = "/home/zby/Programs/AKM/scripts/times.ttf" 
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"Unable to load Times New Roman font: {e}, using default font")
        font = ImageFont.load_default()
    
    # Dynamically calculate text position (based on the ratio of image size)
    position = (int(img_width * position_ratio[0]), int(img_height * position_ratio[1]))
    
    # Draw text at the specified location
    draw.text(position, text, font=font, fill=text_color)
    return new_image


def pil_to_pygame(pil_image):
    pil_image = pil_image.convert("RGB")  
    return pygame.image.fromstring(np.array(pil_image).tobytes(), pil_image.size, "RGB")


def update_image_old(screen, rgb_pil, depth_pil, mask_pil):
    width1, height1 = rgb_pil.size
    width2, height2 = depth_pil.size
    width3, height3 = mask_pil.size

    total_width = width1 + width2 + width3
    max_height = max(height1, height2, height3)

    merged_image = Image.new("RGB", (total_width, max_height))

    merged_image.paste(rgb_pil, (0, 0))  
    merged_image.paste(depth_pil, (width1, 0))  
    merged_image.paste(mask_pil, (width1 + width2, 0)) 
    
    pygame_image = pil_to_pygame(merged_image)
    screen.fill((0, 0, 0))
    screen.blit(pygame_image, (0, 0))
    pygame.display.update()
    

def update_image(screen, rgb_pil):
    # Convert the PIL image to a format that can be displayed by pygame
    pygame_image = pil_to_pygame(rgb_pil)
    screen.fill((0, 0, 0))
    screen.blit(pygame_image, (0, 0))
    pygame.display.update()
    
    
def world_to_camera(world_points, Tw2c):
    """
    Converts a point in world coordinates to the camera coordinate system.

    Args:
        world_points (np.ndarray): 3D world coordinate points, shape (B, 3)
        Tw2c (np.ndarray): Camera extrinsic matrix (4x4), transform from world coordinates to camera coordinates

    Returns:
        np.ndarray: Point in the camera coordinate system, shape (B, 3)
    """
    # Expand world coordinate points to homogeneous coordinates
    world_points_homogeneous = np.hstack((world_points, np.ones((world_points.shape[0], 1))))  # (B, 4)
    # Batch conversion using extrinsic matrix
    camera_points_homogeneous = np.dot(world_points_homogeneous, Tw2c.T)  # (B, 4)
    # Convert to non-homogeneous coordinates
    camera_points = camera_points_homogeneous[:, :3]  # (B, 3)
    return camera_points


def camera_to_image(camera_points, K, image_width=None, image_height=None, normalized_uv=False):
    """
    Projects camera points onto the image plane and normalizes them to the range [0, 1].

    Args:
        camera_points (np.ndarray): Camera points, shape (B, 3)
        K (np.ndarray): Camera intrinsic parameter matrix, shape (3, 3)
        image_width (int): Image width
        image_height (int): Image height

    Returns:
        np.ndarray: Normalized pixel coordinates, shape (B, 2)
    """
    # Projection using the intrinsic matrix
    projected_points = np.dot(camera_points, K.T)  # (B, 3)

    # Convert to non-homogeneous pixel coordinates
    u = projected_points[:, 0] / projected_points[:, 2]  # (B,)
    v = projected_points[:, 1] / projected_points[:, 2]  # (B,)
    uv = np.vstack((u, v)).T  # (B, 2)
    
    depth = projected_points[:, 2]  # (B,)
    if normalized_uv:
        assert image_width is not None and image_height is not None, \
            "image_width and image_height must be provided when normalized_uv is True"
        u = u / image_width
        v = v / image_height

    return uv, depth  # (B, 2), (B, )


def camera_to_image_torch(camera_points, K, image_width=None, image_height=None, normalized_uv=False):
    """
    Projects camera points onto the image plane and normalizes them to the range [0, 1].

    Args:
        camera_points (torch.Tensor): Camera points, shape (B, 3)
        K (torch.Tensor): Camera intrinsic parameter matrix, shape (3, 3)
        image_width (int): Image width
        image_height (int): Image height

    Returns:
        torch.Tensor: Normalized pixel coordinates, shape (B, 2)
        torch.Tensor: Depth value, shape (B,)
    """
    # Projection using the intrinsic matrix
    projected_points = torch.matmul(camera_points, K.T)  # (B, 3)

    # Convert to non-homogeneous pixel coordinates
    u = projected_points[:, 0] / projected_points[:, 2]  # (B,)
    v = projected_points[:, 1] / projected_points[:, 2]  # (B,)
    uv = torch.stack((u, v), dim=1)  # (B, 2)

    depth = projected_points[:, 2]  # (B,)
    if normalized_uv:
        assert image_width is not None and image_height is not None, \
            "image_width and image_height must be provided when normalized_uv is True"
        u = u / image_width
        v = v / image_height
        uv = torch.stack((u, v), dim=1)  # (B, 2)

    return uv, depth  # (B, 2), (B,)


def world_to_image(world_points, K, Tw2c, image_width=None, image_height=None, normalized_uv=False):
    """
    Converts a world point to normalized pixel coordinates (u, v) [0, 1].

    Args:
        world_points (np.ndarray): World points, shape (B, 3)
        K (np.ndarray): Camera intrinsic parameter matrix, shape (3, 3)
        Tw2c (np.ndarray): Camera extrinsic parameter matrix (4x4), transform from world coordinates to camera coordinates
        image_width (int): Image width
        image_height (int): Image height

    Returns:
        np.ndarray: Normalized pixel coordinates, shape (B, 2)
    """
    # Convert to camera coordinate system
    camera_points = world_to_camera(world_points, Tw2c)
    # Project to the image plane and normalize
    uv, depth = camera_to_image(camera_points, K, image_width, image_height, normalized_uv)
    return uv # B, 2


def draw_points_on_image(image, uv_list, radius=1, normalized_uv=False):
    """
    Args:
        image: A PIL.Image object, or an np.array.
        uv_list: A list of (u, v) coordinates representing the points to be drawn.
        Returns a PIL image.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        
    # Get the width and height of an image
    width, height = image.size
    image_draw = image.copy()
    draw = ImageDraw.Draw(image_draw)
    
    for u, v in uv_list:
        # Convert normalized coordinates to pixel coordinates
        if normalized_uv:
            x = int(u * width)
            y = int(v * height)
        else:
            x = int(u)
            y = int(v)
        
        # Draw a red dot at position (x, y) (fill color is red)
        if radius == 1:
            draw.point((x, y), fill=(255, 0, 0))  # (255, 0, 0) 
        else:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 0))
    return image_draw


def pil_images_to_mp4(pil_images, output_filename, fps=30):
    """
    Save a sequence of PIL images as an MP4 video.

    Args:
        pil_images (list of PIL.Image): List of PIL image objects.
        output_filename (str): Path to the output video file.
        fps (int): Frames per second, controls the frame rate of the video.
    """
    width, height = pil_images[0].size
    fourcc = cv2.VideoWriter_fourcc(*'X264')  
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    for pil_img in pil_images:
        img_array = np.array(pil_img)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        video_writer.write(img_array)
    video_writer.release()
    print(f"Video saved as {output_filename}")
    

def image_to_camera(uv, depth, K, image_width=None, image_height=None, normalized_uv=False):
    """
    Converts normalized pixel coordinates and depth values to points in 3D space (camera coordinate system).

    Parameters:
        - uv (np.array): Shape (B, 2), [u, v] coordinates of each point, range [0, 1].
        - depth (np.array): Shape (B,), depth value of each point, unit: meters.
        - K (np.array): Camera intrinsic parameter matrix (3x3), including focal length and principal point coordinates.
        - image_width (int): Image width (unit: pixels).
        - image_height (int): Image height (unit: pixels).

    Return:
        - (np.array): A point cloud array with shape (B, 3), 3D coordinates (X, Y, Z) of each point.
    """
    u = uv[:, 0] 
    v = uv[:, 1] 

    # Calculate the actual pixel coordinates
    if normalized_uv:
        x_pixel = u * image_width
        y_pixel = v * image_height
    else:
        x_pixel = u
        y_pixel = v

    f_x = K[0, 0] 
    f_y = K[1, 1]  
    c_x = K[0, 2]  
    c_y = K[1, 2]  
    
    Z = depth  
    X = (x_pixel - c_x) * Z / f_x
    Y = (y_pixel - c_y) * Z / f_y
    
    # return pointcloud (B, 3)
    point_cloud = np.stack((X, Y, Z), axis=-1)
    return point_cloud


def depth_image_to_pointcloud(depth_image, mask, K):
    """
    Converts a depth image to a point cloud in the camera coordinate system.

    Args:
        - depth_image (np.array): Depth image, size (H, W), units: meters
        - K (np.array): Camera intrinsic parameter matrix (3x3), including focal length and principal point coordinates
        - mask (np.array, optional): Mask, size (H, W), Boolean type. If provided, only points where mask is True are retained.

    Returns:
        - pointcloud (np.array): Point cloud, size (N, 3), representing the 3D coordinates of each pixel in the camera coordinate system
    """
    assert depth_image.ndim == 2, "depth_image must be a 2D array"
    if mask is not None:
        assert mask.dtype == np.bool_, "mask must be a boolean array"
        
    f_x = K[0, 0]  
    f_y = K[1, 1]  
    c_x = K[0, 2]  
    c_y = K[1, 2]  

    # Generate pixel grid
    image_height, image_width = depth_image.shape
    u, v = np.meshgrid(np.arange(image_width), np.arange(image_height))

    # Calculate normalized coordinates
    x_normalized = (u - c_x) / f_x
    y_normalized = (v - c_y) / f_y

    Z = depth_image  
    X = x_normalized * Z
    Y = y_normalized * Z

    pointcloud = np.stack((X, Y, Z), axis=-1)

    if mask is not None:
        pointcloud = pointcloud[mask]

    pointcloud = pointcloud.reshape(-1, 3)
    return pointcloud


def camera_to_world(point_camera, extrinsic_matrix):
    """
    Converts a point in the camera coordinate system to the world coordinate system.

    Args:
        point_camera:
        np.array([N, 3]), point in the camera coordinate system
        extrinsic_matrix (np.array):
        Tw2c, extrinsic matrix (3, 4) or (4, 4), in the form [R | t]

    Returns:
        point_world np.array([N, 3])
    """
    # Extract the rotation matrix R and translation vector t from the extrinsic matrix
    Rw2c = extrinsic_matrix[:3, :3]  # (3, 3)
    tw2c = extrinsic_matrix[:3, 3]   # (3)
    tc2w = -Rw2c.T @ tw2c

    # Calculate the transformation from camera coordinate system to world coordinate system
    point_world = point_camera @ Rw2c + tc2w
    return point_world


def create_cylinder(start, end, radius=0.01):
    """
    Create a cylinder to simulate a line segment, given the start point, end point and radius
    """
    start = np.array(start)
    end = np.array(end)
    direction = end - start
    length = np.linalg.norm(direction)
    if length == 0:
        return None

    direction = direction / length
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
    cylinder.compute_vertex_normals()

    # Calculate the rotation and position of the cylinder
    center = (start + end) / 2 
    z_axis = np.array([0, 0, 1])

    # Calculate the rotation matrix
    if np.allclose(direction, z_axis):
        # If the direction is the Z axis, keep it unchanged
        R = np.eye(3)  
    elif np.allclose(direction, -z_axis):
        # If the direction is opposite to the Z axis, rotate 180 degrees around the X axis
        rotation_axis = np.array([1, 0, 0])
        rotation_angle = np.pi
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
    else:
        # Calculate the rotation axis and rotation angle
        rotation_axis = np.cross(z_axis, direction)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis) 
        rotation_angle = np.arccos(np.dot(z_axis, direction))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)

    # Rotating and moving the cylinder
    cylinder.rotate(R, center=cylinder.get_center())  
    cylinder.translate(center)
    return cylinder


def farthest_point_sampling(point_cloud, M):
    N = point_cloud.shape[0]
    # Randomly select an initial point
    indices = np.zeros(M, dtype=np.int32)
    farthest_index = np.random.randint(0, N)
    indices[0] = farthest_index

    # Calculate the distance from each point to the selected point
    distances = np.full(N, np.inf)

    for i in range(1, M):
        # Update the distance from each point to the nearest selected point
        dist = np.linalg.norm(point_cloud - point_cloud[farthest_index], axis=1)
        distances = np.minimum(distances, dist)

        # Select the point with the greatest distance
        farthest_index = np.argmax(distances)
        indices[i] = farthest_index
    return indices


def create_bbox(bbox, radius=0.01):
    """
    Create a rectangular bbox
    center: Center coordinates [x, y, z]
    half_x, half_y, half_z: Half-length, half-width, half-height
    color: Color
    """
    edges = [
        [0, 1], [0, 2], [2, 4], [4, 1],
        [3, 5], [5, 7], [7, 6], [6, 3],
        [0, 3], [1, 5], [4, 7], [2, 6]
    ]
    points = np.array(bbox)
    geometries = []

    for edge in edges:
        start = points[edge[0]]
        end = points[edge[1]]
        
        direction = end - start
        length = np.linalg.norm(direction)
        
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius,
            height=length,
            resolution=10,
            split=1
        )
        
        z_axis = np.array([0, 0, 1])
        direction_norm = direction / length if length > 0 else z_axis
        axis = np.cross(z_axis, direction_norm)
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm > 1e-6:
            axis = axis / axis_norm
            angle = np.arccos(np.dot(z_axis, direction_norm))
            rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        else:
            rotation_matrix = np.eye(3) if direction_norm[2] > 0 else o3d.geometry.get_rotation_matrix_from_axis_angle([1, 0, 0], np.pi)
        
        cylinder.rotate(rotation_matrix, center=(0, 0, 0))
        cylinder.translate(start + (end - start) / 2)
        
        geometries.append(cylinder)
    
    return geometries


def create_directed_cylinder(start_point, end_point, thickness=0.002, resolution=10, cylinder_per=0.7):
    """
    Creates a cylinder with an arrow pointing from start_point to end_point.

        start_point: (3,) Starting coordinates
        end_point: (3,) Ending coordinates
        thickness: float The thickness (radius) of the cylinder and cone
        resolution: int The resolution of the cylinder and cone

    Returns: A list of geometries containing the cylinder and cone, with the cone tip aligned at end_point
    Assuming the total length is 1, the cylinder height is 0.8, and the cone height is 0.2
    """
    start_point = np.asarray(start_point)
    end_point = np.asarray(end_point)
    direction = end_point - start_point
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return []
    
    direction = direction / length
    cylinder_height = length * cylinder_per
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=thickness, height=cylinder_height, resolution=resolution)
    z_axis = np.array([0, 0, 1])
    axis = np.cross(z_axis, direction)
    if np.linalg.norm(axis) > 1e-6:
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.dot(z_axis, direction))
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        cylinder.rotate(rotation_matrix)
    cylinder.translate(start_point + direction * cylinder_height / 2)
    
    cylinder.compute_vertex_normals()  
    cylinder.paint_uniform_color([0.5, 0.5, 0.5])  
    
    # Cone: height is 0.2 of total length, tip is aligned with end_point
    cone_height = length * (1 - cylinder_per)
    cone = o3d.geometry.TriangleMesh.create_cone(radius=thickness * 2, height=cone_height, resolution=resolution)
    if np.linalg.norm(axis) > 1e-6:
        cone.rotate(rotation_matrix)
    cone.translate(end_point - direction * cone_height)  
    cone.compute_vertex_normals()  
    cone.paint_uniform_color([0.5, 0.5, 0.5])  
    return [cylinder, cone]


def visualize_pc(points, point_size=1, colors=None, voxel_size=0.01, alpha=None, grasp=None, contact_point=None, post_contact_dirs=None, 
                 bboxes=None, bbox_radius=0.01, tracks_3d=None, tracks_3d_colors=None, pivot_point=None, joint_axis=None,
                 tracks_t_step=1, tracks_n_step=1, tracks_norm_threshold=1e-3, visualize_origin=False,
                 camera_intrinsic=None, camera_extrinsic=None, zoom_in_scale=2, online_viewer=False):
    """
    Visualize point clouds, grasps, contact points, bounding boxes, 3D tracks, and joints.

    points: List[np.ndarray], each element is an Nx3 point cloud coordinate.
    point_size: List[float] or float, the point size for each point cloud. A single value applies to all point clouds.
    colors: List[np.ndarray], each element is an Nx3 (RGB, 0-1) or Nx4 (RGBA) point cloud color, or None (defaults to green).
    alpha: List[float], the transparency of each point cloud (0-1). If None, defaults to 1.0.
    grasp: Grasp or GraspGroup object.
    contact_point: contact point coordinates.
    post_contact_dirs: list of contact point directions.
    tracks_3d: (T, N, 3) 3D tracks, where T is the time step and N is the number of tracks.
    tracks_3d_colors: (N, 3) color for each track (0-1), or None (random color). pivot_point: (3,) Joint pivot coordinates
    joint_axis: (3,) Joint axis direction vector
    tracks_t_step: int Time step downsampling interval (default 1, no downsampling)
    tracks_n_step: int Track number downsampling interval (default 1, no downsampling)
    tracks_norm_threshold: float Track segment length threshold; segments below this threshold are not drawn (default 1e-3)
    """
    geometries_to_draw = []
    
    assert isinstance(points, list)
    assert isinstance(colors, list)
    if point_size is None:
        point_size = [1.0] * len(points)
    elif not isinstance(point_size, list):
        point_size = [point_size] * len(points)
    if not isinstance(colors, list):
        colors = [colors] * len(points) if colors is not None else [None] * len(points)
    if alpha is None:
        alpha = [1.0] * len(points)
    elif not isinstance(alpha, list):
        alpha = [alpha] * len(points)
    
    if not (len(points) == len(colors) == len(point_size) == len(alpha)):
        raise ValueError("points, colors, point_size and alpha must be the same length")

    for i, (pts, clr, size, alp) in enumerate(zip(points, colors, point_size, alpha)):
        pts = np.asarray(pts)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points[{i}] must be Nx3")
        print(pts.shape, colors)
        
        pcd = o3d.t.geometry.PointCloud(o3d.core.Tensor(pts))
        if voxel_size is not None:
            try:
                pcd = pcd.voxel_down_sample(voxel_size)
                pts = pcd.point.positions.numpy()  
            except Exception as e:
                raise RuntimeError(f"pointcloud [{i}] downsample failed: {str(e)}")
            
        if clr is None:
            clr = np.zeros((pts.shape[0], 4))
            clr[:, 1] = 1.0 
            clr[:, 3] = alp  
        elif clr.ndim == 1:
            clr = np.tile(clr, (pts.shape[0], 1))
        else:
            clr = np.asarray(clr)
            if clr.shape[0] != pts.shape[0]:
                raise ValueError(f"points, colors, point_size and alpha must be the same length")
            if clr.shape[1] == 3:
                clr = np.concatenate([clr, np.ones((clr.shape[0], 1)) * alp], axis=1)
            elif clr.shape[1] != 4:
                raise ValueError(f"colors[{i}] must be Nx3 (RGB) or Nx4 (RGBA)")
        
        pcd.point.colors = o3d.core.Tensor(clr[:, :3])  
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLitTransparency"
        mat.point_size = size
        mat.base_color = np.concatenate([clr[0, :3] if clr.shape[0] > 0 else [0.0, 1.0, 0.0], [alp]])
        geometries_to_draw.append({"name": f"pcd_{i}", "geometry": pcd, "material": mat})
    
    # process grasp
    if isinstance(grasp, graspnetAPI.grasp.Grasp):
        grasp_o3d = grasp.to_open3d_geometry()
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.base_color = [0.0, 1.0, 0.0, 1.0] 
        mat.line_width = 2.0
        geometries_to_draw.append({"name": "grasp", "geometry": grasp_o3d, "material": mat})
    elif isinstance(grasp, list):
        for i, g in enumerate(grasp):
            grasp_o3d = g.to_open3d_geometry()
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            mat.base_color = [0.0, 1.0, 0.0, 1.0]
            mat.line_width = 2.0
            geometries_to_draw.append({"name": f"grasp_{i}", "geometry": grasp_o3d, "material": mat})
    elif isinstance(grasp, graspnetAPI.grasp.GraspGroup):
        grasp_o3ds = grasp.to_open3d_geometry_list()
        for i, grasp_o3d in enumerate(grasp_o3ds):
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            mat.base_color = [0.0, 1.0, 0.0, 1.0]
            mat.line_width = 2.0
            geometries_to_draw.append({"name": f"grasp_group_{i}", "geometry": grasp_o3d, "material": mat})
    
    # process contact_point
    if contact_point is not None:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(contact_point)
        sphere.paint_uniform_color([1, 0, 0])  
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        mat.base_color = [1.0, 0.0, 0.0, 1.0]
        mat.base_roughness = 1.0
        mat.base_metallic = 0.9
        mat.base_reflectance = 0.9
        geometries_to_draw.append({"name": "contact_point", "geometry": sphere, "material": mat})
        
    # process contact_point and post_contact_dirs
    if contact_point is not None and post_contact_dirs is not None:
        for i, post_contact_dir in enumerate(post_contact_dirs):
            start_point = contact_point
            end_point = start_point + post_contact_dir * 0.1
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector([start_point, end_point])
            line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
            line_set.paint_uniform_color([1, 0, 0])  
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            mat.base_color = [1.0, 0.0, 0.0, 1.0]
            mat.line_width = 2.0
            geometries_to_draw.append({"name": f"contact_dir_{i}", "geometry": line_set, "material": mat})
            
    # draw axis
    if visualize_origin:
        axis_length = 0.1
        axes = [
            ([0, 0, 0], [axis_length, 0, 0], [1, 0, 0]), 
            ([0, 0, 0], [0, axis_length, 0], [0, 1, 0]), 
            ([0, 0, 0], [0, 0, axis_length], [0, 0, 1]), 
        ]
        
        for i, (start, end, color) in enumerate(axes):
            cylinder = create_cylinder(start, end)
            if cylinder is not None:
                cylinder.paint_uniform_color(color)
                mat = o3d.visualization.rendering.MaterialRecord()
                mat.shader = "defaultLit"
                mat.base_color = np.concatenate([color, [1.0]])
                mat.base_roughness = 1.0
                mat.base_metallic = 0.9
                mat.base_reflectance = 0.9
                geometries_to_draw.append({"name": f"axis_{i}", "geometry": cylinder, "material": mat})
    
    # Processing bounding boxes
    if bboxes is not None:
        for i, bbox in enumerate(bboxes):
            bbox_line_sets = create_bbox(bbox, radius=bbox_radius)
            for j, line_set in enumerate(bbox_line_sets):
                mat = o3d.visualization.rendering.MaterialRecord()
                mat.shader = "defaultLit"
                apple_green_color = np.array([159, 191, 82]) / 255
                mat.base_color = np.concatenate([apple_green_color, [1.0]])
                mat.base_roughness = 1.0
                mat.base_metallic = 0.9
                mat.base_reflectance = 0.9
                geometries_to_draw.append({"name": f"bbox_{i}_{j}", "geometry": line_set, "material": mat})
    
    # Processing 3D trajectories
    if tracks_3d is not None:
        T, N, D = tracks_3d.shape
        if D != 3:
            raise ValueError("The last dimension of the trajectory must be 3 (x, y, z coordinates)")
        
        # If no color is provided, a random color is generated.
        if tracks_3d_colors is None:
            tracks_3d_colors = np.random.rand(N, 3) 
        elif tracks_3d_colors.ndim == 1:
            tracks_3d_colors = np.tile(tracks_3d_colors, (N, 1))
        else:
            tracks_3d_colors = np.asarray(tracks_3d_colors)
        
        # Downsampled tracks: Select the track by tracks_n_step and the time step by tracks_t_step
        if tracks_n_step is None:
            tracks_n_step = N - 1
        n_indices = np.linspace(0, N-1, tracks_n_step, dtype=int)
        if tracks_t_step is None:
            tracks_t_step = T - 1
        t_indices = np.linspace(0, T-1, tracks_t_step, dtype=int)
        print(t_indices)
        for i in n_indices: 
            points = tracks_3d[t_indices, i, :] 
            # Add a ball at the starting point of the trajectory
            if len(points) > 0:
                start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
                start_sphere.translate(points[0])
                start_sphere.paint_uniform_color(tracks_3d_colors[i])
                mat = o3d.visualization.rendering.MaterialRecord()
                mat.shader = "defaultLit"
                mat.base_color = np.concatenate([tracks_3d_colors[i], [1.0]])
                mat.base_roughness = 1.0
                mat.base_metallic = 0.9
                mat.base_reflectance = 0.9
                geometries_to_draw.append({"name": f"track_sphere_{i}", "geometry": start_sphere, "material": mat})
            
            # Make sure there are at least two points
            if len(points) > 1: 
                start_point = points[0]
                end_point = points[-1]
                track_vector = end_point - start_point
                track_norm = np.linalg.norm(track_vector)
                if track_norm < tracks_norm_threshold:
                    continue  
            
                # Draw a cylinder with an arrow for each pair of consecutive points
                for j in range(len(points)-1):
                    start_point = points[j]
                    end_point = points[j+1]
                    # Check line segment length
                    segment_vector = end_point - start_point
                    segment_norm = np.linalg.norm(segment_vector)
                    if segment_norm < tracks_norm_threshold:
                        continue  
                    
                    directed_cylinder = create_directed_cylinder(start_point, end_point, thickness=0.002)
                    for k, geom in enumerate(directed_cylinder):
                        geom.paint_uniform_color(tracks_3d_colors[i])
                        mat = o3d.visualization.rendering.MaterialRecord()
                        mat.shader = "defaultUnlit"
                        mat.base_color = np.concatenate([tracks_3d_colors[i], [1.0]])
                        geometries_to_draw.append({"name": f"track_cyl_{i}_{j}_{k}", "geometry": geom, "material": mat})
    
    # process joint (pivot_point and joint_axis)
    if pivot_point is not None and joint_axis is not None:
        pivot_point = np.asarray(pivot_point)
        joint_axis = np.asarray(joint_axis)
        if pivot_point.shape != (3,) or joint_axis.shape != (3,):
            raise ValueError("pivot_point and joint_axis must be vectors of shape (3,)")
        
        # normalize joint_axis
        joint_axis = joint_axis / np.linalg.norm(joint_axis)
        start_point = pivot_point
        end_point = pivot_point + joint_axis * 0.3
        
        # Draw joints using create_directed_cylinder
        directed_cylinder = create_directed_cylinder(start_point, end_point, thickness=0.01, resolution=20)
        for i, geom in enumerate(directed_cylinder):
            geom.paint_uniform_color([1, 0.5, 0]) 
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultLit"
            mat.base_color = [1.0, 0.5, 0.0, 1.0]
            mat.base_roughness = 1.0
            mat.base_metallic = 0.9
            mat.base_reflectance = 0.9
            geometries_to_draw.append({"name": f"joint_{i}", "geometry": geom, "material": mat})

    if online_viewer:
        o3d.visualization.draw(
            geometries_to_draw,
            bg_color=(1, 1, 1, 1.0), 
            show_skybox=False,
            width=800,
            height=600
        )
    else:
        if zoom_in_scale is None:
            zoom_in_scale = 1.
        camera_intrinsic = camera_intrinsic * zoom_in_scale
        camera_intrinsic[-1, -1] = 1
        renderer = o3d.visualization.rendering.OffscreenRenderer(width=800*zoom_in_scale, height=600*zoom_in_scale)
        renderer.setup_camera(
            intrinsic_matrix=camera_intrinsic,
            extrinsic_matrix=camera_extrinsic,
            intrinsic_width_px=800*zoom_in_scale,
            intrinsic_height_px=600*zoom_in_scale
        )
        scene = renderer.scene
        scene.set_background(np.array([1., 1., 1., 1.]))
        scene.set_lighting(scene.LightingProfile.SOFT_SHADOWS, [0, 0, 0])
    
        for item in geometries_to_draw:
            scene.add_geometry(item["name"], item["geometry"], item["material"])
            
        image = renderer.render_to_image()
        # shape: (image_height, image_width, 3), uint8
        rgb_image = np.asarray(image)  
        return rgb_image
        

def match_point_on_featmap(
    feat_1: torch.Tensor, 
    feat_2: torch.Tensor,
    uv_1: List[float],
):
    """_summary_

    Args:
        feat_1 (torch.Tensor): 1, C, H1, W1
        feat_2 (torch.Tensor): 1, C, H2, W2
        uv_1 (List[float]): [u, v], u and v are in [0, 1]
        visualize (bool): whether to visualize the matching probability map
    return:
        similarity_map (torch.Tensor): H2, W2, range [0, 1]
    """
    # Get the coordinates (u, v) of the specified point in the left image and convert them to pixel coordinates
    u, v = uv_1
    h1, w1 = feat_1.shape[2], feat_1.shape[3]
    left_pixel_x = int(u * w1)
    left_pixel_y = int(v * h1)

    # Get the features of the specified point in the left image
    left_point_feature = feat_1[0, :, left_pixel_y, left_pixel_x] # (feature_dim, )

    # Calculate the cosine similarity between the specified point in the left image and all points in the right image
    # Expand the features of the specified point in the left image to each position in the right image for comparison
    right_features = feat_2.view(feat_2.size(1), -1).T # shape: (seq_length, feature_dim)

    # Calculate cosine similarity
    similarity = F.cosine_similarity(left_point_feature.unsqueeze(0), right_features, dim=1) # 65536

    # Extract the coordinates of the matching points
    h2, w2 = feat_2.shape[2], feat_2.shape[3] # Get the height and width
    similarity_map = similarity.reshape(h2, w2)
    
    return similarity_map


def nms_selection(points_uv, probs, threshold=5 / 800., max_points=5):
    """
    Apply Non-Maximum Suppression (NMS) to the selected points to ensure they are not too close to each other.
    
    Args:
    - points_uv (np.array): List of points' (u, v) coordinates from the top-k selection, shape (num_points, 2).
    - probs (np.array): Corresponding probability values of the points.
    - threshold (float): Minimum distance between points (in normalized coordinates) for them to be kept.
    - max_points (int): Maximum number of points to return after NMS.
    
    Returns:
    - nms_points (np.array): The selected points after applying NMS.
    - nms_probs (np.array): Corresponding probability values of the NMS points.
    """
    selected_points = []
    selected_probs = []
    
    # Keep track of the remaining candidates
    candidates = list(range(len(points_uv)))
    
    while candidates and len(selected_points) < max_points:
        # Select the point with the highest probability
        best_idx = np.argmax(probs[candidates])
        best_point = points_uv[candidates[best_idx]]
        best_prob = probs[candidates[best_idx]]
        
        # Add the best point to the selected list
        selected_points.append(best_point)
        selected_probs.append(best_prob)
        
        # Remove candidates that are too close to the selected point
        distances = cdist([best_point], points_uv[candidates], metric='euclidean')
        candidates = [i for i, dist in zip(candidates, distances[0]) if dist >= threshold]
    
    return np.array(selected_points), np.array(selected_probs)
    
    
def reconstruct_mask(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """
    Returns the final filtered Boolean mask given an initial Boolean mask (mask1) and a filter mask (mask2).

    Parameters:
        - mask1: np.ndarray (H, W) -> Initial Boolean mask
        - mask2: np.ndarray (N,) -> Filter mask for the selected N points

    Returns:
        - new_mask: np.ndarray (H, W) -> Boolean mask containing only the pixels selected by mask2
    """
    assert mask1.dtype == np.bool_, "mask1 must be of type bool"
    assert mask2.dtype == np.bool_, "mask2 must be of type bool"
    
    # Get the indices of True in mask1
    indices = np.argwhere(mask1)  
    
    selected_indices = indices[mask2] 
    new_mask = np.zeros_like(mask1, dtype=bool)
    new_mask[selected_indices[:, 0], selected_indices[:, 1]] = True
    return new_mask


def tracksNd_variance_torch(tracks: torch.Tensor) -> torch.Tensor:
    """
    Computes the mean squared error of 3D trajectory data of shape T, M, 3 (PyTorch version).

    Parameters:
        tracks (torch.Tensor): A tensor of shape (T, M, Nd) representing the 2D/3D trajectory of M points at T time instants.

    Returns:
        torch.Tensor: The mean squared error of M points (scalar).
    """
    # Compute the variance of each point at T time steps (M, 3)
    pointwise_variance = torch.var(tracks, dim=0, unbiased=False) # Compute the 3D variance of M points

    # Compute the total variance of each point (average the 3D coordinates)
    pointwise_variance_mean = torch.mean(pointwise_variance, dim=1) # (M,)

    # Compute the mean variance of all points
    average_variance = torch.mean(pointwise_variance_mean)
    return average_variance


def tracksNd_variance_np(tracks: np.ndarray) -> float:
    """
    Computes the mean variance of 2D/3D trajectory data of shape (T, M, Nd) (NumPy version).

    Parameters:
        tracks (np.ndarray): An array of shape (T, M, Nd) representing the 2D/3D trajectory of M points at T time instants.

    Return:
        float: The mean variance of the M points (scalar).
    """
    # Compute the variance of each point at T time steps (M, Nd)
    pointwise_variance = np.var(tracks, axis=0, ddof=0) # Compute the Nd-dimensional variance of M points

    # Compute the total variance of each point (average the Nd-dimensional coordinates)
    pointwise_variance_mean = np.mean(pointwise_variance, axis=1) # (M,)

    # Compute the mean variance of all points
    average_variance = np.mean(pointwise_variance_mean)
    return average_variance


def farthest_scale_sampling(arr, M, include_first=True):
    """
    Select M points from a one-dimensional array, ensuring that the distances between them are as large as possible 
    (maximum and minimum distance sampling).

    Parameters:
        arr (np.ndarray): Input one-dimensional array of data.
        M (int): Number of data points to select.

    Return:
        np.ndarray: The selected representative data points.
    """
    arr = np.array(arr)
    N = len(arr)
    
    if M >= N:
        return arr  

    if include_first:
        selected_indices = [0]
    else:
        selected_indices = [np.random.randint(0, N)]
    
    # for the remaining points
    for _ in range(1, M):
        # Calculate the minimum distance from an unselected point to a selected point
        remaining_indices = list(set(range(N)) - set(selected_indices))
        min_distances = np.array([
            min(abs(arr[i] - arr[j]) for j in selected_indices)
            for i in remaining_indices
        ])
        
        # Select the point with the largest minimum distance
        next_index = remaining_indices[np.argmax(min_distances)]
        selected_indices.append(next_index)
    
    # Returns sorted sampling points for easy reading
    return np.sort(selected_indices)


def sample_array(arr, k):
    """
    Randomly samples k samples from an N x d array and returns a k x d array.
        :param arr: Input N x d numpy array
        :param k: Number of samples to be sampled
        :return: Returns a k x d array
    """
    # Make sure k is not greater than the number of rows in the array
    assert k <= arr.shape[0], "k cannot be greater than the number of rows in the array"
    
    # Randomly select k indices
    indices = np.random.choice(arr.shape[0], size=k, replace=False)
    
    # Extract the row by index and take the first two columns
    sampled_array = arr[indices]
    return sampled_array


def sample_points_within_bbox_and_mask(bbox, mask, N):
    """
    Samples N valid (u, v) points from the given bbox and mask.

    Parameters:
    - bbox: A bbox array of shape (4,) in the format [u_left, v_left, u_right, v_right].
    - mask: A Boolean array of size (H, W) indicating whether each pixel in the image is valid.
    - N: The number of points to sample.

    Returns:
    - A numpy array of shape (N, 2) representing the sampled (u, v) coordinates.

    Where N is the number of valid sampling points.
    """
    u_left, v_left, u_right, v_right = bbox
    bbox_u_range = u_right - u_left
    bbox_v_range = v_right - v_left

    # Calculate the step size so that the number of sampling points is close to N
    step_size_u = np.sqrt(bbox_u_range * bbox_v_range / N)
    step_size_v = step_size_u * bbox_v_range / bbox_u_range  

    # Calculate the number of sampling points
    num_u = int(np.floor(bbox_u_range / step_size_u))
    num_v = int(np.floor(bbox_v_range / step_size_v))

    # Generate uniformly sampled points
    u_coords = np.linspace(u_left, u_right, num_u)
    v_coords = np.linspace(v_left, v_right, num_v)

    u_grid, v_grid = np.meshgrid(u_coords, v_coords)

    u_flat = u_grid.flatten()
    v_flat = v_grid.flatten()
    points = np.stack([u_flat, v_flat], axis=1)

    # Check if the points are valid in the mask
    valid_points = points[mask[points[:, 1].astype(int), points[:, 0].astype(int)]]

    return valid_points


def napari_time_series_transform(original_data):
    """
    Convert the original time series data to the format required for Napari visualization.
        original_data: np.ndarray, (T, N, d)
        returned_data: np.ndarray, (T*N, 1+d), where 1 represents the time dimension
    """
    T = len(original_data)
    napari_data = []
    for i in range(T):
        tmp_data = original_data[i] # M, d
        tmp_data_with_t = np.concatenate([np.ones((tmp_data.shape[0], 1)) * i, tmp_data], axis=1) # M, (1+d)
        napari_data.append(tmp_data_with_t)
    napari_data = np.concatenate(napari_data, axis=0)
    return napari_data


def joint_data_to_transform_np(
    joint_type, # "prismatic" or "revolute"
    joint_dir, # unit vector np.array([3, ])
    joint_start,
    joint_state_ref2tgt # joint_state_tgt - joint_state_ref, a constant
):
    """
    Calculate the transformation matrix of the corresponding points on ref_frame and tgt_frame in the camera coordinate system
    """
    # Get T_ref2tgt based on joint_type and joint_dir and (joint_state2 - joint_state1)
    T_ref2tgt = np.eye(4)
    if joint_type == "prismatic":
        T_ref2tgt[:3, 3] = joint_dir * joint_state_ref2tgt
    elif joint_type == "revolute":
        Rref2tgt = R.from_rotvec(joint_dir * joint_state_ref2tgt).as_matrix()
        T_ref2tgt[:3, :3] = Rref2tgt
        T_ref2tgt[:3, 3] = joint_start @ (np.eye(3) - Rref2tgt.T) 
    else:
        assert False, "joint_type must be either prismatic or revolute"
    return T_ref2tgt


def joint_data_to_transform_torch(
    joint_type, # "prismatic" or "revolute"
    joint_dir, # unit vector torch.array([3, ])
    joint_start,
    joint_state_ref2tgt # joint_state_tgt - joint_state_ref, a constant
):
    # Get T_ref2tgt based on joint_type and joint_dir and (joint_state2 - joint_state1)
    T_ref2tgt = torch.eye(4, device=joint_dir.device)
    joint_dir = joint_dir / torch.norm(joint_dir)
    
    if joint_type == "prismatic":
        T_ref2tgt[:3, 3] = joint_dir * joint_state_ref2tgt
    elif joint_type == "revolute":
        _K = torch.zeros((3, 3), device=joint_dir.device)
        _K[0, 1] = -joint_dir[2]
        _K[0, 2] = joint_dir[1]
        _K[1, 0] = joint_dir[2]
        _K[1, 2] = -joint_dir[0]
        _K[2, 0] = -joint_dir[1]
        _K[2, 1] = joint_dir[0]
        
        theta = joint_state_ref2tgt  
        Rref2tgt = torch.eye(3, device=joint_dir.device) + torch.sin(theta) * _K + (1 - torch.cos(theta)) * (_K @ _K)
        T_ref2tgt[:3, :3] = Rref2tgt
        T_ref2tgt[:3, 3] = joint_start @ (torch.eye(3, device=joint_dir.device) - Rref2tgt.T) 
    else:
        assert False, "joint_type must be either prismatic or revolute"
    return T_ref2tgt


def set_random_seed(seed: int):
    """
    Sets the random seeds for Python, NumPy, and PyTorch to ensure reproducibility of experiments.
    :param seed: The random seed to set.
    """
    # Set the Python random seed.
    random.seed(seed)

    # Set the NumPy random seed.
    np.random.seed(seed)

    # Set the PyTorch random seed.
    torch.manual_seed(seed)

    # If using a GPU, also set the CUDA random seed.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # Set the seed for all GPUs.

        # To ensure reproducible results, set the deterministic CUDA algorithm.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    

def extract_tracked_depths(depth_seq, pred_tracks):
    """
    Args:
    depth_seq (np.ndarray):
    Depth map video, shape (T, H, W)
    pred_tracks (torch.Tensor):
    (T, M, 2)
    The coordinates of each point are in the form [u, v], with a range of [0, W) and [0, H].

    Returns:
    torch.Tensor: The depth value of each point, shape (T, M), the depth value of each point.
    """
    T, H, W = depth_seq.shape
    # _, M, _ = pred_tracks.shape

    # Ensure pred_tracks are integer coordinates
    u_coords = pred_tracks[..., 0].clamp(0, W - 1).long() # Horizontal coordinate u
    v_coords = pred_tracks[..., 1].clamp(0, H - 1).long() # Vertical coordinate v

    # Convert depth_seq to torch.Tensor
    depth_tensor = torch.from_numpy(depth_seq).cuda() # Shape: (T, H, W)

    # Extract depth values using advanced indexing
    depth_tracks = depth_tensor[torch.arange(T).unsqueeze(1), v_coords, u_coords] # Shape: (T, M)
    return depth_tracks.cpu()


def filter_tracks_by_visibility(pred_visibility, threshold=0.9):
    """
    pred_visibility: torch.tensor([T, M])
    Returns a boolean array of size M, representing the visibility of each point.
    If a point is visible in T * thre frames, it is considered visible.
    """
    T, M = pred_visibility.shape
    required_visible_count = int(threshold * T)
    
    # Count the number of visible frames for each cluster
    visible_count = pred_visibility.sum(dim=0)
    
    # Create a mask where the count is greater than or equal to the required count
    visible_mask = visible_count >= required_visible_count
    
    return visible_mask


def filter_tracks2d_by_depth_mask_seq(pred_tracks_2d, depthSeq_mask):
    """
    Filter out tracks that do not fall within the invalid depth_mask.
    Args:
        pred_tracks_2d: torch.tensor([T, M, 2])
        depthSeq_mask: np.array([T, H, W], dtype=np.bool_)
    """
    T = pred_tracks_2d.shape[0]
    # First read out the mask value of pred_tracks_2d at the corresponding position
    pred_tracks_2d_floor = torch.floor(pred_tracks_2d).long() # T M 2 (u, v)
    t_index = torch.arange(T, device=pred_tracks_2d.device).view(T, 1)  # (T, M)
    pred_tracks_2d_mask = depthSeq_mask[t_index, pred_tracks_2d_floor[:, :, 1], pred_tracks_2d_floor[:, :, 0]] # T M
    
    # Then intersect in the time dimension to get a mask of size M, retain the ones that are always True and return
    mask_and = np.all(pred_tracks_2d_mask, axis=0) # M
    pred_tracks_2d = pred_tracks_2d[:, mask_and, :]
    return pred_tracks_2d


def filter_tracks_by_consistency(tracks, threshold=0.1):
    """
    Filters smoothly varying tracks based on their 3D consistency.
        tracks: T, M, 3
    return:
        consist_mask: (M, ), boolen
    """
    # Get the number of time steps T and tracks M
    T, M, _ = tracks.shape

    # Initialize the consistency mask
    consis_mask = np.ones(M, dtype=bool)

    # Iterate over each track
    for m in range(M):
    # Calculate the variance of this track
        diffs = np.linalg.norm(tracks[1:T, m] - tracks[0:T-1, m], axis=1)

        # If any variance exceeds a threshold, mark it as inconsistent
        if np.any(diffs > threshold):
            consis_mask[m] = False
    return consis_mask


def classify_dynamic(mask, moving_points, static_points):
    """
    Classify points where the mask is True based on the nearest neighbor classification of point sets A and B.

    Parameters:
        mask (np.ndarray): A set of points of size (H, W) to which the matching network is applied. 
        Points where mask=True need to be classified.
        moving_points (np.ndarray): A set of points of size (M, 2) with coordinates (u, v).
        static_points (np.ndarray): A set of points of size (N, 2) with coordinates (u, v).

    Returns:
        dynamic_mask (np.ndarray): The classification result of size (H, W).
    """
    H, W = mask.shape

    tree_A = cKDTree(moving_points[:, [1, 0]])
    tree_B = cKDTree(static_points[:, [1, 0]])

    mask_indices = np.argwhere(mask) # N, 2 (v, u)

    distances_A, _ = tree_A.query(mask_indices)
    distances_B, _ = tree_B.query(mask_indices)

    dynamic_mask = np.zeros((H, W), dtype=np.uint8)

    A_closer = distances_A < distances_B # N
    B_closer = ~A_closer

    dynamic_mask[mask_indices[A_closer, 0], mask_indices[A_closer, 1]] = MOVING_LABEL
    dynamic_mask[mask_indices[B_closer, 0], mask_indices[B_closer, 1]] = STATIC_LABEL

    return dynamic_mask


def get_depth_mask(depth, K, Tw2c, height=0.02):
    """
    Returns a mask that marks points where depth is non-zero and where the reprojected world coordinate system is not on the ground.
        depth: np.array([H, W])
        height: By default, only points higher than 2 cm are retained.
    """
    H, W = depth.shape
    pc_camera = depth_image_to_pointcloud(depth, None, K) # H*W, 3
    pc_world = camera_to_world(pc_camera, Tw2c) # H*W, 3
    height_mask = (pc_world[:, 2] > height).reshape(H, W) # H*W
    depth_mask = (depth > 0.0) & height_mask
    return depth_mask


def get_depth_mask_seq(depth_seq, K, Tw2c, height=0.01):
    """
    Returns a mask that marks points where depth is non-zero and where the reprojected world coordinate system is not on the ground.
        depth: np.array([H, W])
        height: By default, only points higher than 1 cm are retained.
    """
    depth_mask_seq = depth_seq > 0
    for i in range(len(depth_seq)):
        depth_mask_seq[i] = get_depth_mask(depth_seq[i], K, Tw2c, height)
    return depth_mask_seq


def quantile_sampling(arr, M):
    """
    Samples quantiles from an array and returns representative data points.

    Parameters:
        arr (np.ndarray): The input 1D array of data.
        M (int): The number of data points to sample.

    Returns:
        np.ndarray: The sampled representative data points.
    """
    arr_sorted = np.sort(arr)
    quantiles = np.linspace(0, 1, M)
    indices = (quantiles * (len(arr_sorted) - 1)).astype(int)
    sampled_data = arr_sorted[indices]
    return sampled_data


def compute_normals(target_pc, k_neighbors=30):
    """
    Calculate the normal vector of the target point cloud.
        :param target_pc: NumPy array of shape (N, 3), target point cloud.
        :param k_neighbors: The number of neighbors to use when calculating the normal vector.
        :return: Normal vector array of shape (N, 3).
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(target_pc)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=k_neighbors))
    normals = np.asarray(pcd.normals)
    return normals


def find_correspondences(ref_pc, target_pc, max_distance=0.01):
    """
    Use a KD-Tree nearest neighbor search to find the nearest neighbors of the reference point cloud ref_pc in the target point cloud target_pc.
        :param ref_pc: NumPy array, shape (M, 3), reference point cloud
        :param target_pc: NumPy array, shape (N, 3), target point cloud
        :param max_distance: Maximum distance; points exceeding this distance will be ignored
        :return: Array of matching indices
    """
    tree = cKDTree(target_pc)
    distances, indices = tree.query(ref_pc, workers=4)
    
    if max_distance > 0.0:
        valid_mask = distances < max_distance
    else:
        valid_mask = np.ones(distances.shape, dtype=np.bool_)
    return indices, distances, valid_mask


def rotation_matrix_between_vectors(a, b):
    """
    Assume that R @ a = b, find R
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    
    # Calculate the rotation axis u and the rotation angle theta
    u = np.cross(a, b)  # The rotation axis is the cross product of a and b
    sin_theta = np.linalg.norm(u)
    cos_theta = np.dot(a, b)  
    u = u / sin_theta if sin_theta != 0 else u  

    # compute rotation matrix
    I = np.eye(3)
    u_cross = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]]) 

    R = I + np.sin(np.arccos(cos_theta)) * u_cross + (1 - cos_theta) * np.dot(u_cross, u_cross)
    return R


def compute_bbox_from_pc(points, offset=0):
    """
    Calculates the bounding box of a point cloud and expands the bbox by the offset.

    Parameters:
        - points: A numpy array of shape (N, 3) representing the coordinates of the point cloud.
        - offset: A floating point number representing the amount to expand the bbox.

    Returns:
        - bbox_min: A numpy array of shape (3,) representing the minimum corner of the bbox.
        - bbox_max: A numpy array of shape (3,) representing the maximum corner of the bbox.
    """
    bbox_min = np.min(points, axis=0) - offset
    bbox_max = np.max(points, axis=0) + offset
    return bbox_min, bbox_max


def sample_points_on_bbox_surface(bbox_min, bbox_max, num_samples):
    """
    Sample some points on the surface of the bbox.

    Parameters:
        - bbox_min: A numpy array of shape (3,) representing the minimum corner of the bbox.
        - bbox_max: A numpy array of shape (3,) representing the maximum corner of the bbox.
        - num_samples: An integer representing the number of points to sample.

    Return:
        - samples: A numpy array of shape (num_samples, 3) representing the coordinates of the sample points.
    """
    # The sampling points are evenly distributed on the 6 faces
    samples_per_face = num_samples // 6
    samples = []

    for i in range(6):
        if i < 2:
            # Front and back surfaces (x-axis direction)
            x = bbox_min[0] if i == 0 else bbox_max[0]
            x = np.ones(samples_per_face) * x
            y = np.random.uniform(bbox_min[1], bbox_max[1], samples_per_face)
            z = np.random.uniform(bbox_min[2], bbox_max[2], samples_per_face)
        elif i < 4:
            # Left and right surfaces (y-axis direction)
            y = bbox_min[1] if i == 2 else bbox_max[1]
            y = np.ones(samples_per_face) * y
            x = np.random.uniform(bbox_min[0], bbox_max[0], samples_per_face)
            z = np.random.uniform(bbox_min[2], bbox_max[2], samples_per_face)
        else:
            # Upper and lower surfaces (z-axis direction)
            z = bbox_min[2] if i == 4 else bbox_max[2]
            z = np.ones(samples_per_face) * z
            x = np.random.uniform(bbox_min[0], bbox_max[0], samples_per_face)
            y = np.random.uniform(bbox_min[1], bbox_max[1], samples_per_face)

        face_samples = np.column_stack((x, y, z))
        samples.append(face_samples)

    samples = np.vstack(samples)
    return samples


def normalize_cos_map_exp(cosine_similarity_map, sharpen_factor=5):
    """
    Normalizes the input cosine_similarity_map into a probability distribution.
    sharpen_factor: Used to adjust the relative size of the distribution. 
    A larger sharpen_factor makes the distribution sharper.
    """
    exp_map = np.exp(cosine_similarity_map * sharpen_factor)
    probability_map = exp_map / np.sum(exp_map)
    return probability_map


def concatenate_images(image1, image2, direction='h'):
    """
    Concatenate two PIL images either horizontally or vertically.

    Parameters:
        image1 (PIL.Image): The first image.
        image2 (PIL.Image): The second image.
        direction (str): 'horizontal' for horizontal concatenation, 'vertical' for vertical concatenation.

    Returns:
        PIL.Image: The concatenated image.
    """

    if direction == 'h':
        # Create a new image with width as the sum of both images' widths and the height of the larger one
        new_width = image1.width + image2.width
        new_height = max(image1.height, image2.height)
        new_image = Image.new('RGB', (new_width, new_height))

        # Paste the images into the new image
        new_image.paste(image1, (0, 0))
        new_image.paste(image2, (image1.width, 0))
    
    else:  # vertical
        # Create a new image with height as the sum of both images' heights and the width of the larger one
        new_width = max(image1.width, image2.width)
        new_height = image1.height + image2.height
        new_image = Image.new('RGB', (new_width, new_height))

        # Paste the images into the new image
        new_image.paste(image1, (0, 0))
        new_image.paste(image2, (0, image1.height))

    return new_image


def crop_nearby_points(point_clouds, contact3d, radius=0.1):
    '''Return a boolean mask indicating points close to a given point.'''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_clouds)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    # Find points within a given radius through KDTree search
    [k, idx, _] = pcd_tree.search_radius_vector_3d(contact3d, radius)
    
    # Create a Boolean mask, initialized to all False
    mask = np.zeros(len(point_clouds), dtype=bool)
    
    # Set the index position corresponding to the point within the given radius to True
    mask[idx] = True
    return mask


def fit_plane_normal(points):
    """
    Input a point cloud, fit a plane to it, and return the unit normal vector of the plane.
    point_cloud: np.array([N, 3])
    """
    # Calculate the mean of the points
    centroid = np.mean(points, axis=0)

    # Center the point cloud
    centered_points = points - centroid

    # Calculate the covariance matrix
    cov_matrix = np.cov(centered_points, rowvar=False)

    # Perform eigenvalue decomposition
    _, _, vh = np.linalg.svd(cov_matrix)

    # The normal vector is the last eigenvector of the eigenvalue decomposition
    normal_vector = vh[-1]

    # Normalize the normal vector
    normalized_normal_vector = normal_vector / np.linalg.norm(normal_vector)

    return centroid, normalized_normal_vector


def fit_plane_ransac(points, threshold=0.01, max_iterations=100, visualize=False):
    """
    Fits a plane to a point cloud using the RANSAC algorithm and returns the unit normal to the plane.
        points: np.array([N, 3]) - Input point cloud
        threshold: float - Maximum distance allowed before a point is considered an outlier
        max_iterations: int - Maximum number of RANSAC iterations
    """
    if len(points) < 3:
        print("Less than 3 points in fit_plane_ransac")
        return np.array([0, 0, 1])
    
    best_normal = None
    best_inliers_mask = None
    best_inliers_count = 0

    for _ in range(max_iterations):
        indices = np.random.choice(points.shape[0], 3, replace=False)
        sample_points = points[indices]

        _, normal_vector = fit_plane_normal(sample_points)
        dists = np.abs(np.dot(points - sample_points[0], normal_vector))

        inliers_mask = dists < threshold
        inliers_count = inliers_mask.sum()

        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_normal = normal_vector
            best_inliers_mask = inliers_mask

    print("num of inliers:", best_inliers_count, " / ", len(points))
    
    if visualize:
        v_points = points
        v_colors = np.zeros_like(points)
        v_colors[best_inliers_mask, 1] = 1
        v_colors[~best_inliers_mask, 0] = 1
        visualize_pc(
            points=v_points,
            colors=v_colors,
            contact_point=points.mean(axis=0),
            post_contact_dirs=[best_normal]
        )
    return best_normal


def remove_dir_component(vector, direction, return_normalized=False):
    """
    Returns the vector without the direction component.
        vector: np.array([3, ])
        direction: np.array([3, ])
    """
    direction_unit = direction / np.linalg.norm(direction)
    projection_length = np.dot(vector, direction_unit)
    projection = projection_length * direction_unit
    result_vector = vector - projection
    
    if return_normalized:
        result_vector = result_vector / np.linalg.norm(result_vector)
    
    return result_vector


def classify_open_close(tracks3d, moving_mask, visualize=False):
    """
    Determines whether the input track represents an open or closed object.
        tracks3d: T, N, 3, derived from the filtered tracks3d.
        moving_mask: N
    """
    static_mask = ~moving_mask
    tracks_3d_moving_c, tracks_3d_static_c = tracks3d[:, moving_mask, :], tracks3d[:, static_mask, :]
    moving_mean_start, moving_mean_end = tracks_3d_moving_c[0].mean(0), tracks_3d_moving_c[-1].mean(0)
    static_mean_start, static_mean_end = tracks_3d_static_c[0].mean(0), tracks_3d_static_c[-1].mean(0)
    
    # First, we determine whether the current track is open or closed over time based on the variance of the tracks3d_moving and tracks3d_static class centers.
    if np.linalg.norm(moving_mean_start - static_mean_start) > np.linalg.norm(moving_mean_end - static_mean_end):
        track_type = "close"
    else:
        track_type = "open"
        
    if visualize:
        viewer = napari.Viewer(title="open_close_classification", ndisplay=3)
        napari_tracks3d = np.copy(tracks3d)
        napari_tracks3d[..., 2] *= -1
        viewer.add_points(napari_tracks3d[0, moving_mask], name="moving_start", size=0.01, face_color="red")
        viewer.add_points(napari_tracks3d[0, static_mask], name="static_start", size=0.01, face_color="green")
        
        viewer.add_points(napari_tracks3d[0, moving_mask].mean(0), name="moving_mean_start", size=0.01, face_color="blue")
        viewer.add_points(napari_tracks3d[0, static_mask].mean(0), name="static_mean_start", size=0.01, face_color="blue")
        
        viewer.add_points(napari_tracks3d[-1, moving_mask], name="moving_end", size=0.01, face_color="red")
        viewer.add_points(napari_tracks3d[-1, static_mask], name="static_end", size=0.01, face_color="green")
        
        viewer.add_points(napari_tracks3d[-1, moving_mask].mean(0), name="moving_mean_end", size=0.01, face_color="blue")
        viewer.add_points(napari_tracks3d[-1, static_mask].mean(0), name="static_mean_end", size=0.01, face_color="blue")
        
        napari.run()
    
    return track_type


def reverse_joint_dict(joint_dict):
    joint_dict["joint_dir"] = -joint_dict["joint_dir"]
    joint_dict["joint_states"] = -joint_dict["joint_states"]
    

def make_bbox(bbox_extents):
    """Get the coordinates of the corners of a
    bounding box from the extents

    Parameters
    ----------
    bbox_extents : list (4xN)
        List of the extents of the bounding boxes for each of the N regions.
        Should be ordered: [min_row, min_column, max_row, max_column]

    Returns
    -------
    bbox_rect : np.ndarray
        The corners of the bounding box. Can be input directly into a
        napari Shapes layer.
    """
    minr = bbox_extents[0]
    minc = bbox_extents[1]
    maxr = bbox_extents[2]
    maxc = bbox_extents[3]

    bbox_rect = np.array(
        [[minr, minc], [maxr, minc], [maxr, maxc], [minr, maxc]]
    )
    bbox_rect = np.moveaxis(bbox_rect, 2, 0)

    return bbox_rect


def filter_dynamic(
    K, 
    query_depth, # H, W
    query_dynamic, # H, W
    ref_depths,  # T, H, W
    joint_type,
    joint_dir,
    joint_start,
    query_state,
    ref_states,
    depth_tolerance=0.01, # 1cm tolerance
    visualize=False
):
    """
    Based on the current joint state, verify all moving points and mark uncertain points as unknown.   
    """
    Tquery2refs = [
        joint_data_to_transform_np(
            joint_type=joint_type,
            joint_dir=joint_dir,
            joint_start=joint_start,
            joint_state_ref2tgt=ref_state - query_state,
    ) for ref_state in ref_states] 
    Tquery2refs = np.array(Tquery2refs) # T, 4, 4
        
    T, H, W = ref_depths.shape
    query_dynamic_updated = query_dynamic.copy()
    
    # Get the pixel coordinates of the current frame MOVING_LABEL
    moving_mask = query_dynamic == MOVING_LABEL
    if not np.any(moving_mask):
        return query_dynamic_updated        

    y, x = np.where(moving_mask) # N
    pc_moving = depth_image_to_pointcloud(query_depth, moving_mask, K)  # (N, 3)
    pc_moving_aug = np.concatenate([pc_moving, np.ones((len(pc_moving), 1))], axis=1)  # (N, 4)
    
    # Batch calculate the 3D coordinates of moving_pc in other frames
    pc_pred = np.einsum('tij,jk->tik', Tquery2refs, pc_moving_aug.T).transpose(0, 2, 1)[:, :, :3] # T, N, 3
    
    # Project to all frames
    uv_pred, depth_pred = camera_to_image(pc_pred.reshape(-1, 3), K) # T*N, 2
    uv_pred_int = np.floor(uv_pred.reshape(T, len(pc_moving), 2)).astype(int) # T, N, 2
    depth_pred = depth_pred.reshape(T, len(pc_moving)) # T, N
    
    # Perform a filter here and mark those moving points that do not get valid depth_obs as Unknown
    valid_idx = (uv_pred_int[..., 0] >= 0) & (uv_pred_int[..., 0] < W) & \
                (uv_pred_int[..., 1] >= 0) & (uv_pred_int[..., 1] < H) # T, N
                
    # If only one time frame is not observed, it is considered that no valid observation is obtained.
    valid_idx = valid_idx.all(axis=0) # N
    uv_pred_int = uv_pred_int[:, valid_idx] # T, M, 2
    depth_pred = depth_pred[:, valid_idx] # T, M
    
    T_idx = np.arange(T)[:, None]
    depth_obs = ref_depths[T_idx, uv_pred_int[..., 1], uv_pred_int[..., 0]]  # T, M
    
    # Calculate the error and update dynamic_mask, M, if there is a frame rejection, it is set to UNKNOWN
    unknown_mask = (depth_pred + depth_tolerance < depth_obs).any(axis=0)  # M
    query_dynamic_updated[y[valid_idx][unknown_mask], x[valid_idx][unknown_mask]] = UNKNOWN_LABEL
    
    if visualize:
        viewer = napari.view_image((query_dynamic != 0).astype(np.int32), rgb=False)
        viewer.title = "filter current dynamic mask using other frames"
        viewer.add_labels(query_dynamic.astype(np.int32), name='before filtering')
        viewer.add_labels(query_dynamic_updated.astype(np.int32), name='after filtering')
        napari.run()
    return query_dynamic_updated  


def line_to_line_distance(P1, d1, P2, d2):
    """
    Calculates the shortest distance between two lines in 3D space

    Parameters:
        P1 (np.array): Coordinates of a point on line 1, shape (3,)
        d1 (np.array): Unit direction vector of line 1, shape (3,)
        P2 (np.array): Coordinates of a point on line 2, shape (3,)
        d2 (np.array): Unit direction vector of line 2, shape (3,)

    Returns:
        Float: The shortest distance between the two lines
    """
    v = P2 - P1
    
    # Compute the cross product of the direction vector n = d1 × d2
    n = np.cross(d1, d2)
    
    # Compute the norm of the cross product ||n||
    n_norm = np.linalg.norm(n)
    
    if n_norm > 1e-10: 
        distance = np.abs(np.dot(v, n)) / n_norm
    else:  
        # If the two lines are parallel (n ≈ 0)
        # Calculate the component of v in the direction perpendicular to d1: v_perp = v - (v·d1) * d1
        v_parallel = np.dot(v, d1) * d1
        v_perp = v - v_parallel
        distance = np.linalg.norm(v_perp)
    return distance


def custom_linspace(a, b, delta):
    if delta <= 0:
        raise ValueError("delta must be positive")
    
    # Determine the increasing/decreasing direction
    direction = 1 if b >= a else -1
    diff = abs(b - a)
    
    # Calculate the maximum number of full steps
    n = int(np.floor(diff / delta))
    
    # Generate basic point sequence
    if n == 0:
        return np.array([b])
    else:
        points = a + direction * delta * np.arange(1, n+1)
        
        # Determine whether endpoint b needs to be added
        last = points[-1]
        if (direction == 1 and not np.isclose(last, b) and last < b) or \
           (direction == -1 and not np.isclose(last, b) and last > b):
            points = np.append(points, b)
        
        return points


def extract_pos_quat_from_matrix(T):
    """
    Extract position and quaternion from a 4x4 transformation matrix
    Input:
        T: Tph2w, np.array(4, 4), transformation matrix
    Output:
        pos: np.array([3, ]), position vector
        quat: np.array([4, ]), quaternion in (w, x, y, z) order
    """
    pos = T[:3, 3]
    R_matrix = T[:3, :3]
    # Return (x, y, z, w) order
    quat = R.from_matrix(R_matrix).as_quat(scalar_first=False) 
    quat = np.array([quat[3], quat[0], quat[1], quat[2]]) 
    return pos, quat


def dis_point_to_range(point, range):
    a = point
    b, c = range[0], range[1]
    assert b <= c
    if a >= b and a <= c:
        return 0
    elif a < b:
        return b - a
    else:
        return a - c
    
    
def distance_between_transformation(mat1, mat2):
    """
    Calculate the distance between two Transformations
    mat1: (4, 4)
    """
    def translation_distance(mat1, mat2):
        translation1 = mat1[:3, 3]
        translation2 = mat2[:3, 3]
        return np.linalg.norm(translation1 - translation2)
    
    def rotation_difference(mat1, mat2):
        R1 = mat1[:3, :3]
        R2 = mat2[:3, :3]
        angle_diff = np.arccos((np.trace(R1.T @ R2) - 1) / 2)
        return angle_diff
    
    trans_dist = translation_distance(mat1, mat2)
    rot_dist = rotation_difference(mat1, mat2)
    return trans_dist + rot_dist  


def numpy_to_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist() # Convert a NumPy array to a list
    elif isinstance(obj, np.float32):
        return float(obj) # Convert a NumPy float32 to a Python float
    elif isinstance(obj, np.ints32):
        return int(obj) # Convert a NumPy int32 to a Python int
    raise TypeError(f"Type {type(obj)} not serializable")
