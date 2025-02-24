import os
import pygame
import torch
import random
from PIL import Image
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import open3d as o3d
import graspnetAPI
from pytorch_lightning import seed_everything
from featup.util import pca, remove_axes
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from typing import List, Union
import torch.nn.functional as F
import matplotlib.cm as cm
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree

from embodied_analogy.utility.constants import *
from embodied_analogy.visualization.vis_tracks_2d import vis_tracks2d_napari

def initialize_napari():
    # there are 2 ways:
    # 1) need to manually close
    # import napari
    # viewer = napari.Viewer()
    # napari.run()
    # 2) automatically close
    import napari
    from qtpy.QtCore import QTimer

    with napari.gui_qt() as app:
        viewer = napari.Viewer()
        time_in_msec = 100
        QTimer().singleShot(time_in_msec, app.quit)
    viewer.close()
    
def pil_to_pygame(pil_image):
    pil_image = pil_image.convert("RGB")  # 转换为 RGB 格式
    return pygame.image.fromstring(np.array(pil_image).tobytes(), pil_image.size, "RGB")

# 更新图像函数
def update_image_old(screen, rgb_pil, depth_pil, mask_pil):
    # 获取每个图像的尺寸
    width1, height1 = rgb_pil.size
    width2, height2 = depth_pil.size
    width3, height3 = mask_pil.size

    # 计算最终合并图像的尺寸：宽度是三个图像的宽度之和，高度取最大值
    total_width = width1 + width2 + width3
    max_height = max(height1, height2, height3)

    # 创建一个空白图像来存放合并后的图像
    merged_image = Image.new("RGB", (total_width, max_height))

    # 将三个图像粘贴到合并图像上
    merged_image.paste(rgb_pil, (0, 0))  # (0, 0) 是第一个图像的位置
    merged_image.paste(depth_pil, (width1, 0))  # 第二个图像紧接着第一个
    merged_image.paste(mask_pil, (width1 + width2, 0))  # 第三个图像紧接着第二个
    
    # 将PIL图片转换为pygame可以显示的格式
    pygame_image = pil_to_pygame(merged_image)
    
    # 填充背景
    screen.fill((0, 0, 0))
    
    # 绘制图片
    screen.blit(pygame_image, (0, 0))
    
    # 更新屏幕显示
    pygame.display.update()

def update_image(screen, rgb_pil):
    # 将PIL图片转换为pygame可以显示的格式
    pygame_image = pil_to_pygame(rgb_pil)
    
    # 填充背景
    screen.fill((0, 0, 0))
    
    # 绘制图片
    screen.blit(pygame_image, (0, 0))
    
    # 更新屏幕显示
    pygame.display.update()
    
def world_to_camera(world_points, Tw2c):
    """
    将世界坐标点转换到相机坐标系。

    Args:
        world_points (np.ndarray): 3D世界坐标点, 形状 (B, 3)
        Tw2c (np.ndarray): 相机外参矩阵 (4x4)，从世界坐标系到相机坐标系的变换矩阵

    Returns:
        np.ndarray: 相机坐标系中的点，形状 (B, 3)
    """
    # 将世界坐标点扩展为齐次坐标
    world_points_homogeneous = np.hstack((world_points, np.ones((world_points.shape[0], 1))))  # (B, 4)

    # 使用外参矩阵进行批量转换
    camera_points_homogeneous = np.dot(world_points_homogeneous, Tw2c.T)  # (B, 4)

    # 转换为非齐次坐标
    camera_points = camera_points_homogeneous[:, :3]  # (B, 3)

    return camera_points

def camera_to_image(camera_points, K, image_width=None, image_height=None, normalized_uv=False):
    """
    将相机坐标点投影到图像平面，并归一化到 [0, 1] 范围。

    Args:
        camera_points (np.ndarray): 相机坐标点，形状 (B, 3)
        K (np.ndarray): 相机内参矩阵，形状 (3, 3)
        image_width (int): 图像宽度
        image_height (int): 图像高度

    Returns:
        np.ndarray: 归一化像素坐标，形状 (B, 2)
    """
    # 使用内参矩阵进行投影
    projected_points = np.dot(camera_points, K.T)  # (B, 3)

    # 转换为非齐次像素坐标
    u = projected_points[:, 0] / projected_points[:, 2]  # (B,)
    v = projected_points[:, 1] / projected_points[:, 2]  # (B,)
    uv = np.vstack((u, v)).T  # (B, 2)
    
    # 提取深度值
    depth = projected_points[:, 2]  # (B,)
    
    # 归一化到 [0, 1]
    if normalized_uv:
        assert image_width is not None and image_height is not None, \
            "image_width and image_height must be provided when normalized_uv is True"
        u = u / image_width
        v = v / image_height

    return uv, depth  # (B, 2), (B, )

def camera_to_image_torch(camera_points, K, image_width=None, image_height=None, normalized_uv=False):
    """
    将相机坐标点投影到图像平面，并归一化到 [0, 1] 范围。

    Args:
        camera_points (torch.Tensor): 相机坐标点，形状 (B, 3)
        K (torch.Tensor): 相机内参矩阵，形状 (3, 3)
        image_width (int): 图像宽度
        image_height (int): 图像高度

    Returns:
        torch.Tensor: 归一化像素坐标，形状 (B, 2)
        torch.Tensor: 深度值，形状 (B,)
    """
    # 使用内参矩阵进行投影
    projected_points = torch.matmul(camera_points, K.T)  # (B, 3)

    # 转换为非齐次像素坐标
    u = projected_points[:, 0] / projected_points[:, 2]  # (B,)
    v = projected_points[:, 1] / projected_points[:, 2]  # (B,)
    uv = torch.stack((u, v), dim=1)  # (B, 2)

    # 提取深度值
    depth = projected_points[:, 2]  # (B,)

    # 归一化到 [0, 1]
    if normalized_uv:
        assert image_width is not None and image_height is not None, \
            "image_width and image_height must be provided when normalized_uv is True"
        u = u / image_width
        v = v / image_height
        uv = torch.stack((u, v), dim=1)  # (B, 2)

    return uv, depth  # (B, 2), (B,)


def world_to_image(world_points, K, Tw2c, image_width=None, image_height=None, normalized_uv=False):
    """
    将世界坐标点转换为归一化像素坐标 (u, v) [0, 1]。

    Args:
        world_points (np.ndarray): 世界坐标点，形状 (B, 3)
        K (np.ndarray): 相机内参矩阵，形状 (3, 3)
        Tw2c (np.ndarray): 相机外参矩阵 (4x4)，从世界坐标系到相机坐标系的变换矩阵
        image_width (int): 图像宽度
        image_height (int): 图像高度

    Returns:
        np.ndarray: 归一化像素坐标，形状 (B, 2)
    """
    # 转换到相机坐标系
    camera_points = world_to_camera(world_points, Tw2c)

    # 投影到图像平面并归一化
    uv, depth = camera_to_image(camera_points, K, image_width, image_height, normalized_uv)

    return uv # B, 2

def draw_points_on_image(image, uv_list, radius=1, normalized_uv=False):
    """
    Args:
        image: PIL.Image 对象，表示要绘制点的图像。
        uv_list: 一个包含 (u, v) 坐标的列表，表示要绘制的点。
    """
    # 获取图像的宽度和高度
    width, height = image.size
    image_draw = image.copy()
    draw = ImageDraw.Draw(image_draw)
    
    for u, v in uv_list:
        # 将归一化坐标转换为像素坐标
        if normalized_uv:
            x = int(u * width)
            y = int(v * height)
        else:
            x = int(u)
            y = int(v)
        
        # 在 (x, y) 位置画一个红色的点 (填充颜色为红色)
        if radius == 1:
            draw.point((x, y), fill=(255, 0, 0))  # (255, 0, 0) 表示红色
        else:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 0))
    return image_draw

def pil_images_to_mp4(pil_images, output_filename, fps=30):
    """
    将一系列PIL图像保存为MP4视频。

    Args:
        pil_images (list of PIL.Image): PIL图像对象的列表。
        output_filename (str): 输出视频文件的路径。
        fps (int): 每秒的帧数，控制视频的帧率。
    """
    # 获取图像的尺寸 (假设所有图像的大小相同)
    width, height = pil_images[0].size

    # 定义视频编写器
    fourcc = cv2.VideoWriter_fourcc(*'X264')  # 使用 mp4 编码
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    # 将 PIL 图像逐一转换为 NumPy 数组并写入视频
    for pil_img in pil_images:
        # 将 PIL 图像转换为 NumPy 数组
        img_array = np.array(pil_img)

        # 将 RGB 图像转换为 BGR（OpenCV 默认的颜色顺序是 BGR）
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # 写入视频帧
        video_writer.write(img_array)

    # 释放资源
    video_writer.release()
    print(f"Video saved as {output_filename}")
    
def add_text_to_image(image, text, position=(10, 10), font_size=30):
    """
    在图像的指定位置添加文本。
    
    Args:
        image (PIL.Image): 输入的PIL图像。
        text (str): 要添加的文本。
        position (tuple): 文本的左上角位置（默认在左上角，坐标为(10, 10)）。
        font_size (int): 字体大小, default 30。
    
    Returns:
        PIL.Image: 添加了文本的图像。
    """
    # 创建ImageDraw对象
    draw = ImageDraw.Draw(image)
    
    # 使用默认字体，如果需要可以提供路径
    font = ImageFont.load_default()  # 使用Pillow默认字体
    try:
        font = ImageFont.truetype("arial.ttf", font_size)  # 如果有系统字体，可以选择
    except IOError:
        print("Arial font not found, using default font.")
    
    # 在图像上绘制文本
    draw.text(position, text, font=font, fill="black")  # fill指定文本颜色，黑色
    
    return image

def image_to_camera(uv, depth, K, image_width=None, image_height=None, normalized_uv=False):
    """
    将归一化的像素坐标和深度值转换为三维空间中的点（相机坐标系）。
    
    参数:
    - uv (np.array): 形状为 (B, 2)，每个点的 [u, v] 坐标，值域为 [0, 1]。
    - depth (np.array): 形状为 (B,)，每个点的深度值，单位：米。
    - K (np.array): 相机内参矩阵 (3x3)，包括焦距和主点坐标。
    - image_width (int): 图像的宽度（单位：像素）。
    - image_height (int): 图像的高度（单位：像素）。
    
    返回:
    - (np.array): 形状为 (B, 3) 的点云数组，每个点的三维坐标 (X, Y, Z)。
    """
    # 1. 提取 uv 坐标
    u = uv[:, 0]  # 水平像素坐标
    v = uv[:, 1]  # 垂直像素坐标

    # 2. 计算实际的像素坐标
    if normalized_uv:
        x_pixel = u * image_width
        y_pixel = v * image_height
    else:
        x_pixel = u
        y_pixel = v

    # 3. 获取相机内参
    f_x = K[0, 0]  # 水平焦距 (单位: 像素)
    f_y = K[1, 1]  # 垂直焦距 (单位: 像素)
    c_x = K[0, 2]  # 主点 x 坐标 (单位: 像素)
    c_y = K[1, 2]  # 主点 y 坐标 (单位: 像素)
    
    # 4. 计算三维坐标 (X, Y, Z)
    Z = depth  # 深度值，单位：米
    X = (x_pixel - c_x) * Z / f_x
    Y = (y_pixel - c_y) * Z / f_y
    
    # 5. 返回点云 (B, 3)
    point_cloud = torch.stack((X, Y, Z), dim=1)  # 将 X, Y, Z 合并成 (B, 3)
    
    return point_cloud


def depth_image_to_pointcloud(depth_image, mask, K):
    """
    将深度图像转换为相机坐标系中的点云。

    Args:
    - depth_image (np.array): 深度图像，大小为 (H, W)，单位：米
    - K (np.array): 相机内参矩阵 (3x3)，包括焦距和主点坐标
    - mask (np.array, optional): 掩码，大小为 (H, W)，布尔类型。如果提供，只保留 mask 为 True 的点

    Returns:
    - pointcloud (np.array): 点云，大小为 (N, 3)，表示每个像素点在相机坐标系下的三维坐标
    """
    assert depth_image.ndim == 2, "depth_image must be a 2D array"
    if mask is not None:
        assert mask.dtype == np.bool_, "mask must be a boolean array"
    # 获取相机内参
    f_x = K[0, 0]  # 水平焦距 (单位: 像素)
    f_y = K[1, 1]  # 垂直焦距 (单位: 像素)
    c_x = K[0, 2]  # 主点 x 坐标 (单位: 像素)
    c_y = K[1, 2]  # 主点 y 坐标 (单位: 像素)

    # 生成像素网格
    image_height, image_width = depth_image.shape
    u, v = np.meshgrid(np.arange(image_width), np.arange(image_height))

    # 计算归一化坐标
    x_normalized = (u - c_x) / f_x
    y_normalized = (v - c_y) / f_y

    # 获取深度值
    Z = depth_image  # 深度图像直接作为 Z 值

    # 计算相机坐标系中的三维坐标
    X = x_normalized * Z
    Y = y_normalized * Z

    # 合并为点云 (H, W, 3)
    pointcloud = np.stack((X, Y, Z), axis=-1)

    # 如果提供了掩码，仅保留掩码为 True 的点
    if mask is not None:
        pointcloud = pointcloud[mask]

    pointcloud = pointcloud.reshape(-1, 3)
    return pointcloud

def camera_to_world(point_camera, extrinsic_matrix):
    """
    将相机坐标系中的点转换到世界坐标系。
    
    Args:
        point_camera:
            np.array([N, 3]), 相机坐标系中的点
        extrinsic_matrix (np.array): 
            Tw2c, 外参矩阵(3, 4)或者(4, 4), 形式为 [R | t]
    
    Returns:
        point_world np.array([N, 3])
    """
    # 从外参矩阵中提取旋转矩阵 R 和平移向量 t
    Rw2c = extrinsic_matrix[:3, :3]  # (3, 3)
    tw2c = extrinsic_matrix[:3, 3]   # (3)
    tc2w = -Rw2c.T @ tw2c

    # 计算从相机坐标系到世界坐标系的转换
    point_world = point_camera @ Rw2c + tc2w
    
    return point_world

def visualize_pc(points, colors=None, grasp=None):
    """
        visualize pointcloud
        points: Nx3
        colors: Nx3 (0-1)
        grasp: None
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if isinstance(grasp, graspnetAPI.grasp.Grasp):
        grasp_o3d = grasp.to_open3d_geometry()
        o3d.visualization.draw_geometries([grasp_o3d, pcd])
    elif isinstance(grasp, graspnetAPI.grasp.GraspGroup):
        grasp_o3ds = grasp.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([*grasp_o3ds, pcd])
    else:
        o3d.visualization.draw_geometries([pcd])
        
@torch.no_grad()
def plot_matching(image1, image2, hr1, hr2, span):
    seed_everything(0)
    [hr_feats_pca_1, hr_feats_pca_2], _ = pca([hr1.unsqueeze(0), hr2.unsqueeze(0)])
    fig, ax = plt.subplots(1, 5, figsize=(25, 5))
    
    ax[0].imshow(image1.permute(1, 2, 0).detach().cpu())
    ax[0].set_title("Image 1")
    ax[1].imshow(image2.permute(1, 2, 0).detach().cpu())
    ax[1].set_title("Image 2")
    ax[2].imshow(hr_feats_pca_1[0].permute(1, 2, 0).detach().cpu())
    ax[2].set_title("Features 1")
    ax[3].imshow(hr_feats_pca_2[0].permute(1, 2, 0).detach().cpu())
    ax[3].set_title("Features 2")
    ax[4].imshow(span.detach().cpu(), cmap='jet') # "viridis"
    ax[4].set_title("Span")
    remove_axes(ax)
    plt.show()
    
@torch.no_grad()
def plot_matching_2(feat1, feat2, similarity_map, uv_1):
    """
        feat1: 1, C, H, W
        similarity_map: H, W
        uv_1: [0.5, 0.5]
    """
    seed_everything(0)
    [feat1_pca, feat2_pca], _ = pca([feat1, feat2])
    fig, ax = plt.subplots(1, 3, figsize=(5 * 3, 5))

    ax0_img = feat1_pca[0].permute(1, 2, 0).detach().cpu().numpy() # H, W, C
    ax0_img = Image.fromarray((ax0_img * 255).astype(np.uint8))
    ax0_img = draw_points_on_image(image=ax0_img, uv_list=[uv_1], radius=3)
    ax[0].imshow(ax0_img) 
    ax[0].set_title("Features 1")
    ax[1].imshow(feat2_pca[0].permute(1, 2, 0).detach().cpu())
    ax[1].set_title("Features 2")
    ax[2].imshow(similarity_map.detach().cpu(), cmap='jet') # "viridis"
    ax[2].set_title("Similarity Map")
    remove_axes(ax)
    plt.show()

def match_point_on_featmap(
    feat_1: torch.Tensor, 
    feat_2: torch.Tensor,
    uv_1: List[float],
    visualize: bool=False, 
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
    # 获取左图指定点的坐标 (u, v)，并转换为像素坐标
    u, v = uv_1
    h1, w1 = feat_1.shape[2], feat_1.shape[3]  # 获取高和宽
    left_pixel_x = int(u * w1)  # 对应的像素坐标
    left_pixel_y = int(v * h1)  # 对应的像素坐标
    
    # 获取左图指定点的特征
    left_point_feature = feat_1[0, :, left_pixel_y, left_pixel_x]  # (feature_dim, )

    # 计算左图指定点与右图所有点的余弦相似度
    # 将左图指定点的特征扩展到右图的每个位置进行比较
    right_features = feat_2.view(feat_2.size(1), -1).T  # shape: (seq_length, feature_dim)

    # 计算余弦相似度
    similarity = F.cosine_similarity(left_point_feature.unsqueeze(0), right_features, dim=1) # 65536

    # 提取匹配点的坐标
    h2, w2 = feat_2.shape[2], feat_2.shape[3]  # 获取高和宽
    similarity_map = similarity.reshape(h2, w2)
    
    if visualize:
        plot_matching_2(feat_1, feat_2, similarity_map, uv_1)
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

class SimilarityMap:
    def __init__(self, similarity_map: torch.Tensor, alpha: float=20):
        """
            alpha: torch.exp(alpha * x) / sum(torch.exp(alpha * x))
                alpha越大, 概率分布越sharp
        """
        self.similarity_map = similarity_map # H, W, in range [0, 1]
        self.H, self.W = similarity_map.shape
        
        # pil image for visualization
        # cmap = cm.get_cmap("jet")
        cmap = cm.get_cmap("viridis")
        colored_image = cmap(self.similarity_map.cpu().numpy())  # Returns (H, W, 4) RGBA array
        # Convert to 8-bit RGB (ignore the alpha channel)
        colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)
        # Convert to PIL Image
        self.pil_image = Image.fromarray(colored_image, mode="RGB")
        
        # Flatten the tensor and normalize it to create a probability distribution
        flat_tensor = self.similarity_map.flatten()
        exp_tensor = torch.exp(flat_tensor * alpha)
        probabilities = exp_tensor / torch.sum(exp_tensor) # H * W
        # Convert probabilities to numpy for sampling
        probabilities_np = probabilities.cpu().numpy()
        self.prob_np = probabilities_np

    def sample(self, num_samples=50, visualize=True):
        # Sample indices based on the probability distribution
        sampled_indices = np.random.choice(len(self.prob_np), size=num_samples, p=self.prob_np, replace=False)
        # Convert flat indices to 2D coordinates (y, x)
        y_coords, x_coords = np.unravel_index(sampled_indices, (self.H, self.W))

        # Normalize coordinates to range [0, 1]
        u_coords = x_coords / self.W
        v_coords = y_coords / self.H

        # Combine u and v into a single array of shape (num_samples, 2)
        sampled_coordinates = np.stack((u_coords, v_coords), axis=-1)
        
        if visualize:
            img = draw_points_on_image(self.pil_image, sampled_coordinates, radius=3)
            img.show()
            
        return sampled_coordinates
    

import numpy as np

def reconstruct_mask(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """
    根据初始布尔掩码 (mask1) 和筛选掩码 (mask2)，返回最终筛选后的布尔掩码。

    参数:
    - mask1: np.ndarray (H, W)  -> 初始布尔掩码
    - mask2: np.ndarray (N,)     -> 选中的N个点的筛选掩码

    返回:
    - new_mask: np.ndarray (H, W) -> 仅包含 mask2 选中像素点的布尔掩码
    """
    assert mask1.dtype == np.bool_, "mask1 必须是 bool 类型"
    assert mask2.dtype == np.bool_, "mask2 必须是 bool 类型"
    
    # 获取 mask1 中为 True 的索引
    indices = np.argwhere(mask1)  # (N, 2) 形状数组，每行是 (y, x)
    
    # 选出最终保留的 M 个像素点
    selected_indices = indices[mask2]  # (M, 2) 形状

    # 构造新的 (H, W) 掩码
    new_mask = np.zeros_like(mask1, dtype=bool)
    
    # 设置 M 个有效位置为 True
    new_mask[selected_indices[:, 0], selected_indices[:, 1]] = True
    
    return new_mask

def tracksNd_variance(tracks: torch.Tensor) -> torch.Tensor:
    """
    计算 T, M, 3 形状的 3D 轨迹数据的平均方差 (PyTorch 版本)。

    参数:
        tracks (torch.Tensor): 形状为 (T, M, Nd) 的张量，表示 T 个时刻 M 个点的 2D/3D 轨迹。

    返回:
        torch.Tensor: M 个点的方差均值（标量）。
    """
    # 计算每个点在 T 个时刻的方差 (M, 3)
    pointwise_variance = torch.var(tracks, dim=0, unbiased=False)  # 计算 M 个点的 3 维方差
    
    # 计算每个点的总方差（对 3 维坐标求均值）
    pointwise_variance_mean = torch.mean(pointwise_variance, dim=1)  # (M,)

    # 计算所有点的方差均值
    average_variance = torch.mean(pointwise_variance_mean)

    return average_variance


def farthest_scale_sampling(arr, M):
    """
    从一维数组中选择 M 个点，确保它们之间的距离尽可能大（最大最小距离采样）。
    
    参数:
    arr (np.ndarray): 输入的一维数据数组。
    M (int): 需要选择的数据点数量。

    返回:
    np.ndarray: 选择出的代表性数据点。
    """
    arr = np.array(arr)
    N = len(arr)
    
    if M >= N:
        return arr  # 如果需要的点数大于等于数组长度，直接返回原数组

    # 随机选择第一个点（也可以选择固定的起点，如最小值或最大值）
    selected_indices = [np.random.randint(0, N)]
    
    # 迭代选择剩余的点
    for _ in range(1, M):
        # 计算未选点到已选点的最小距离
        remaining_indices = list(set(range(N)) - set(selected_indices))
        min_distances = np.array([
            min(abs(arr[i] - arr[j]) for j in selected_indices)
            for i in remaining_indices
        ])
        
        # 选择最小距离最大的点
        next_index = remaining_indices[np.argmax(min_distances)]
        selected_indices.append(next_index)
    
    # 返回排序后的抽样点，方便阅读
    # return arr[np.sort(selected_indices)]
    return np.sort(selected_indices)


def sample_array(arr, k):
    """
    从大小为 N x d 的数组中随机采样 k 个样本，并返回 k x d 的数组。
    :param arr: 输入的 N x d 的 numpy 数组
    :param k: 需要采样的样本数量
    :return: 返回大小为 k x d 的数组
    """
    # 确保 k 不大于数组的行数
    assert k <= arr.shape[0], "k 不能大于数组的行数"
    
    # 随机选择 k 个索引
    indices = np.random.choice(arr.shape[0], size=k, replace=False)
    
    # 根据索引提取行，并取前两列
    sampled_array = arr[indices]
    
    return sampled_array

def sample_points_within_bbox_and_mask(bbox, mask, N):
    """
    从给定的bbox和mask中采样N个有效的(u, v)点。
    
    参数：
    - bbox: 形状为(4,)的bbox数组，格式为[u_left, v_left, u_right, v_right]。
    - mask: 大小为(H, W)的布尔数组，表示图像中每个像素是否有效。
    - N: 需要采样的点数。
    
    返回：
    - 一个形状为(N, 2)的numpy数组，表示采样的(u, v)坐标。
      其中，N是有效采样点的数量。
    """
    # 获取bbox的坐标
    u_left, v_left, u_right, v_right = bbox

    # bbox的宽高
    bbox_u_range = u_right - u_left
    bbox_v_range = v_right - v_left

    # 计算步长，使得采样点的数量接近N
    step_size_u = np.sqrt(bbox_u_range * bbox_v_range / N)
    step_size_v = step_size_u * bbox_v_range / bbox_u_range  # 保持长宽比

    # 计算采样点的数量
    num_u = int(np.floor(bbox_u_range / step_size_u))
    num_v = int(np.floor(bbox_v_range / step_size_v))

    # 生成均匀采样的点
    u_coords = np.linspace(u_left, u_right, num_u)
    v_coords = np.linspace(v_left, v_right, num_v)

    # 使用meshgrid生成(u, v)坐标网格
    u_grid, v_grid = np.meshgrid(u_coords, v_coords)

    # 将(u_grid, v_grid)展平为一维数组
    u_flat = u_grid.flatten()
    v_flat = v_grid.flatten()

    # 转换为 (N, 2) 形状
    points = np.stack([u_flat, v_flat], axis=1)

    # 在mask中检查这些点是否有效
    valid_points = points[mask[points[:, 1].astype(int), points[:, 0].astype(int)]]

    return valid_points

def napari_time_series_transform(original_data):
    """
        将原始的时序数据转换为 napari 可视化所需的格式。
        original_data: np.ndarray, (T, N, d)
        returned_data: np.ndarray, (T*N, 1+d), 1代表时间维度
    """
    T = len(original_data)
    napari_data = []
    for i in range(T):
        tmp_data = original_data[i] # M, d
        tmp_data_with_t = np.concatenate([np.ones((tmp_data.shape[0], 1)) * i, tmp_data], axis=1) # M, (1+d)
        napari_data.append(tmp_data_with_t)
    napari_data = np.concatenate(napari_data, axis=0)
    return napari_data
def joint_data_to_transform(
    joint_type, # "prismatic" or "revolute"
    joint_axis, # unit vector np.array([3, ])
    joint_state_ref2tgt # joint_state_tgt - joint_state_ref, a constant
):
    # 根据 joint_type 和 joint_axis 和 (joint_state2 - joint_state1) 得到 T_ref2tgt
    T_ref2tgt = np.eye(4)
    if joint_type == "prismatic":
        # coor_tgt = coor_ref + joint_axis * (joint_state_tgt - joint_state_ref)
        T_ref2tgt[:3, 3] = joint_axis * joint_state_ref2tgt
    elif joint_type == "revolute":
        # coor_tgt = coor_ref @ Rref2tgt.T
        Rref2tgt = R.from_rotvec(joint_axis * joint_state_ref2tgt).as_matrix()
        T_ref2tgt[:3, :3] = Rref2tgt
    else:
        assert False, "joint_type must be either prismatic or revolute"
    return T_ref2tgt


def set_random_seed(seed: int):
    """
    设置 Python, NumPy 和 PyTorch 的随机种子，确保实验的可重复性
    :param seed: 要设置的随机种子
    """
    # 设置 Python 的随机种子
    random.seed(seed)
    
    # 设置 NumPy 的随机种子
    np.random.seed(seed)
    
    # 设置 PyTorch 的随机种子
    torch.manual_seed(seed)
    
    # 如果使用 GPU，还需要设置 CUDA 随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 的种子
        
    # 为了确保结果可重复，设置 cudnn 的确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def extract_tracked_depths(depth_seq, pred_tracks):
    """
    Args:
        depth_seq (np.ndarray): 
            深度图视频,形状为 (T, H, W)
        pred_tracks (torch.Tensor): 
            (T, M, 2)
            每个点的坐标形式为 [u, v],值域为 [0, W) 和 [0, H)。
        
    返回:
        torch.Tensor: 点的深度值,形状为 (T, M),每个点的深度值。
    """
    T, H, W = depth_seq.shape
    # _, M, _ = pred_tracks.shape

    # 确保 pred_tracks 是整数坐标
    u_coords = pred_tracks[..., 0].clamp(0, W - 1).long()  # 水平坐标 u
    v_coords = pred_tracks[..., 1].clamp(0, H - 1).long()  # 垂直坐标 v

    # 将 depth_seq 转为 torch.Tensor
    depth_tensor = torch.from_numpy(depth_seq).cuda()  # Shape: (T, H, W)

    # 使用高级索引提取深度值
    depth_tracks = depth_tensor[torch.arange(T).unsqueeze(1), v_coords, u_coords]  # Shape: (T, M)
    return depth_tracks.cpu()

def filter_tracks2d_by_visibility(rgb_seq, pred_tracks_2d, pred_visibility, visualize=False):
    """
        pred_tracks_2d: torch.tensor([T, M, 2])
        pred_visibility: torch.tensor([T, M])
    """
    always_visible_mask = pred_visibility.all(dim=0) # num_clusters
    pred_tracks_2d = pred_tracks_2d[:, always_visible_mask, :] # T M_ 2
    
    if visualize:
        vis_tracks2d_napari(rgb_seq, pred_tracks_2d, viewer_title="filter_tracks2d_by_visibility")
    return pred_tracks_2d

def filter_tracks2d_by_depthSeq_mask(rgb_seq, pred_tracks_2d, depthSeq_mask, visualize=False):
    """
    筛选出那些不曾落到 invalid depth_mask 中的 tracks
    Args:
        pred_tracks_2d: torch.tensor([T, M, 2])
        depthSeq_mask: np.array([T, H, W], dtype=np.bool_)
    """
    T = pred_tracks_2d.shape[0]
    # 首先读取出 pred_tracks_2d 在对应位置的 mask 的值
    pred_tracks_2d_floor = torch.floor(pred_tracks_2d).long() # T M 2 (u, v)
    t_index = torch.arange(T, device=pred_tracks_2d.device).view(T, 1)  # (T, M)
    pred_tracks_2d_mask = depthSeq_mask[t_index, pred_tracks_2d_floor[:, :, 1], pred_tracks_2d_floor[:, :, 0]] # T M
    
    # 然后在时间维度上取交, 得到一个大小为 M 的 mask, 把其中一直为 True 的保留下来并返回
    mask_and = np.all(pred_tracks_2d_mask, axis=0) # M
    
    pred_tracks_2d = pred_tracks_2d[:, mask_and, :]
    if visualize:
        vis_tracks2d_napari(rgb_seq, pred_tracks_2d, viewer_title="filter_tracks2d_by_depthSeq_mask")
    return pred_tracks_2d

def filter_tracks2d_by_depthSeq_diff(pred_tracks, depth_tracks, thr=1.5):
    """
    过滤深度序列中的无效序列,基于相邻元素的相对变化率。
    
    参数:
        pred_tracks (torch.Tensor): 
            形状为 (T, N, 2),每个点的坐标形式为 [u, v],值域为 [0, W) 和 [0, H)。
        depth_tracks (torch.Tensor): 
            形状为 (T, N),表示 T 个时间步上的 N 个深度序列。
        thr (float): 
            相对变化率的阈值,超过此比率的序列会被判定为无效。
    
    返回:
        torch.Tensor: 
            筛选后的 pred_tracks, 形状为 (T, M, 2),其中 M 是筛选后的有效点数量。
    """
    depth_tracks += 1e-6  # 防止除零

    # 转置 depth_tracks,使其形状为 (N, T)
    depth_tracks_T = depth_tracks.T  # (N, T)

    # 计算相邻元素的变化比率 (N, T-1)
    ratios = torch.max(depth_tracks_T[:, :-1] / depth_tracks_T[:, 1:], depth_tracks_T[:, 1:] / depth_tracks_T[:, :-1])

    # 检测每个序列是否存在无效的变化 (N,)
    invalid_sequences = torch.any(ratios > thr, dim=1)  # 检查每一行是否有任何值大于阈值

    # 合法序列的掩码 (N,)
    valid_mask = ~invalid_sequences  # 合法序列为 False,其他为 True
    
    # 筛选 pred_tracks 和 depth_tracks 的合法序列
    pred_tracks = pred_tracks[:, valid_mask, :]  # 筛选有效的 M 点,(T, M, 2)
    depth_tracks = depth_tracks[:, valid_mask]   # 筛选有效的 M 深度,(T, M)
    
    return pred_tracks, depth_tracks

def visualize_tracks2d_on_video(rgb_frames, pred_tracks, file_name="video_output", vis_folder="./"):
    """
        Args:
            rgb_frames: np.array([T, H, W, 3])
            pred_tracks: torch.Tensor([T, M, 2])
    """
    os.makedirs(vis_folder, exist_ok=True)
    video = torch.tensor(np.stack(rgb_frames), device="cuda").permute(0, 3, 1, 2)[None]
    vis = Visualizer(save_dir=vis_folder, pad_value=120, linewidth=1)
    
    pred_tracks = pred_tracks[None] # [1, T, M, 2]
    # pred_visibility = torch.ones((pred_tracks.shape[:3], 1), dtype=torch.bool, device="cuda")
    vis.visualize(video, pred_tracks, filename=file_name)


def get_dynamic_mask(mask, moving_points, static_points, visualize=False):
    """
    根据A和B点集的最近邻分类让mask中为True的点分类。

    参数:
        mask (np.ndarray): 大小为(H, W)的应用匹配网络,mask=True的点需要分类
        points_A (np.ndarray): 大小为(M, 2)的A类点集,具有(u, v)坐标
        points_B (np.ndarray): 大小为(N, 2)的B类点集,具有(u, v)坐标

    返回:
        dynamic_mask (np.ndarray): 大小为(H, W)的分类结果,A类标记1,B类标记2
    """
    H, W = mask.shape

    # 构建KD树以加快最近邻搜索，确保用(v, u)坐标格式
    tree_A = cKDTree(moving_points[:, [1, 0]])
    tree_B = cKDTree(static_points[:, [1, 0]])

    # 找到mask中为True的点坐标
    mask_indices = np.argwhere(mask) # N, 2 (v, u)

    # 对每个True点计算自A和B集合的最近距离
    distances_A, _ = tree_A.query(mask_indices)
    distances_B, _ = tree_B.query(mask_indices)

    # 初始化结果网络
    dynamic_mask = np.zeros((H, W), dtype=np.uint8)

    # 根据最近邻距离分类
    A_closer = distances_A < distances_B # N
    B_closer = ~A_closer

    # 将分类结果填入到对应位置
    dynamic_mask[mask_indices[A_closer, 0], mask_indices[A_closer, 1]] = MOVING_LABEL
    dynamic_mask[mask_indices[B_closer, 0], mask_indices[B_closer, 1]] = STATIC_LABEL

    if visualize:
        import napari 
        viewer = napari.view_image(mask, rgb=True)
        viewer.add_labels(mask.astype(np.int32), name='articulated objects')
        viewer.add_labels((dynamic_mask == MOVING_LABEL).astype(np.int32) * 2, name='moving parts')
        viewer.add_labels((dynamic_mask == STATIC_LABEL).astype(np.int32) * 3, name='static parts')
        napari.run()
    return dynamic_mask

def get_dynamic_mask_seq(mask_seq, moving_points_seq, static_points_seq, visualize=False):
    T = mask_seq.shape[0]
    dynamic_seg_seq = []
    
    for i in range(T):
        dynamic_seg = get_dynamic_mask(mask_seq[i], moving_points_seq[i], static_points_seq[i])
        dynamic_seg_seq.append(dynamic_seg)
    
    dynamic_seg_seq = np.array(dynamic_seg_seq)
    
    if visualize:
        import napari 
        viewer = napari.view_image(mask_seq, rgb=False)
        viewer.title = "dynamic segment video mask by tracks2d"
        # viewer.add_labels(mask_seq.astype(np.int32), name='articulated objects')
        viewer.add_labels((dynamic_seg_seq == MOVING_LABEL).astype(np.int32) * 2, name='moving parts')
        viewer.add_labels((dynamic_seg_seq == STATIC_LABEL).astype(np.int32) * 3, name='static parts')
        napari.run()
    return dynamic_seg_seq # T, H, W


def quantile_sampling(arr, M):
    """
    从数组中按分位数抽样，返回代表性数据点。
    
    参数:
    arr (np.ndarray): 输入的一维数据数组。
    M (int): 要抽取的数据点数量。

    返回:
    np.ndarray: 抽取的代表性数据点。
    """
    # 确保输入为NumPy数组并排序
    arr_sorted = np.sort(arr)
    
    # 计算分位数对应的索引
    quantiles = np.linspace(0, 1, M)
    indices = (quantiles * (len(arr_sorted) - 1)).astype(int)
    
    # 根据索引抽取数据
    sampled_data = arr_sorted[indices]
    return sampled_data

def compute_normals(target_pc, k_neighbors=10):
    """
    计算目标点云的法向量
    :param target_pc: numpy 数组，形状为 (N, 3)，目标点云
    :param k_neighbors: 计算法向量时的近邻数
    :return: 法向量数组，形状为 (N, 3)
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(target_pc)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors))
    normals = np.asarray(pcd.normals)
    return normals

def find_correspondences(ref_pc, target_pc, max_distance=0.01):
    """
    使用 KD-Tree 进行最近邻搜索，找到参考点云 ref_pc 在目标点云 target_pc 中的最近邻
    :param ref_pc: numpy 数组，形状 (M, 3)，参考点云
    :param target_pc: numpy 数组，形状 (N, 3)，目标点云
    :param max_distance: 最大距离，超过这个距离的点会被忽略
    :return: 匹配的索引数组
    """
    # TODO: 将 leafsize 和 num_workers 再优化
    tree = cKDTree(target_pc)
    distances, indices = tree.query(ref_pc, workers=4)
    
    if max_distance > 0.0:
        valid_mask = distances < max_distance
    else:
        valid_mask = np.ones(distances.shape, dtype=np.bool_)
    return indices, distances, valid_mask

def rotation_matrix_between_vectors(a, b):
    # 假设满足 R @ a = b, 求R
    
    # 确保 a 和 b 是单位向量
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    
    # 计算旋转轴 u 和旋转角度 theta
    u = np.cross(a, b)  # 旋转轴是 a 和 b 的叉积
    sin_theta = np.linalg.norm(u)
    cos_theta = np.dot(a, b)  # 夹角余弦
    u = u / sin_theta if sin_theta != 0 else u  # 归一化旋转轴

    # 计算旋转矩阵
    I = np.eye(3)
    u_cross = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])  # 叉积矩阵

    R = I + np.sin(np.arccos(cos_theta)) * u_cross + (1 - cos_theta) * np.dot(u_cross, u_cross)
    return R


if __name__ == "__main__":
    bbox = np.array([50, 50, 150, 150])  # 示例bbox
    mask = np.random.choice([False, True], size=(200, 200))  # 随机生成一个mask
    N = 200  # 需要采样的点数

    valid_points = sample_points_within_bbox_and_mask(bbox, mask, N)
    print(f"有效采样点: {valid_points.shape[0]}个")
    print(valid_points)
