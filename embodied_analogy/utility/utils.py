import pygame
import torch
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


def napari_time_series_transform(original_data):
    """
        将原始的时序数据转换为 napari 可视化所需的格式。
        original_data: np.ndarray, (T, N, d)
        returned_data: np.ndarray, (T*N, 1+d), 1代表时间维度
    """
    T = original_data.shape[0]
    napari_data = []
    for i in range(T):
        tmp_data = original_data[i] # M, d
        tmp_data_with_t = np.concatenate([np.ones((tmp_data.shape[0], 1)) * i, tmp_data], axis=1) # M, (1+d)
        napari_data.append(tmp_data_with_t)
    napari_data = np.concatenate(napari_data, axis=0)
    return napari_data


if __name__ == "__main__":
    # 示例数据
    data = np.random.randn(100)  # 生成标准正态分布数据
    # sampled_points = quantile_sampling(data, 10)
    sampled_points = farthest_scale_sampling(data, 5)

    print("抽取的分位数点：", sampled_points)
    import matplotlib.pyplot as plt

    # 可视化原始数据和抽取的数据点
    plt.hist(data, bins=50, alpha=0.5, label='Original Data')
    plt.scatter(sampled_points, np.zeros_like(sampled_points), color='red', label='Quantile Samples', zorder=5)
    plt.legend()
    plt.title('Quantile Sampling Visualization')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()
