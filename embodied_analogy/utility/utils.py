import os
# import napari
# import pygame
import torch
import random
from PIL import Image
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import open3d as o3d
import graspnetAPI
from pytorch_lightning import seed_everything
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from typing import List, Union
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from embodied_analogy.utility.constants import *

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
    
def pil_to_pygame(pil_image):
    pil_image = pil_image.convert("RGB")  # 转换为 RGB 格式
    return pygame.image.fromstring(np.array(pil_image).tobytes(), pil_image.size, "RGB")

# 更新图像函数
def update_image_old(screen, rgb_pil, depth_pil, mask_pil):
    # 获取每个图像的尺寸
    width1, height1 = rgb_pil.size
    width2, height2 = depth_pil.size
    width3, height3 = mask_pil.size

    # 计算最终合并图像的尺寸：宽度是三个图像的宽度之和,高度取最大值
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
        Tw2c (np.ndarray): 相机外参矩阵 (4x4),从世界坐标系到相机坐标系的变换矩阵

    Returns:
        np.ndarray: 相机坐标系中的点,形状 (B, 3)
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
    将相机坐标点投影到图像平面,并归一化到 [0, 1] 范围。

    Args:
        camera_points (np.ndarray): 相机坐标点,形状 (B, 3)
        K (np.ndarray): 相机内参矩阵,形状 (3, 3)
        image_width (int): 图像宽度
        image_height (int): 图像高度

    Returns:
        np.ndarray: 归一化像素坐标,形状 (B, 2)
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
    将相机坐标点投影到图像平面,并归一化到 [0, 1] 范围。

    Args:
        camera_points (torch.Tensor): 相机坐标点,形状 (B, 3)
        K (torch.Tensor): 相机内参矩阵,形状 (3, 3)
        image_width (int): 图像宽度
        image_height (int): 图像高度

    Returns:
        torch.Tensor: 归一化像素坐标,形状 (B, 2)
        torch.Tensor: 深度值,形状 (B,)
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
        world_points (np.ndarray): 世界坐标点,形状 (B, 3)
        K (np.ndarray): 相机内参矩阵,形状 (3, 3)
        Tw2c (np.ndarray): 相机外参矩阵 (4x4),从世界坐标系到相机坐标系的变换矩阵
        image_width (int): 图像宽度
        image_height (int): 图像高度

    Returns:
        np.ndarray: 归一化像素坐标,形状 (B, 2)
    """
    # 转换到相机坐标系
    camera_points = world_to_camera(world_points, Tw2c)

    # 投影到图像平面并归一化
    uv, depth = camera_to_image(camera_points, K, image_width, image_height, normalized_uv)

    return uv # B, 2

def draw_points_on_image(image, uv_list, radius=1, normalized_uv=False):
    """
    Args:
        image: PIL.Image 对象, 或是一个 np.array。
        uv_list: 一个包含 (u, v) 坐标的列表,表示要绘制的点。
        返回一个 pil image
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        
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
        fps (int): 每秒的帧数,控制视频的帧率。
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
        position (tuple): 文本的左上角位置（默认在左上角,坐标为(10, 10)）。
        font_size (int): 字体大小, default 30。
    
    Returns:
        PIL.Image: 添加了文本的图像。
    """
    # 创建ImageDraw对象
    draw = ImageDraw.Draw(image)
    
    # 使用默认字体,如果需要可以提供路径
    font = ImageFont.load_default()  # 使用Pillow默认字体
    try:
        font = ImageFont.truetype("arial.ttf", font_size)  # 如果有系统字体,可以选择
    except IOError:
        print("Arial font not found, using default font.")
    
    # 在图像上绘制文本
    draw.text(position, text, font=font, fill="black")  # fill指定文本颜色,黑色
    
    return image

def image_to_camera(uv, depth, K, image_width=None, image_height=None, normalized_uv=False):
    """
    将归一化的像素坐标和深度值转换为三维空间中的点（相机坐标系）。
    
    参数:
    - uv (np.array): 形状为 (B, 2),每个点的 [u, v] 坐标,值域为 [0, 1]。
    - depth (np.array): 形状为 (B,),每个点的深度值,单位：米。
    - K (np.array): 相机内参矩阵 (3x3),包括焦距和主点坐标。
    - image_width (int): 图像的宽度（单位：像素）。
    - image_height (int): 图像的高度（单位：像素）。
    
    返回:
    - (np.array): 形状为 (B, 3) 的点云数组,每个点的三维坐标 (X, Y, Z)。
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
    Z = depth  # 深度值,单位：米
    X = (x_pixel - c_x) * Z / f_x
    Y = (y_pixel - c_y) * Z / f_y
    
    # 5. 返回点云 (B, 3)
    # point_cloud = torch.stack((X, Y, Z), dim=1)  # 将 X, Y, Z 合并成 (B, 3)
    point_cloud = np.stack((X, Y, Z), axis=-1)
    return point_cloud


def depth_image_to_pointcloud(depth_image, mask, K):
    """
    将深度图像转换为相机坐标系中的点云。

    Args:
    - depth_image (np.array): 深度图像,大小为 (H, W),单位：米
    - K (np.array): 相机内参矩阵 (3x3),包括焦距和主点坐标
    - mask (np.array, optional): 掩码,大小为 (H, W),布尔类型。如果提供,只保留 mask 为 True 的点

    Returns:
    - pointcloud (np.array): 点云,大小为 (N, 3),表示每个像素点在相机坐标系下的三维坐标
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

    # 如果提供了掩码,仅保留掩码为 True 的点
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

def create_cylinder(start, end, radius=0.01):
    """创建一个圆柱体来模拟线段,给定起点、终点和半径"""
    start = np.array(start)
    end = np.array(end)
    direction = end - start
    length = np.linalg.norm(direction)
    if length == 0:
        return None

    # 归一化方向
    direction = direction / length

    # 创建圆柱体
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
    cylinder.compute_vertex_normals()

    # 计算圆柱体的旋转和位置
    center = (start + end) / 2  # 圆柱体的中心
    z_axis = np.array([0, 0, 1])

    # 计算旋转矩阵
    if np.allclose(direction, z_axis):
        R = np.eye(3)  # 如果方向是Z轴,保持不变
    elif np.allclose(direction, -z_axis):
        # 如果方向与Z轴相反,旋转180度绕X轴
        rotation_axis = np.array([1, 0, 0])
        rotation_angle = np.pi
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
    else:
        # 计算旋转轴和旋转角度
        rotation_axis = np.cross(z_axis, direction)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)  # 归一化旋转轴
        rotation_angle = np.arccos(np.dot(z_axis, direction))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)

    # 旋转和移动圆柱体
    cylinder.rotate(R, center=cylinder.get_center())  # 绕圆柱体自身中心旋转
    cylinder.translate(center)  # 平移到正确位置

    return cylinder


def farthest_point_sampling(point_cloud, M):
    N = point_cloud.shape[0]
    # 随机选择一个初始点
    indices = np.zeros(M, dtype=np.int32)
    farthest_index = np.random.randint(0, N)
    indices[0] = farthest_index

    # 计算每个点到已选择点的距离
    distances = np.full(N, np.inf)

    for i in range(1, M):
        # 更新每个点到最近已选择点的距离
        dist = np.linalg.norm(point_cloud - point_cloud[farthest_index], axis=1)
        distances = np.minimum(distances, dist)

        # 选择距离最远的点
        farthest_index = np.argmax(distances)
        indices[i] = farthest_index

    return indices


def visualize_pc(points, colors=None, grasp=None, contact_point=None, post_contact_dirs=None):
    """
    visualize pointcloud
    points: Nx3
    colors: Nx3 (0-1)
    grasp: None
    """
    if post_contact_dirs is not None:
        assert isinstance(post_contact_dirs, List)
    # 初始化点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 处理点云颜色
    if colors is None:
        colors = np.zeros([points.shape[0], 3])
        colors[:, 1] = 1  # 默认颜色为蓝色
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 初始化要绘制的几何对象列表
    geometries_to_draw = [pcd]
    
    # 处理 grasp
    if isinstance(grasp, graspnetAPI.grasp.Grasp):
        grasp_o3d = grasp.to_open3d_geometry()
        geometries_to_draw.append(grasp_o3d)
    elif isinstance(grasp, graspnetAPI.grasp.GraspGroup):
        grasp_o3ds = grasp.to_open3d_geometry_list()
        geometries_to_draw.extend(grasp_o3ds)
    
    if contact_point is not None:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)  # 创建小球
        sphere.translate(contact_point)  # 平移到 contact_point
        sphere.paint_uniform_color([1, 0, 0])  # 设置颜色为红色
        geometries_to_draw.append(sphere)
        
    # 处理 contact_point 和 post_contact_dirs
    if contact_point is not None and post_contact_dirs is not None:
        for post_contact_dir in post_contact_dirs:
            start_point = contact_point
            end_point = start_point + post_contact_dir * 0.1
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector([start_point, end_point])
            line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
            line_set.paint_uniform_color([1, 0, 0])  # 红色
            geometries_to_draw.append(line_set)
            
    # 绘制坐标系
    axis_length = 0.1  # 坐标轴长度
    axes = [
        ([0, 0, 0], [axis_length, 0, 0], [1, 0, 0]),  # X轴 (红色)
        ([0, 0, 0], [0, axis_length, 0], [0, 1, 0]),  # Y轴 (绿色)
        ([0, 0, 0], [0, 0, axis_length], [0, 0, 1]),  # Z轴 (蓝色)
    ]
    
    # for start, end, color in axes:
    #     line_set = o3d.geometry.LineSet()
    #     line_set.points = o3d.utility.Vector3dVector([start, end])
    #     line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    #     line_set.paint_uniform_color(color)  # 设置颜色
    #     geometries_to_draw.append(line_set)
    
    for start, end, color in axes:
        cylinder = create_cylinder(start, end)
        if cylinder is not None:
            cylinder.paint_uniform_color(color)  # 设置颜色
            geometries_to_draw.append(cylinder)
    
    # 统一绘制所有几何对象
    o3d.visualization.draw_geometries(geometries_to_draw)

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
    # 获取左图指定点的坐标 (u, v),并转换为像素坐标
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
    
def reconstruct_mask(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """
    根据初始布尔掩码 (mask1) 和筛选掩码 (mask2),返回最终筛选后的布尔掩码。

    参数:
    - mask1: np.ndarray (H, W)  -> 初始布尔掩码
    - mask2: np.ndarray (N,)     -> 选中的N个点的筛选掩码

    返回:
    - new_mask: np.ndarray (H, W) -> 仅包含 mask2 选中像素点的布尔掩码
    """
    assert mask1.dtype == np.bool_, "mask1 必须是 bool 类型"
    assert mask2.dtype == np.bool_, "mask2 必须是 bool 类型"
    
    # 获取 mask1 中为 True 的索引
    indices = np.argwhere(mask1)  # (N, 2) 形状数组,每行是 (y, x)
    
    # 选出最终保留的 M 个像素点
    selected_indices = indices[mask2]  # (M, 2) 形状

    # 构造新的 (H, W) 掩码
    new_mask = np.zeros_like(mask1, dtype=bool)
    
    # 设置 M 个有效位置为 True
    new_mask[selected_indices[:, 0], selected_indices[:, 1]] = True
    
    return new_mask

def tracksNd_variance_torch(tracks: torch.Tensor) -> torch.Tensor:
    """
    计算 T, M, 3 形状的 3D 轨迹数据的平均方差 (PyTorch 版本)。

    参数:
        tracks (torch.Tensor): 形状为 (T, M, Nd) 的张量,表示 T 个时刻 M 个点的 2D/3D 轨迹。

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

def tracksNd_variance_np(tracks: np.ndarray) -> float:
    """
    计算形状为 (T, M, Nd) 的 2D/3D 轨迹数据的平均方差 (NumPy 版本)。

    参数:
        tracks (np.ndarray): 形状为 (T, M, Nd) 的数组,表示 T 个时刻 M 个点的 2D/3D 轨迹。

    返回:
        float: M 个点的方差均值（标量）。
    """
    # 计算每个点在 T 个时刻的方差 (M, Nd)
    pointwise_variance = np.var(tracks, axis=0, ddof=0)  # 计算 M 个点的 Nd 维方差

    # 计算每个点的总方差（对 Nd 维坐标求均值）
    pointwise_variance_mean = np.mean(pointwise_variance, axis=1)  # (M,)

    # 计算所有点的方差均值
    average_variance = np.mean(pointwise_variance_mean)

    return average_variance

def farthest_scale_sampling(arr, M, include_first=True):
    """
    从一维数组中选择 M 个点,确保它们之间的距离尽可能大（最大最小距离采样）。
    
    参数:
    arr (np.ndarray): 输入的一维数据数组。
    M (int): 需要选择的数据点数量。

    返回:
    np.ndarray: 选择出的代表性数据点。
    """
    arr = np.array(arr)
    N = len(arr)
    
    if M >= N:
        return arr  # 如果需要的点数大于等于数组长度,直接返回原数组

    # 随机选择第一个点（也可以选择固定的起点,如最小值或最大值）
    if include_first:
        selected_indices = [0]
    else:
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
    
    # 返回排序后的抽样点,方便阅读
    # return arr[np.sort(selected_indices)]
    return np.sort(selected_indices)


def sample_array(arr, k):
    """
    从大小为 N x d 的数组中随机采样 k 个样本,并返回 k x d 的数组。
    :param arr: 输入的 N x d 的 numpy 数组
    :param k: 需要采样的样本数量
    :return: 返回大小为 k x d 的数组
    """
    # 确保 k 不大于数组的行数
    assert k <= arr.shape[0], "k 不能大于数组的行数"
    
    # 随机选择 k 个索引
    indices = np.random.choice(arr.shape[0], size=k, replace=False)
    
    # 根据索引提取行,并取前两列
    sampled_array = arr[indices]
    
    return sampled_array

def sample_points_within_bbox_and_mask(bbox, mask, N):
    """
    从给定的bbox和mask中采样N个有效的(u, v)点。
    
    参数：
    - bbox: 形状为(4,)的bbox数组,格式为[u_left, v_left, u_right, v_right]。
    - mask: 大小为(H, W)的布尔数组,表示图像中每个像素是否有效。
    - N: 需要采样的点数。
    
    返回：
    - 一个形状为(N, 2)的numpy数组,表示采样的(u, v)坐标。
      其中,N是有效采样点的数量。
    """
    # 获取bbox的坐标
    u_left, v_left, u_right, v_right = bbox

    # bbox的宽高
    bbox_u_range = u_right - u_left
    bbox_v_range = v_right - v_left

    # 计算步长,使得采样点的数量接近N
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

def joint_data_to_transform_np(
    joint_type, # "prismatic" or "revolute"
    joint_dir, # unit vector np.array([3, ])
    joint_start,
    joint_state_ref2tgt # joint_state_tgt - joint_state_ref, a constant
):
    """
    计算 ref_frame 和 tgt_frame 上的对应点在 camera 坐标系下坐标的转换矩阵
    """
    # 根据 joint_type 和 joint_dir 和 (joint_state2 - joint_state1) 得到 T_ref2tgt
    T_ref2tgt = np.eye(4)
    if joint_type == "prismatic":
        # coor_tgt = coor_ref + joint_dir * (joint_state_tgt - joint_state_ref)
        T_ref2tgt[:3, 3] = joint_dir * joint_state_ref2tgt
    elif joint_type == "revolute":
        # coor_tgt = (coor_ref - joint_start) @ Rref2tgt.T + joint_start
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
    # 根据 joint_type 和 joint_dir 和 (joint_state2 - joint_state1) 得到 T_ref2tgt
    T_ref2tgt = torch.eye(4, device=joint_dir.device)
    joint_dir = joint_dir / torch.norm(joint_dir)
    
    if joint_type == "prismatic":
        # coor_tgt = coor_ref + joint_dir * (joint_state_tgt - joint_state_ref)
        T_ref2tgt[:3, 3] = joint_dir * joint_state_ref2tgt
    elif joint_type == "revolute":
        # coor_tgt = (coor_ref - joint_start) @ Rref2tgt.T + joint_start
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
    设置 Python, NumPy 和 PyTorch 的随机种子,确保实验的可重复性
    :param seed: 要设置的随机种子
    """
    # 设置 Python 的随机种子
    random.seed(seed)
    
    # 设置 NumPy 的随机种子
    np.random.seed(seed)
    
    # 设置 PyTorch 的随机种子
    torch.manual_seed(seed)
    
    # 如果使用 GPU,还需要设置 CUDA 随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 的种子
        
    # 为了确保结果可重复,设置 cudnn 的确定性算法
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

def filter_tracks_by_visibility(pred_visibility, threshold=0.9):
    """
        pred_visibility: torch.tensor([T, M])
        返回一个大小为 M 的布尔数组,表示每个点的可见性，如果一个点在 T * thre 的 frame 都是可见的, 则认为他是 visible 的
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
    
    return pred_tracks_2d

def filter_tracks_by_consistency(tracks, threshold=0.1):
    """
    根据 tracks 的 3d consistency 来筛选出平稳变化的 tracks
    tracks: T, M, 3
    return: 
        consis_mask: (M, ), boolen
    """
    # 获取时间步数 T 和轨迹数量 M
    T, M, _ = tracks.shape
    
    # 初始化一致性掩码
    consis_mask = np.ones(M, dtype=bool)
    
    # 遍历每个轨迹
    for m in range(M):
        # 计算该轨迹的变化量
        diffs = np.linalg.norm(tracks[1:T, m] - tracks[0:T-1, m], axis=1)
        
        # 如果有任意变化量超过阈值，则标记为不一致
        if np.any(diffs > threshold):
            consis_mask[m] = False
    
    return consis_mask


def classify_dynamic(mask, moving_points, static_points):
    """
    根据A和B点集的最近邻分类让mask中为True的点分类。

    参数:
        mask (np.ndarray): 大小为(H, W)的应用匹配网络,mask=True的点需要分类
        moving_points (np.ndarray): 大小为(M, 2)的A类点集,具有(u, v)坐标
        static_points (np.ndarray): 大小为(N, 2)的B类点集,具有(u, v)坐标

    返回:
        dynamic_mask (np.ndarray): 大小为(H, W)的分类结果
    """
    H, W = mask.shape

    # 构建KD树以加快最近邻搜索,确保用(v, u)坐标格式
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

    return dynamic_mask

def get_depth_mask(depth, K, Tw2c, height=0.02):
    """
    返回一个 mask, 标记出 depth 不为 0, 且重投影会世界坐标系不在地面上的点
    depth: np.array([H, W])
    height: 默认高于 2 cm 的点才会保留
    """
    H, W = depth.shape
    pc_camera = depth_image_to_pointcloud(depth, None, K) # H*W, 3
    pc_world = camera_to_world(pc_camera, Tw2c) # H*W, 3
    height_mask = (pc_world[:, 2] > height).reshape(H, W) # H*W
    depth_mask = (depth > 0.0) & height_mask
    return depth_mask

def get_depth_mask_seq(depth_seq, K, Tw2c, height=0.01):
    """
    返回一个 mask, 标记出 depth 不为 0, 且重投影会世界坐标系不在地面上的点
    depth: np.array([H, W])
    height: 默认高于 1 cm 的点才会保留
    """
    depth_mask_seq = depth_seq > 0
    for i in range(len(depth_seq)):
        depth_mask_seq[i] = get_depth_mask(depth_seq[i], K, Tw2c, height)
    return depth_mask_seq

def quantile_sampling(arr, M):
    """
    从数组中按分位数抽样,返回代表性数据点。
    
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

def compute_normals(target_pc, k_neighbors=30):
    """
    计算目标点云的法向量
    :param target_pc: numpy 数组,形状为 (N, 3),目标点云
    :param k_neighbors: 计算法向量时的近邻数
    :return: 法向量数组,形状为 (N, 3)
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(target_pc)
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=k_neighbors))
    normals = np.asarray(pcd.normals)
    return normals

def find_correspondences(ref_pc, target_pc, max_distance=0.01):
    """
    使用 KD-Tree 进行最近邻搜索,找到参考点云 ref_pc 在目标点云 target_pc 中的最近邻
    :param ref_pc: numpy 数组,形状 (M, 3),参考点云
    :param target_pc: numpy 数组,形状 (N, 3),目标点云
    :param max_distance: 最大距离,超过这个距离的点会被忽略
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
    """
    假设满足 R @ a = b, 求R
    """
    
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


# 从点云中获取 bbox
def compute_bbox_from_pc(points, offset=0):
    """
    计算点云的边界框,并根据offset扩张bbox。

    参数:
    - points: numpy数组,形状为 (N, 3),表示点云的坐标。
    - offset: 浮点数,表示bbox的扩张量。

    返回:
    - bbox_min: numpy数组,形状为 (3,),表示bbox的最小角点。
    - bbox_max: numpy数组,形状为 (3,),表示bbox的最大角点。
    """
    bbox_min = np.min(points, axis=0) - offset
    bbox_max = np.max(points, axis=0) + offset
    return bbox_min, bbox_max

def sample_points_on_bbox_surface(bbox_min, bbox_max, num_samples):
    """
    在bbox的表面上采样一些点。

    参数:
    - bbox_min: numpy数组,形状为 (3,),表示bbox的最小角点。
    - bbox_max: numpy数组,形状为 (3,),表示bbox的最大角点。
    - num_samples: 整数,表示要采样的点的数量。

    返回:
    - samples: numpy数组,形状为 (num_samples, 3),表示采样点的坐标。
    """
    # 计算bbox的尺寸
    bbox_size = bbox_max - bbox_min

    # 采样点在6个面上均匀分布
    samples_per_face = num_samples // 6
    samples = []

    for i in range(6):
        if i < 2:
            # 前后两个面 (x轴方向)
            x = bbox_min[0] if i == 0 else bbox_max[0]
            x = np.ones(samples_per_face) * x
            y = np.random.uniform(bbox_min[1], bbox_max[1], samples_per_face)
            z = np.random.uniform(bbox_min[2], bbox_max[2], samples_per_face)
        elif i < 4:
            # 左右两个面 (y轴方向)
            y = bbox_min[1] if i == 2 else bbox_max[1]
            y = np.ones(samples_per_face) * y
            x = np.random.uniform(bbox_min[0], bbox_max[0], samples_per_face)
            z = np.random.uniform(bbox_min[2], bbox_max[2], samples_per_face)
        else:
            # 上下两个面 (z轴方向)
            z = bbox_min[2] if i == 4 else bbox_max[2]
            z = np.ones(samples_per_face) * z
            x = np.random.uniform(bbox_min[0], bbox_max[0], samples_per_face)
            y = np.random.uniform(bbox_min[1], bbox_max[1], samples_per_face)

        face_samples = np.column_stack((x, y, z))
        samples.append(face_samples)

    # 合并所有面的采样点
    samples = np.vstack(samples)

    return samples

def normalize_cos_map_exp(cosine_similarity_map, sharpen_factor=5):
    """
    对于输入的 cosine_similarity_map 进行归一化变为一个概率分布
    sharpen_factor: 用于调整分布的相对大小, sharpen_factor 越大分布越尖锐
    """
    # 使用 exp(cos) 计算
    exp_map = np.exp(cosine_similarity_map * sharpen_factor)
    # 转换为概率图
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
    # 通过KDTree搜索找到在给定半径内的点
    [k, idx, _] = pcd_tree.search_radius_vector_3d(contact3d, radius)
    
    # 创建一个布尔掩码，初始化为全False
    mask = np.zeros(len(point_clouds), dtype=bool)
    # 将在给定半径内的点对应的索引位置设为True
    mask[idx] = True
    return mask

def fit_plane_normal(points):
    """
        输入一个点云, 用一个平面拟合它, 并返回该平面的单位法向量
        point_cloud: np.array([N, 3])
    """
    # 计算点的均值
    centroid = np.mean(points, axis=0)

    # 中心化点云
    centered_points = points - centroid

    # 计算协方差矩阵
    cov_matrix = np.cov(centered_points, rowvar=False)

    # 进行特征值分解
    _, _, vh = np.linalg.svd(cov_matrix)

    # 法向量是特征值分解的最后一个特征向量
    normal_vector = vh[-1]

    # 归一化法向量
    normalized_normal_vector = normal_vector / np.linalg.norm(normal_vector)

    return centroid, normalized_normal_vector


def fit_plane_ransac(points, threshold=0.01, max_iterations=100, visualize=False):
    """
    使用 RANSAC 算法从点云中拟合一个平面，并返回该平面的单位法向量
    points: np.array([N, 3]) - 输入的点云
    threshold: float - 允许的最大距离，超过该距离的点被视为离群点
    max_iterations: int - RANSAC 的最大迭代次数
    """
    if len(points) < 3:
        print("Less than 3 points in fit_plane_ransac")
        return np.array([0, 0, 1])
    
    best_normal = None
    best_inliers_mask = None
    best_inliers_count = 0

    for _ in range(max_iterations):
        # 随机选择 3 个点来定义一个平面
        indices = np.random.choice(points.shape[0], 3, replace=False)
        sample_points = points[indices]

        # 计算平面的法向量
        _, normal_vector = fit_plane_normal(sample_points)

        # 计算每个点到平面的距离
        dists = np.abs(np.dot(points - sample_points[0], normal_vector))

        # 计算内点
        inliers_mask = dists < threshold
        inliers_count = inliers_mask.sum()

        # 更新最佳平面
        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_normal = normal_vector
            best_inliers_mask = inliers_mask

    print("num of inliers:", best_inliers_count, " / ", len(points))
    
    if visualize:
        v_points = points
        v_colors = np.zeros_like(points)
        # inliers 用绿色, outliers 用红色
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
        返回去除掉 direction 上分量后的 vector
        vector: np.array([3, ])
        direction: np.array([3, ])
    """
    # 计算方向的单位向量
    direction_unit = direction / np.linalg.norm(direction)
    
    # 计算原始向量在方向上的投影
    projection_length = np.dot(vector, direction_unit)
    projection = projection_length * direction_unit
    
    # 去除方向上的分量
    result_vector = vector - projection
    
    if return_normalized:
        result_vector = result_vector / np.linalg.norm(result_vector)
    
    return result_vector


def classify_open_close(tracks3d, moving_mask, visualize=False):
    """
    判断输入轨迹到底是打开物体还是关闭物体
    tracks3d: T, N, 3, 来自于 filter 过后的 tracks3d
    moving_mask: N
    """
    static_mask = ~moving_mask
    tracks_3d_moving_c, tracks_3d_static_c = tracks3d[:, moving_mask, :], tracks3d[:, static_mask, :]
    moving_mean_start, moving_mean_end = tracks_3d_moving_c[0].mean(0), tracks_3d_moving_c[-1].mean(0)
    static_mean_start, static_mean_end = tracks_3d_static_c[0].mean(0), tracks_3d_static_c[-1].mean(0)
    
    # 首先根据 tracks3d_moving 和 tracks3d_static 类中心的方差判断当前 track 随着时间是 open 还是 close
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
    """
    将 joint_dict 中存储的 joint_dir 进行反转，并同步更改 joint_states
    """
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
    K, # 相机内参
    query_depth, # H, W
    query_dynamic, # H, W
    ref_depths,  # T, H, W
    # ref_dynamics, # T, H, W
    joint_type,
    joint_dir,
    joint_start,
    query_state,
    ref_states,
    depth_tolerance=0.01, # 能容忍 1cm 的深度不一致
    visualize=False
):
    """
        根据当前的 joint state
        验证所有的 moving points, 把不确定的 points 标记为 unknown
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
    
    # 获取当前帧 MOVING_LABEL 的像素坐标
    moving_mask = query_dynamic == MOVING_LABEL
    if not np.any(moving_mask):
        return query_dynamic_updated        

    y, x = np.where(moving_mask) # N
    pc_moving = depth_image_to_pointcloud(query_depth, moving_mask, K)  # (N, 3)
    pc_moving_aug = np.concatenate([pc_moving, np.ones((len(pc_moving), 1))], axis=1)  # (N, 4)
    
    # 批量计算 moving_pc 在其他帧的 3d 坐标
    pc_pred = np.einsum('tij,jk->tik', Tquery2refs, pc_moving_aug.T).transpose(0, 2, 1)[:, :, :3] # T, N, 3
    
    # 投影到所有帧
    uv_pred, depth_pred = camera_to_image(pc_pred.reshape(-1, 3), K) # T*N, 2
    uv_pred_int = np.floor(uv_pred.reshape(T, len(pc_moving), 2)).astype(int) # T, N, 2
    depth_pred = depth_pred.reshape(T, len(pc_moving)) # T, N
    
    # 在这里进行一个筛选, 把那些得不到有效 depth_obs 的 moving point 也标记为 Unknown TODO: 这里会不会过于严格
    valid_idx = (uv_pred_int[..., 0] >= 0) & (uv_pred_int[..., 0] < W) & \
                (uv_pred_int[..., 1] >= 0) & (uv_pred_int[..., 1] < H) # T, N
    # 且只有一个时间帧观测不到就认为得不到有效观测
    valid_idx = valid_idx.all(axis=0) # N
    uv_pred_int = uv_pred_int[:, valid_idx] # T, M, 2
    depth_pred = depth_pred[:, valid_idx] # T, M
    
    # TODO: 是否要严格到必须 score_moving > score_static 的点才被保留
    # TODO：获取目标帧的真实深度, 是不是要考虑 depth_ref 等于 0 的情况是否需要拒绝
    
    T_idx = np.arange(T)[:, None]
    depth_obs = ref_depths[T_idx, uv_pred_int[..., 1], uv_pred_int[..., 0]]  # T, M
    
    # 计算误差并更新 dynamic_mask， M, 只要有一帧拒绝，则置为 UNKNOWN
    unknown_mask = (depth_pred + depth_tolerance < depth_obs).any(axis=0)  # M
    query_dynamic_updated[y[valid_idx][unknown_mask], x[valid_idx][unknown_mask]] = UNKNOWN_LABEL
    
    if visualize:
        viewer = napari.view_image((query_dynamic != 0).astype(np.int32), rgb=False)
        viewer.title = "filter current dynamic mask using other frames"
        # viewer.add_labels(mask_seq.astype(np.int32), name='articulated objects')
        viewer.add_labels(query_dynamic.astype(np.int32), name='before filtering')
        viewer.add_labels(query_dynamic_updated.astype(np.int32), name='after filtering')
        napari.run()
    
    return query_dynamic_updated  


def vis_tracks2d_napari(image_frames, tracks_2d, colors=None, viewer_title="napari"):
    """
    Args:
        image_frames: np.array([T, H, W, C])
        tracks_2d: np.array of shape (T, M, 2), (u, v)
        colors: np.array of shape (M, 3)
    """
    if isinstance(tracks_2d, torch.Tensor):
        tracks_2d = tracks_2d.cpu().numpy()
        
    T, M, _ = tracks_2d.shape
    viewer = napari.view_image(image_frames, rgb=True)
    viewer.title = viewer_title
    
    # 把 tracks_2d 转换成 napari 支持的格式
    napari_data = []
    for i in range(T):
        track_2d = tracks_2d[i] # M, 2
        track_2d_with_t = np.concatenate([np.ones((track_2d.shape[0], 1)) * i, track_2d], axis=1) # M, 3
        # 把 (u, v) 两个维度交换
        track_2d_with_t = track_2d_with_t[:, [0, 2, 1]]
        napari_data.append(track_2d_with_t)
    napari_data = np.concatenate(napari_data, axis=0)
    
    if colors is None:
        colors = np.random.rand(M, 3)
        # 将 M, 3 大小的 colors 变换为 T*M, 3 的大小
    colors = np.tile(colors, (T, 1))
        
    viewer.add_points(napari_data, size=3, name='tracks_2d', opacity=1., face_color=colors)
    
    napari.run()


def line_to_line_distance(P1, d1, P2, d2):
    """
    计算三维空间中两条直线的最短距离
    
    参数:
        P1 (np.array): 直线1上一点的坐标，形状 (3,)
        d1 (np.array): 直线1的单位方向向量，形状 (3,)
        P2 (np.array): 直线2上一点的坐标，形状 (3,)
        d2 (np.array): 直线2的单位方向向量，形状 (3,)
    
    返回:
        float: 两条直线的最短距离
    """
    # 计算连接向量 v = P2 - P1
    v = P2 - P1
    
    # 计算方向向量的叉积 n = d1 × d2
    n = np.cross(d1, d2)
    
    # 计算叉积的范数 ||n||
    n_norm = np.linalg.norm(n)
    
    if n_norm > 1e-10:  # 如果两条直线不平行（n 不是零向量）
        # 最短距离 = |v · n| / ||n||
        distance = np.abs(np.dot(v, n)) / n_norm
    else:  # 如果两条直线平行（n ≈ 0）
        # 计算 v 在 d1 垂直方向的分量：v_perp = v - (v·d1) * d1
        v_parallel = np.dot(v, d1) * d1
        v_perp = v - v_parallel
        distance = np.linalg.norm(v_perp)
    
    return distance


def custom_linspace(a, b, delta):
    if delta <= 0:
        raise ValueError("delta must be positive")
    
    # 确定递增/递减方向
    direction = 1 if b >= a else -1
    diff = abs(b - a)
    
    # 计算最大完整步数
    n = int(np.floor(diff / delta))
    
    # 生成基础点序列
    if n == 0:
        return np.array([b])
    else:
        points = a + direction * delta * np.arange(1, n+1)
        
        # 判断是否需要追加端点b
        last = points[-1]
        if (direction == 1 and not np.isclose(last, b) and last < b) or \
           (direction == -1 and not np.isclose(last, b) and last > b):
            points = np.append(points, b)
        
        return points


def extract_pos_quat_from_matrix(T):
    """
    从 4x4 变换矩阵中提取位置和四元数
    输入:
        T: Tph2w, np.array(4, 4), 变换矩阵
    输出:
        pos: np.array([3, ]), 位置向量
        quat: np.array([4, ]), 四元数 (w, x, y, z) 顺序
    """
    # 提取位置向量
    pos = T[:3, 3]

    # 提取旋转矩阵并转换为四元数
    R_matrix = T[:3, :3]
    quat = R.from_matrix(R_matrix).as_quat(scalar_first=False)  # 返回 (x, y, z, w) 顺序
    quat = np.array([quat[3], quat[0], quat[1], quat[2]])  # 转换为 (w, x, y, z) 顺序

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
    计算两个 Transformation 之间的距离
    mat1: (4, 4)
    """
    def translation_distance(mat1, mat2):
        translation1 = mat1[:3, 3]
        translation2 = mat2[:3, 3]
        return np.linalg.norm(translation1 - translation2)
    
    def rotation_difference(mat1, mat2):
        R1 = mat1[:3, :3]
        R2 = mat2[:3, :3]
        # 计算旋转矩阵之间的差异
        angle_diff = np.arccos((np.trace(R1.T @ R2) - 1) / 2)
        return angle_diff
    
    trans_dist = translation_distance(mat1, mat2)
    rot_dist = rotation_difference(mat1, mat2)
    return trans_dist + rot_dist  # 或者其他加权方式


def explore_actually_valid(result):
    joint_type = result["joint_type"]
    joint_delta = result["joint_state_end"] - result["joint_state_start"]
    if joint_type == "prismatic":
        # if joint_delta < 0.09:
        #     pass
        return joint_delta >= EXPLORE_PRISMATIC_VALID
    elif joint_type == "revolute":
        return np.rad2deg(joint_delta) >= EXPLORE_REVOLUTE_VALID

def numpy_to_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # 将 NumPy 数组转换为列表
    elif isinstance(obj, np.float32):
        return float(obj)  # 将 NumPy float32 转换为 Python float
    elif isinstance(obj, np.int32):
        return int(obj)  # 将 NumPy int32 转换为 Python int
    raise TypeError(f"Type {type(obj)} not serializable")

    
if __name__ == "__main__":
    print(custom_linspace(5, 1, 2.5))   # [2.5, 1.0]
    print(custom_linspace(0, 5, 1))      # [1., 2., 3., 4., 5.]
    print(custom_linspace(10, 5, 2))     # [8., 6., 5.]
    print(custom_linspace(3, 3, 0.5))    # [3]
    print(custom_linspace(2, 5, 0.5))    # [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

