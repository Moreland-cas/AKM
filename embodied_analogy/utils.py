import pygame
from PIL import Image
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import open3d as o3d

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
    
def world_to_camera(world_point, K, Tw2c):
    """
    将世界坐标系中的点转换到相机坐标系

    Args:
        world_point (np.ndarray): 3D世界坐标点，形状 (3,)
        K (np.ndarray): 相机的内参矩阵，3x3
        Tw2c (np.ndarray): 相机的外参矩阵 (4x4)，从世界坐标系到相机坐标系的变换矩阵

    Returns:
        np.ndarray: 相机坐标系中的点 (3,) 
    """
    # 将世界坐标点扩展为齐次坐标
    world_point_homogeneous = np.append(world_point, 1)  # 变为 (4,)
    
    # 使用外参矩阵将世界坐标转换为相机坐标
    camera_point_homogeneous = np.dot(Tw2c, world_point_homogeneous)  # 得到相机坐标系中的点 (4,)
    
    # 将相机坐标转换为非齐次坐标
    camera_point = camera_point_homogeneous[:3]  # 去掉齐次坐标的最后一维，返回 (3,)
    
    return camera_point

def camera_to_image(camera_point, K, image_width, image_height):
    """
    将相机坐标系中的点投影到图像平面，并将像素坐标归一化到 [0, 1] 范围内

    Args:
        camera_point (np.ndarray): 相机坐标系中的点，形状 (3,)
        K (np.ndarray): 相机内参矩阵，3x3
        image_width (int): 图像的宽度
        image_height (int): 图像的高度

    Returns:
        (float, float): 像素坐标 (u, v)，归一化到 [0, 1]
    """
    # 使用内参矩阵进行投影，得到像素坐标
    u, v, depth = np.dot(K, camera_point)  # (u, v, depth)，注意这里是齐次坐标 (3,)
    
    # 将齐次坐标除以深度得到非齐次像素坐标
    u /= depth
    v /= depth
    
    # 将像素坐标归一化到 [0, 1] 范围内
    u_normalized = u / image_width
    v_normalized = v / image_height
    
    return u_normalized, v_normalized

def world_to_normalized_uv(world_point, K, Tw2c, image_width, image_height):
    """
    将世界坐标系中的点转换为归一化的像素坐标 (u, v) [0, 1]

    Args:
        world_point (np.ndarray): 世界坐标系中的点，形状 (3,)
        K (np.ndarray): 相机的内参矩阵，3x3
        Tw2c (np.ndarray): 相机的外参矩阵 (4x4)，从世界坐标系到相机坐标系的变换矩阵
        image_width (int): 图像宽度
        image_height (int): 图像高度

    Returns:
        (float, float): 归一化的像素坐标 (u, v)，范围在 [0, 1]
    """
    # 将世界坐标转换为相机坐标
    camera_point = world_to_camera(world_point, K, Tw2c)
    
    # 将相机坐标投影到图像平面，并归一化到 [0, 1]
    u_normalized, v_normalized = camera_to_image(camera_point, K, image_width, image_height)
    
    return u_normalized, v_normalized

def draw_red_dot(image: Image, u: float, v: float, radius: int = 1):
    """
    在给定的 PIL 图像上，在归一化坐标 (u, v) 处画一个红色的点。
    
    参数：
    - image: 输入的 PIL Image 对象
    - u: x 坐标，归一化到 [0, 1]
    - v: y 坐标，归一化到 [0, 1]
    
    返回：
    - 修改后的 PIL Image 对象
    """
    # 获取图像的宽度和高度
    width, height = image.size
    
    # 将归一化坐标转换为像素坐标
    x = int(u * width)
    y = int(v * height)
    
    # 创建 ImageDraw 对象
    image_draw = image.copy()
    draw = ImageDraw.Draw(image_draw)
    
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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
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
        font_size (int): 字体大小（默认30）。
    
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

def uv_to_camera(u, v, depth, K, image_width, image_height):
    """
    将归一化的像素坐标和深度值转换为三维空间中的点（相机坐标系）。
    
    Args:
    - u (float): 归一化的水平像素坐标，范围 [0, 1]
    - v (float): 归一化的垂直像素坐标，范围 [0, 1]
    - depth (float): 深度值，表示该像素点到相机的距离（单位：米）
    - K (np.array): 相机内参矩阵(3x3), 包括焦距和主点坐标
    - image_width (int): 图像的宽度（单位：像素）
    - image_height (int): 图像的高度（单位：像素）
    
    Returns:
    - (X, Y, Z) (tuple): 在相机坐标系下的三维坐标（单位：米）
    """
    # 1. 计算实际的像素坐标
    x_pixel = u * image_width
    y_pixel = v * image_height

    # 2. 获取相机内参
    f_x = K[0, 0]  # 水平焦距 (单位: 像素)
    f_y = K[1, 1]  # 垂直焦距 (单位: 像素)
    c_x = K[0, 2]  # 主点 x 坐标 (单位: 像素)
    c_y = K[1, 2]  # 主点 y 坐标 (单位: 像素)
    
    # 3. 计算三维坐标 (X, Y, Z)
    Z = depth  # 深度值，单位：米
    X = (x_pixel - c_x) * Z / f_x
    Y = (y_pixel - c_y) * Z / f_y
    
    return np.array([X, Y, Z])

def camera_to_world(point_camera, extrinsic_matrix):
    """
    将相机坐标系中的点转换到世界坐标系。
    
    Args:
    - point_camera (np.array): 相机坐标系中的点
    - extrinsic_matrix (np.array): Tw2c, 外参矩阵(3, 4), 形式为 [R | t]
    
    Returns:
    - point_world (np.array): 世界坐标系中的点 (3x1)
    """
    # 从外参矩阵中提取旋转矩阵 R 和平移向量 t
    R = extrinsic_matrix[:, :3]  # 旋转矩阵
    t = extrinsic_matrix[:, 3]   # 平移向量
    
    # 计算从相机坐标系到世界坐标系的转换
    point_world = np.dot(R.T, (point_camera - t))  # R^T * (P_camera - t)
    
    return point_world

def visualize_pc(points, colors):
    # visualize pointcloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])