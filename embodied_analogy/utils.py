import pygame
from PIL import Image
import numpy as np

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