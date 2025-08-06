import json
import os
import numpy as np
from PIL import Image
from embodied_analogy.utility.utils import (
    visualize_pc,
    add_text_to_image,
    camera_to_image
)
from embodied_analogy.representation.obj_repr import Obj_repr
from embodied_analogy.representation.basic_structure import Frame
from graspnetAPI.grasp import Grasp
from embodied_analogy.utility.constants import MOVING_LABEL, STATIC_LABEL

from PIL import Image, ImageDraw
import math

mobile_color = np.array([0, 255, 0]) 
static_color = np.array([255, 0, 0]) 

def farthest_point_sampling_2d(points, M):
    """
    Perform farthest point sampling on a 2D point set.
    
    Args:
        points: np.ndarray of shape (N, 2), input 2D points
        M: int, number of points to retain
    
    Returns:
        mask: np.ndarray of shape (N,), boolean mask where True indicates retained points
    """
    N = points.shape[0]
    if M > N or M <= 0:
        raise ValueError("M must be between 1 and N")
    
    # Initialize mask
    mask = np.zeros(N, dtype=bool)
    
    # Randomly select the first point
    first_idx = np.random.randint(0, N)
    mask[first_idx] = True
    
    # Compute all pairwise distances
    dist_matrix = np.sqrt(np.sum((points[:, np.newaxis] - points) ** 2, axis=2))
    
    # Track minimum distance from each point to selected points
    min_distances = dist_matrix[first_idx].copy()
    
    # Select M-1 more points
    for _ in range(M - 1):
        # Find point with maximum minimum distance to selected points
        next_idx = np.argmax(min_distances)
        mask[next_idx] = True
        
        # Update minimum distances
        min_distances = np.minimum(min_distances, dist_matrix[next_idx])
    
    return mask

def draw_directed_axis(image, start_point, end_point, line_color=(255, 0, 0), line_width=2, arrow_size=10):
    """
    在 PIL Image 上绘制带箭头的有向轴
    参数:
    - image: PIL Image 对象
    - start_point: 起点坐标 (u, v)
    - end_point: 终点坐标 (u, v)
    - line_color: 线条颜色 (R,G,B)
    - line_width: 线条宽度
    - arrow_size: 箭头大小
    """
    # 创建 ImageDraw 对象
    draw = ImageDraw.Draw(image)
    
    if isinstance(start_point, np.ndarray):
        start_point = tuple(start_point.astype(np.int32))
    if isinstance(end_point, np.ndarray):
        end_point = tuple(end_point.astype(np.int32))
    
    line_width = int(line_width)
    arrow_size = int(arrow_size)
    
    # start_point, end_point = start_point.astype(np.int32), int(end_point).astype(np.int32)
    # 绘制主线段
    draw.line([start_point, end_point], fill=line_color, width=line_width)
    
    # 计算箭头方向
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    
    # 计算线段长度
    length = math.sqrt(dx**2 + dy**2)
    
    if length == 0:
        return  # 避免除零错误
    
    # 单位向量
    ux = dx / length
    uy = dy / length
    
    # 计算箭头两个侧翼点
    # 箭头角度（相对于主线方向，30度）
    arrow_angle = math.radians(30)
    
    # 箭头左翼点
    arrow_left_x = end_point[0] - arrow_size * (ux * math.cos(arrow_angle) + uy * math.sin(arrow_angle))
    arrow_left_y = end_point[1] - arrow_size * (uy * math.cos(arrow_angle) - ux * math.sin(arrow_angle))
    
    # 箭头右翼点
    arrow_right_x = end_point[0] - arrow_size * (ux * math.cos(arrow_angle) - uy * math.sin(arrow_angle))
    arrow_right_y = end_point[1] - arrow_size * (uy * math.cos(arrow_angle) + ux * math.sin(arrow_angle))
    
    # 绘制箭头（填充三角形）
    draw.polygon([
        end_point,
        (arrow_left_x, arrow_left_y),
        (arrow_right_x, arrow_right_y)
    ], fill=line_color)
    return image


data_path = "/home/zby/Programs/Embodied_Analogy/assets/logs_draw"
save_folder = f"/home/zby/Programs/Embodied_Analogy/scripts/draw_figs/paper_figs/reconstruct"
method_names = ["gflow", "gpnet", "ours"]
obj_idx_list = os.listdir(os.path.join(data_path, method_names[0]))

def draw_axis_on_image(
    image,
    obj_repr,
    draw_gt=False,
    draw_pred=False
):
    """
    在 image 上画出 joint axis
    pivot_point, joint_dir 都是在 camera 坐标系下的
    需要根据投影位置对于 joint 进行一个位置调整
    """
    joint_length = 0.1
    gt_joint_dict = obj_repr.get_joint_param(resolution="gt", frame="camera")
    fine_joint_dict = obj_repr.get_joint_param(resolution="fine", frame="camera")
    
    # 将 pil image 进行缩放, 同时将 K 进行缩放
    image_width, image_height = image.size
    # scale = 3200 / image_width
    resize_scale = 1.
    image = image.resize((int(image_width * resize_scale), int(image_height * resize_scale)), Image.Resampling.LANCZOS)
    
    K = obj_repr.K
    K = K * resize_scale
    K[-1, -1] = 1
    
    joint_type = gt_joint_dict["joint_type"]
    gt_joint_start = gt_joint_dict["joint_start"]
    gt_joint_end = gt_joint_start + gt_joint_dict["joint_dir"] * joint_length
    
    gt_start_point = camera_to_image(gt_joint_start[None], K)[0][0]
    gt_end_point = camera_to_image(gt_joint_end[None], K)[0][0]
    
    yellow = (255, 200, 0)
    blue = (0, 0, 255)
    if draw_gt:
        # 缩放 
        cur_2d_length = np.linalg.norm(gt_end_point - gt_start_point)
        joint_scale = (image.height / 6) / max(cur_2d_length, 1)
        gt_end_point = gt_start_point + (gt_end_point - gt_start_point) * joint_scale
        
        image = draw_directed_axis(image, gt_start_point, gt_end_point, line_color=blue, line_width=2*resize_scale, arrow_size=10*resize_scale)
        
    if draw_pred:
        pred_joint_start = fine_joint_dict["joint_start"]
        pred_joint_end = pred_joint_start + fine_joint_dict["joint_dir"] * joint_length
        
        pred_start_point = camera_to_image(pred_joint_start[None], K)[0][0]
        pred_end_point = camera_to_image(pred_joint_end[None], K)[0][0]
    
        cur_2d_length = np.linalg.norm(pred_end_point - pred_start_point)
        joint_scale = (image.height / 6) / max(cur_2d_length, 1)
        
        if joint_type == "prismatic":
            draw_start_point = gt_start_point
            draw_end_point = draw_start_point + (pred_end_point - pred_start_point) * joint_scale
        else:
            # TODO 對於 revolute 的情況, 修正 pred_start_point 使得他距離 gt_joint 所確定的 line 最近
            draw_start_point = pred_start_point
            draw_end_point = pred_start_point + (pred_end_point - pred_start_point) * joint_scale
            
        image = draw_directed_axis(image, draw_start_point, draw_end_point, line_color=yellow, line_width=2*resize_scale, arrow_size=10*resize_scale)
    return image

def draw_tracking_trajectories(image, tracks, colors):
    """
    在 PIL 图像上绘制二维跟踪轨迹。
    
    参数:
    - image: PIL 图像对象
    - tracks: (T, M, 2) 的 NumPy 数组，表示 M 个轨迹，每个轨迹有 T 个点，每个点是 (x, y) 坐标
    - colors: (3,) 的 NumPy 数组，表示每个轨迹的颜色，颜色格式为 (R, G, B)
    """
    # 确保 tracks 和 colors 的形状正确
    T, M, _ = tracks.shape
    colors = np.tile(colors, (M, 1))
    
    # 创建一个绘图对象
    draw = ImageDraw.Draw(image)
    
    # 遍历每个轨迹
    for m in range(M):
        track = tracks[:, m, :]  # 获取第 m 个轨迹 (T, 2)
        color = colors[m]  # 获取第 m 个轨迹的颜色 (R, G, B)
        
        # 遍历轨迹中的每个点对，绘制线段
        for t in range(T - 1):
            start_point = tuple(track[t])  # 起点 (x1, y1)
            end_point = tuple(track[t + 1])  # 终点 (x2, y2)
            # TODO 對 color 做一個 漸變
            tmp_color = t/T * color + (1-t/T) * np.array([128, 128, 128])
            tmp_color = tuple(tmp_color.astype(int)) 
            draw.line([start_point, end_point], fill=tmp_color, width=1)
    
    return image

def idx_available(obj_idx):
    """
    分别对于 三个方法读取 obj_idx 下的 recon_result.json, 都有 valid recon 才去绘制
    """
    data_path = "/home/zby/Programs/Embodied_Analogy/assets/logs_draw"
    ours_json = json.load(open(os.path.join(data_path, "ours", str(obj_idx), "recon_result.json"), "r"))
    gpnet_json = json.load(open(os.path.join(data_path, "gpnet", str(obj_idx), "recon_result.json"), "r"))
    gflow_json = json.load(open(os.path.join(data_path, "gflow", str(obj_idx), "recon_result.json"), "r"))
    if ours_json["has_valid_recon"] and gpnet_json["has_valid_recon"] and gflow_json["has_valid_recon"]:
        return True
    else:
        return False

obj_idx_list_filtered = [obj_idx for obj_idx in obj_idx_list if idx_available(obj_idx)]
for method_name in method_names:
    for obj_idx in obj_idx_list_filtered:
        os.makedirs(os.path.join(save_folder, method_name, obj_idx), exist_ok=True)
        obj_repr: Obj_repr
        obj_repr = Obj_repr.load(f"{data_path}/{method_name}/{obj_idx}/obj_repr.npy")
        
        init_rgb = obj_repr.initial_frame.rgb
        init_pil = Image.fromarray(init_rgb)
        # init_pil = init_pil.resize((3200, 2400))
        
        recon_result = json.load(open(os.path.join(data_path, method_name, str(obj_idx), "recon_result.json"), "r"))
        fine_loss_dict = recon_result["fine_loss"]
        # joint_type = recon_result["gt_w"]["joint_type"]
        type_error = int(fine_loss_dict["type_err"])
        pos_error = fine_loss_dict["pos_err"] * 100
        angle_error = np.rad2deg(fine_loss_dict["angle_err"])
        err_str = f'\nType Err: {type_error}\nPivot Err: {pos_error:.1f}cm\nAngle Err: {angle_error:.1f}°'
        
        if method_name == "ours":
            # 绘制不同颜色的点云, 绘制轴
            dynamic_mask = obj_repr.kframes[0].dynamic_mask
            moving_mask = dynamic_mask == MOVING_LABEL
            static_mask = dynamic_mask == STATIC_LABEL
            
            ours_rgb = obj_repr.initial_frame.rgb
            
            # 绘制 dynamic mask
            alpha = 0.8
            ours_rgb[static_mask] = alpha * init_rgb[static_mask] + (1-alpha) * static_color 
            ours_rgb[moving_mask] = alpha * init_rgb[moving_mask] + (1-alpha) * mobile_color 
            
            # 绘制 tracking
            track2d_seq = obj_repr.frames.track2d_seq # T, M, 2
            moving_mask = obj_repr.frames.moving_mask
            ours_pil = Image.fromarray(ours_rgb)
            ours_pil = draw_tracking_trajectories(ours_pil, track2d_seq[:, moving_mask], mobile_color)
            ours_pil = draw_tracking_trajectories(ours_pil, track2d_seq[:, ~moving_mask], static_color)
            # ours_pil.show()
            
            ours_pil = draw_axis_on_image(
                ours_pil,
                obj_repr,
                draw_gt=True,
                draw_pred=True
            )
            ours_str = f'Ours\nType Err: {type_error}\nPivot Err: {pos_error}\nAngle Err: {angle_error}'
            ours_pil = add_text_to_image(
                image=ours_pil,
                text="Ours"+err_str, 
                text_color = (255, 255, 255), 
            )
            ours_pil.save(os.path.join(save_folder, method_name, obj_idx, f"ours_image_{obj_idx}.png"))
            
        elif method_name == "gpnet":
            pass
            gpnet_rgb = obj_repr.initial_frame.rgb
            bbox_start = obj_repr.fine_joint_dict["moving_bbox_first"] # 8, 3
            bbox_end = obj_repr.fine_joint_dict["moving_bbox_last"] # 8, 3
            lines = [
                # 0 1 4 2
                [0, 1],
                [0, 2],
                [2, 4],
                [4, 1],
                # 3 5 7 6
                [3, 5],
                [5, 7],
                [7, 6],
                [6, 3],
                #
                [0, 3],
                [1, 5],
                [4, 7],
                [2, 6]
            ]
            # TODO 繪制 bbox
        elif method_name == "gflow":
            # continue
            gflow_rgb = obj_repr.initial_frame.rgb
            gflow_c = obj_repr.fine_joint_dict["general_flow_c"] # T, M, 3
            T, M, _ = gflow_c.shape
            
            gflow_2d = camera_to_image(gflow_c.reshape(-1, 3), obj_repr.K)[0].reshape(T, M, 2)
            gflow_pil = Image.fromarray(gflow_rgb)
            
            random_flow_idx = farthest_point_sampling_2d(gflow_2d[0], gflow_2d.shape[1] // 4)
            gflow_pil = draw_tracking_trajectories(gflow_pil, gflow_2d[:, random_flow_idx], mobile_color)
        
            gflow_pil = draw_axis_on_image(
                gflow_pil,
                obj_repr,
                draw_gt=True,
                draw_pred=True
            )
            gflow_pil = add_text_to_image(
                image=gflow_pil,
                text= f"GeneralFlow"+err_str, 
                text_color = (255, 255, 255), 
            )
            gflow_pil.save(os.path.join(save_folder, method_name, obj_idx, f"gflow_image_{obj_idx}.png"))
        
        gt_image = draw_axis_on_image(
            init_pil,
            obj_repr,
            draw_gt=True,
            draw_pred=False
        )
        gt_image = add_text_to_image(
            image=gt_image,
            text= f"GT", 
            text_color = (255, 255, 255), 
        )
        gt_image.save(os.path.join(save_folder, method_name, obj_idx, f"gt_image_{obj_idx}.png"))