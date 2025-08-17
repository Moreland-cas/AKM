import os
import math
import json
import numpy as np
from PIL import Image
from PIL import Image, ImageDraw
from graspnetAPI.grasp import Grasp

from akm.utility.utils import (
    add_text_to_image,
    camera_to_image
)
from akm.representation.obj_repr import Obj_repr
from akm.utility.constants import MOVING_LABEL, STATIC_LABEL

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
    Draws a directed axis with an arrow on a PIL Image.
    Parameters:
    - image: PIL Image object
    - start_point: Starting point coordinates (u, v)
    - end_point: End point coordinates (u, v)
    - line_color: Line color (R, G, B)
    - line_width: Line width
    - arrow_size: Arrow size
    """
    draw = ImageDraw.Draw(image)
    
    if isinstance(start_point, np.ndarray):
        start_point = tuple(start_point.astype(np.int32))
    if isinstance(end_point, np.ndarray):
        end_point = tuple(end_point.astype(np.int32))
    
    line_width = int(line_width)
    arrow_size = int(arrow_size)
    draw.line([start_point, end_point], fill=line_color, width=line_width)
    
    # arrow direction
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    
    # arrow length
    length = math.sqrt(dx**2 + dy**2)
    
    if length == 0:
        return  
    
    # unit vector
    ux = dx / length
    uy = dy / length
    
    # Calculate the two flanking points of the arrow
    # Arrow angle (relative to the main line direction, 30 degrees)
    arrow_angle = math.radians(30)
    
    # Arrow left wing point
    arrow_left_x = end_point[0] - arrow_size * (ux * math.cos(arrow_angle) + uy * math.sin(arrow_angle))
    arrow_left_y = end_point[1] - arrow_size * (uy * math.cos(arrow_angle) - ux * math.sin(arrow_angle))
    
    # Arrow right wing point
    arrow_right_x = end_point[0] - arrow_size * (ux * math.cos(arrow_angle) - uy * math.sin(arrow_angle))
    arrow_right_y = end_point[1] - arrow_size * (uy * math.cos(arrow_angle) + ux * math.sin(arrow_angle))
    
    # Draw an arrow (filled triangle)
    draw.polygon([
        end_point,
        (arrow_left_x, arrow_left_y),
        (arrow_right_x, arrow_right_y)
    ], fill=line_color)
    return image

data_path = "/home/Programs/AKM/assets/logs_draw"
save_folder = f"/home/Programs/AKM/scripts/draw_figs/paper_figs/reconstruct"
method_names = ["gflow", "gpnet", "ours"]
obj_idx_list = os.listdir(os.path.join(data_path, method_names[0]))

def draw_axis_on_image(
    image,
    obj_repr,
    draw_gt=False,
    draw_pred=False
):
    """
    Draw the joint axis on the image.
    The pivot_point and joint_dir are both in the camera coordinate system.
    The joint position needs to be adjusted based on the projection position.
    """
    joint_length = 0.1
    gt_joint_dict = obj_repr.get_joint_param(resolution="gt", frame="camera")
    fine_joint_dict = obj_repr.get_joint_param(resolution="fine", frame="camera")
    
    # Scale the pil image and K at the same time
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
            draw_start_point = pred_start_point
            draw_end_point = pred_start_point + (pred_end_point - pred_start_point) * joint_scale
            
        image = draw_directed_axis(image, draw_start_point, draw_end_point, line_color=yellow, line_width=2*resize_scale, arrow_size=10*resize_scale)
    return image

def draw_tracking_trajectories(image, tracks, colors):
    """
    Draws 2D tracking tracks on a PIL image.

    Parameters:
        - image: PIL image object
        - tracks: NumPy array of (T, M, 2) representing M tracks, each track has T points, each point is an (x, y) coordinate
        - colors: NumPy array of (3,) representing the color of each track, in (R, G, B) format
    """
    T, M, _ = tracks.shape
    colors = np.tile(colors, (M, 1))
    
    draw = ImageDraw.Draw(image)
    for m in range(M):
        # Get the mth trajectory (T, 2)
        track = tracks[:, m, :]  
        # Get the color (R, G, B) of the mth track
        color = colors[m]  
        
        # Traverse each point pair in the trajectory and draw line segments
        for t in range(T - 1):
            start_point = tuple(track[t]) 
            end_point = tuple(track[t + 1]) 
            # Make a gradient for the color
            tmp_color = t/T * color + (1-t/T) * np.array([128, 128, 128])
            tmp_color = tuple(tmp_color.astype(int)) 
            draw.line([start_point, end_point], fill=tmp_color, width=1)
    
    return image

def idx_available(obj_idx):
    """
    Read recon_result.json under obj_idx for each of the three methods, and only draw if there is a valid recon
    """
    data_path = "/home/Programs/AKM/assets/logs_draw"
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
        
        recon_result = json.load(open(os.path.join(data_path, method_name, str(obj_idx), "recon_result.json"), "r"))
        fine_loss_dict = recon_result["fine_loss"]
        type_error = int(fine_loss_dict["type_err"])
        pos_error = fine_loss_dict["pos_err"] * 100
        angle_error = np.rad2deg(fine_loss_dict["angle_err"])
        err_str = f'\nType Err: {type_error}\nPivot Err: {pos_error:.1f}cm\nAngle Err: {angle_error:.1f}°'
        
        if method_name == "ours":
            # Draw point clouds of different colors and draw axes
            dynamic_mask = obj_repr.kframes[0].dynamic_mask
            moving_mask = dynamic_mask == MOVING_LABEL
            static_mask = dynamic_mask == STATIC_LABEL
            
            ours_rgb = obj_repr.initial_frame.rgb
            
            # draw dynamic mask
            alpha = 0.8
            ours_rgb[static_mask] = alpha * init_rgb[static_mask] + (1-alpha) * static_color 
            ours_rgb[moving_mask] = alpha * init_rgb[moving_mask] + (1-alpha) * mobile_color 
            
            # draw tracking
            track2d_seq = obj_repr.frames.track2d_seq # T, M, 2
            moving_mask = obj_repr.frames.moving_mask
            ours_pil = Image.fromarray(ours_rgb)
            ours_pil = draw_tracking_trajectories(ours_pil, track2d_seq[:, moving_mask], mobile_color)
            ours_pil = draw_tracking_trajectories(ours_pil, track2d_seq[:, ~moving_mask], static_color)
            
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
        elif method_name == "gflow":
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