import json
import os
import numpy as np
from PIL import Image
from embodied_analogy.utility.utils import (
    visualize_pc,
    add_text_to_image,
    depth_image_to_pointcloud
)
from embodied_analogy.representation.obj_repr import Obj_repr
from embodied_analogy.representation.basic_structure import Frame
from embodied_analogy.utility.constants import MOVING_LABEL, STATIC_LABEL

from PIL import Image, ImageDraw

def get_rotation_matrix_x(alpha_deg):
    """
    生成绕 X 轴正方向旋转 alpha 度的旋转矩阵。

    参数：
        alpha_deg (float): 旋转角度（度）

    返回：
        np.ndarray: 3x3 旋转矩阵
    """
    # 转换为弧度
    alpha_rad = np.deg2rad(alpha_deg)
    
    # 构建旋转矩阵
    cos_a = np.cos(alpha_rad)
    sin_a = np.sin(alpha_rad)
    rotation_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_a, -sin_a],
        [0.0, sin_a, cos_a]
    ])
    
    return rotation_matrix

gray_color = np.array([128, 128, 128]) / 255
apple_green_color = np.array([159, 191, 82]) / 255
slight_green_color = np.array([149, 187, 114]) / 255
mobile_pc_color = np.array([147, 205, 245]) / 255
static_pc_color = np.array([255, 0, 0]) / 255

render_intrinsic = np.array(
    [[300.,   0., 400.],
    [  0., 300., 300.],
    [  0.,   0.,   1.]], dtype=np.float32)
render_extrinsic = np.eye(4)
# render_extrinsic[:3, :3] = get_rotation_matrix_x(-10)
# render_extrinsic[1, -1] = 0.1 # y 軸
zoom_in_scale = 2

# position_ratio_method = (0.45, 0.05)
position_ratio_error = (0.05, 0.8)

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
    if M > N:
        M = N
    
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

def find_medium_alpha(pos, dir, K):
    """
    找到使得 pos + alpha * dir 後能完全展示在屏幕內的 joint alpha 值。
    """
    # import pdb;pdb.set_trace()
    alpha = np.linspace(-2, 2, 100)
    from embodied_analogy.utility.utils import camera_to_image
    mid_points = pos[None] + (alpha[:, None] + 0.5) * dir[None]
    uv, depth = camera_to_image(mid_points, K)
    W, H = K[0, -1] * 2, K[1, -1] * 2
    in_mask = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    mid_points = mid_points[in_mask & (depth > 0.1)] # M, 2
    depth = depth[in_mask & (depth > 0.1)]
    # 找到 depth 最小的那個
    min_point = mid_points[np.argmin(depth)]
    alpha = ((min_point - pos) / dir )[0] - 0.5
    new_pivot = pos + alpha * dir
    return alpha, new_pivot
    
    
def find_closest_alpha(pos, dir, pos2):
    """
    找到 alpha，使得 pos + alpha * dir 到 pos2 的距离最小。
    
    参数：
        pos (np.ndarray): 起始点，形状 (3,)
        dir (np.ndarray): 方向向量，形状 (3,)
        pos2 (np.ndarray): 目标点，形状 (3,)
    
    返回：
        tuple: (alpha, closest_point, distance)
            - alpha (float): 使得距离最小的 alpha 值
            - closest_point (np.ndarray): 直线上的最近点，形状 (3,)
            - distance (float): 最近点到 pos2 的距离
    """
    # 转换为 NumPy 数组并验证形状
    pos = np.asarray(pos)
    dir = np.asarray(dir)
    pos2 = np.asarray(pos2)
    
    if pos.shape != (3,) or dir.shape != (3,) or pos2.shape != (3,):
        raise ValueError("pos, dir, pos2 必须是形状为 (3,) 的向量")
    
    # 计算 dir 的模平方
    dir_norm_sq = np.dot(dir, dir)
    
    # 检查 dir 是否为零向量
    if dir_norm_sq < 1e-10:  # 阈值避免数值误差
        raise ValueError("dir 不能是零向量")
    
    # 计算 v = pos - pos2
    v = pos - pos2
    
    # 计算 alpha
    alpha = -np.dot(v, dir) / dir_norm_sq
    
    # 计算最近点
    closest_point = pos + alpha * dir
    
    # 计算最小距离
    distance = np.linalg.norm(closest_point - pos2)
    
    return alpha, closest_point, distance


data_path = "/home/zby/Programs/Embodied_Analogy/assets/logs_draw"
method_names = ["gflow", "gpnet", "ours"]
obj_idx_list = os.listdir(os.path.join(data_path, method_names[0]))

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
"""
保存方式爲
/reconstruct
    /obj_idx
        obj.png
        ours.png
        gpnet.png
        gflow.png
""" 
save_folder = f"/home/zby/Programs/Embodied_Analogy/scripts/draw_figs/paper_figs/reconstruct"
for obj_idx in obj_idx_list_filtered:
    print("obj_idx:", obj_idx)
    
    for method_name in method_names:
        if obj_idx != "104" or method_name != "gflow":
            continue
        print("method_name:", method_name)
        obj_save_folder = save_folder
        os.makedirs(obj_save_folder, exist_ok=True)
        obj_repr: Obj_repr
        obj_repr = Obj_repr.load(f"{data_path}/{method_name}/{obj_idx}/obj_repr.npy")
        
        obj_repr.initial_frame.segment_obj(
            obj_description=obj_repr.obj_description,
            post_process_mask=True,
            filter=True,
            visualize=False,
        )

        init_rgb = obj_repr.initial_frame.rgb
        init_obj_mask = obj_repr.initial_frame.obj_mask
        init_rgb[~init_obj_mask] = 255
        init_pil = Image.fromarray(init_rgb)
        init_pil = init_pil.resize((3200, 2400))
        init_pil.save(os.path.join(obj_save_folder, f"obj_{obj_idx}.png"))
        
        # 不妨先用第一幀的點暈做背景
        obj_pc, pc_colors = obj_repr.initial_frame.get_obj_pc(
            use_robot_mask=True, 
            use_height_filter=True,
            world_frame=False,
        )
        pc_colors = pc_colors / 255
        
        recon_result = json.load(open(os.path.join(data_path, method_name, str(obj_idx), "recon_result.json"), "r"))
        fine_loss_dict = recon_result["fine_loss"]
        # joint_type = recon_result["gt_w"]["joint_type"]
        type_error = int(fine_loss_dict["type_err"])
        pos_error = fine_loss_dict["pos_err"] * 100
        angle_error = np.rad2deg(fine_loss_dict["angle_err"])
        # err_str = f'Pivot Err: {pos_error:.1f}cm\nAngle Err: {angle_error:.1f}°'
        err_str = f'Pivot/Angle Err: {pos_error:.1f}cm/{angle_error:.1f}°'
        
        # joint information
        fine_joint_dict = obj_repr.get_joint_param(resolution="fine", frame="camera")
            
        if method_name == "ours":
            # 計算點暈
            first_kframe_mobile_mask = (obj_repr.kframes[0].dynamic_mask == MOVING_LABEL)
            
            moving_pc = depth_image_to_pointcloud(depth_image=obj_repr.initial_frame.depth, mask=first_kframe_mobile_mask, K=obj_repr.K)
            static_pc = depth_image_to_pointcloud(depth_image=obj_repr.initial_frame.depth, mask=obj_repr.initial_frame.obj_mask&(~first_kframe_mobile_mask), K=obj_repr.K)
            
            ours_tracks_3d = obj_repr.frames.track3d_seq[:, obj_repr.frames.moving_mask]
            random_track_idx = farthest_point_sampling_2d(ours_tracks_3d[0], 200)
            ours_rgb = visualize_pc(
                points=[moving_pc, static_pc],
                colors=[mobile_pc_color, gray_color],
                point_size=[5, 5],
                voxel_size=0.01,
                alpha=[1.0, 0.6],
                camera_intrinsic=render_intrinsic,
                camera_extrinsic=render_extrinsic,
                pivot_point=fine_joint_dict["joint_start"],
                joint_axis=fine_joint_dict["joint_dir"],
                tracks_3d=ours_tracks_3d[:, random_track_idx],
                tracks_3d_colors=apple_green_color,
                tracks_t_step=3, 
                tracks_n_step=None,
                tracks_norm_threshold=0.2e-2,
                zoom_in_scale=zoom_in_scale,
                online_viewer=False
            )
            # continue
            ours_pil = Image.fromarray(ours_rgb)
            ours_pil = add_text_to_image(
                image=ours_pil,
                text=err_str, 
                text_color=(0, 0, 0), 
                position_ratio=position_ratio_error
            )
            ours_pil.save(os.path.join(obj_save_folder, f"{obj_idx}_ours.png"))
            
        elif method_name == "gpnet":
            if fine_joint_dict["joint_type"] != "prismatic":
                # alpha, new_pivot, _ = find_closest_alpha(fine_joint_dict["joint_start"], fine_joint_dict["joint_dir"], np.array([0, 0, 0]))
                # fine_joint_dict["joint_start"] = new_pivot
                
                alpha, new_pivot = find_medium_alpha(fine_joint_dict["joint_start"], fine_joint_dict["joint_dir"], render_intrinsic)
                fine_joint_dict["joint_start"] = new_pivot
                
            gpnet_bboxes = [fine_joint_dict["moving_bbox_first"], fine_joint_dict["moving_bbox_last"]]
            gpnet_rgb = visualize_pc(
                points=[obj_pc],
                colors=[gray_color],
                point_size=[5],
                voxel_size=0.01,
                alpha=[0.6],
                camera_intrinsic=render_intrinsic,
                camera_extrinsic=render_extrinsic,
                pivot_point=fine_joint_dict["joint_start"],
                joint_axis=fine_joint_dict["joint_dir"],
                bboxes=gpnet_bboxes,
                bbox_radius=0.0015,
                zoom_in_scale=zoom_in_scale,
                online_viewer=False,
            )
            # continue
            gpnet_pil = Image.fromarray(gpnet_rgb)
            gpnet_pil = add_text_to_image(
                image=gpnet_pil,
                text=err_str, 
                text_color=(0, 0, 0), 
                position_ratio=position_ratio_error
            )
            gpnet_pil.save(os.path.join(obj_save_folder, f"{obj_idx}_gpnet.png"))
            
        elif method_name == "gflow":
            if fine_joint_dict["joint_type"] != "prismatic":
                alpha, new_pivot, _ = find_closest_alpha(fine_joint_dict["joint_start"], fine_joint_dict["joint_dir"], np.array([0, 0, 0]))
                fine_joint_dict["joint_start"] = new_pivot
                
                # alpha, new_pivot = find_medium_alpha(fine_joint_dict["joint_start"], fine_joint_dict["joint_dir"], render_intrinsic)
                # fine_joint_dict["joint_start"] = new_pivot
                
            gflow_tracks_3d = fine_joint_dict["general_flow_c"]
            random_track_idx = farthest_point_sampling_2d(gflow_tracks_3d[0], 25)
            gflow_rgb = visualize_pc(
                points=[obj_pc],
                colors=[gray_color],
                point_size=[5],
                voxel_size=0.01,
                alpha=[0.6],
                camera_intrinsic=render_intrinsic,
                camera_extrinsic=render_extrinsic,
                pivot_point=fine_joint_dict["joint_start"],
                joint_axis=fine_joint_dict["joint_dir"],
                tracks_3d=gflow_tracks_3d[:, random_track_idx],
                tracks_3d_colors=apple_green_color,
                tracks_t_step=3, 
                tracks_n_step=None,
                tracks_norm_threshold=0.2e-2,
                zoom_in_scale=zoom_in_scale,
                online_viewer=False
            )
            gflow_pil = Image.fromarray(gflow_rgb)
            gflow_pil = add_text_to_image(
                image=gflow_pil,
                text=err_str, 
                text_color=(0, 0, 0), 
                position_ratio=position_ratio_error
            )
            gflow_pil.save(os.path.join(obj_save_folder, f"{obj_idx}_gflow.png"))
        
    
        if method_name == "ours":
            gt_joint_dict = obj_repr.get_joint_param(resolution="gt", frame="camera")
            
            if gt_joint_dict["joint_type"] == "prismatic":
                gt_joint_dict["joint_start"] = fine_joint_dict["joint_start"]
            else:
                alpha, new_pivot, _ = find_closest_alpha(gt_joint_dict["joint_start"], gt_joint_dict["joint_dir"], fine_joint_dict["joint_start"])
                gt_joint_dict["joint_start"] = new_pivot
                
            gt_rgb = visualize_pc(
                points=[obj_pc],
                colors=[pc_colors],
                point_size=[3],
                voxel_size=None,
                alpha=[0.8],
                camera_intrinsic=render_intrinsic,
                camera_extrinsic=render_extrinsic,
                pivot_point=gt_joint_dict["joint_start"],
                joint_axis=gt_joint_dict["joint_dir"],
                zoom_in_scale=zoom_in_scale,
                online_viewer=False
            )
            # continue
            gt_pil = Image.fromarray(gt_rgb)
            # gt_pil = add_text_to_image(
            #     image=gt_pil,
            #     text="Ground Truth", 
            #     text_color=(0, 0, 0), 
            #     position_ratio=position_ratio_method
            # )
            gt_pil = add_text_to_image(
                image=gt_pil,
                text=f'Pivot/Angle Err: {0.0}cm/{0.0}°', 
                text_color=(0, 0, 0), 
                position_ratio=position_ratio_error
            )
            gt_pil.save(os.path.join(obj_save_folder, f"{obj_idx}_gt.png"))
        