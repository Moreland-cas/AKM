"""
Input:
    rgb_seq: [t, h, w]
    depth_seq: [t, h, w]
    object_mask: [t, h, w]
    franka_mask: [t, h, w]
Output:
    moving_mask: [t, h, w]
    joint_states: [t, 1]
    joint_params:
        (3,) for prismatic joint
        (6,) for revolute joint
"""
import os
import torch
import random
import numpy as np
from PIL import Image
import sklearn.cluster as cluster
from cotracker.utils.visualizer import Visualizer
from embodied_analogy.pipeline.process_record import RecordDataReader
from embodied_analogy.perception.online_cotracker import track_any_points
from embodied_analogy.visualization.vis_tracks_3d import vis_tracks_3d_napari
from embodied_analogy.utility.utils import (
    depth_image_to_pointcloud, 
    visualize_pc, 
    camera_to_image,
    draw_points_on_image,
    image_to_camera
)

def filter_by_visibility(pred_tracks, pred_visibility):
    """
        pred_tracks: [T, M, 2]
        pred_visibility: [T, M]
    """
    always_visible_mask = pred_visibility.all(dim=0) # num_clusters
    pred_tracks = pred_tracks[:, always_visible_mask, :] # T M_ 2
    return pred_tracks

def filter_by_sam2(pred_tracks, sam2_mask):
    # TODO
    return pred_tracks

def extract_tracked_depths(depth_seq, pred_tracks):
    """
    Args:
        depth_seq (np.ndarray): 
            深度图视频，形状为 (T, H, W, 1)
        pred_tracks (torch.Tensor): 
            (T, M, 2)
            每个点的坐标形式为 [u, v]，值域为 [0, W) 和 [0, H)。
        
    返回:
        torch.Tensor: 点的深度值，形状为 (T, M)，每个点的深度值。
    """
    depth_seq = depth_seq.squeeze(-1)
    T, H, W = depth_seq.shape
    # _, M, _ = pred_tracks.shape

    # 确保 pred_tracks 是整数坐标
    u_coords = pred_tracks[..., 0].clamp(0, W - 1).long()  # 水平坐标 u
    v_coords = pred_tracks[..., 1].clamp(0, H - 1).long()  # 垂直坐标 v

    # 将 depth_seq 转为 torch.Tensor
    depth_tensor = torch.from_numpy(depth_seq).cuda()  # Shape: (T, H, W)

    # 使用高级索引提取深度值
    depth_tracks = depth_tensor[torch.arange(T).unsqueeze(1), v_coords, u_coords]  # Shape: (T, M)
    return depth_tracks

def filter_by_depthSeq(pred_tracks, depth_tracks, thr=1.5):
    """
    过滤深度序列中的无效序列，基于相邻元素的相对变化率。
    
    参数:
        pred_tracks (torch.Tensor): 
            形状为 (T, N, 2)，每个点的坐标形式为 [u, v]，值域为 [0, W) 和 [0, H)。
        depth_tracks (torch.Tensor): 
            形状为 (T, N)，表示 T 个时间步上的 N 个深度序列。
        thr (float): 
            相对变化率的阈值，超过此比率的序列会被判定为无效。
    
    返回:
        torch.Tensor: 
            筛选后的 pred_tracks, 形状为 (T, M, 2)，其中 M 是筛选后的有效点数量。
    """
    depth_tracks += 1e-6  # 防止除零

    # 转置 depth_tracks，使其形状为 (N, T)
    depth_tracks_T = depth_tracks.T  # (N, T)

    # 计算相邻元素的变化比率 (N, T-1)
    ratios = torch.max(depth_tracks_T[:, :-1] / depth_tracks_T[:, 1:], depth_tracks_T[:, 1:] / depth_tracks_T[:, :-1])

    # 检测每个序列是否存在无效的变化 (N,)
    invalid_sequences = torch.any(ratios > thr, dim=1)  # 检查每一行是否有任何值大于阈值

    # 合法序列的掩码 (N,)
    valid_mask = ~invalid_sequences  # 合法序列为 False，其他为 True
    
    # 筛选 pred_tracks 和 depth_tracks 的合法序列
    pred_tracks = pred_tracks[:, valid_mask, :]  # 筛选有效的 M 点，(T, M, 2)
    depth_tracks = depth_tracks[:, valid_mask]   # 筛选有效的 M 深度，(T, M)
    
    return pred_tracks, depth_tracks

def visualize_tracks_on_video(rgb_frames, pred_tracks, file_name="video_output"):
    """
        Args:
            rgb_frames: [T, H, W, 3]
            pred_tracks: [T, M, 2]
    """
    video = torch.tensor(np.stack(rgb_frames), device="cuda").permute(0, 3, 1, 2)[None]
    vis = Visualizer(save_dir="./", pad_value=120, linewidth=1)
    
    pred_tracks = pred_tracks[None] # [1, T, M, 2]
    # pred_visibility = torch.ones((pred_tracks.shape[:3], 1), dtype=torch.bool, device="cuda")
    vis.visualize(video, pred_tracks, filename=file_name)
    
    
record_path_prefix = "/home/zby/Programs/Embodied_Analogy/assets/recorded_data"
file_name = "/2025-01-07_18-06-10.npz"
dr = RecordDataReader(record_path_prefix, file_name)
dr.process_data()

rgb_seq = dr.rgb # T H W C
depth_seq = dr.depth # T H W 1
K = dr.intrinsic # 3, 3
Tw2c = dr.data["extrinsic"] # 4, 4
object_mask_0 = dr.seg
visualize = False
napari = False
save_intermidiate = True

# 保存数据到 tmp_folder
tmp_folder = "/home/zby/Programs/Embodied_Analogy/assets/tmp/"
rgb_folder = os.path.join(tmp_folder, "rgbs")
depth_folder = os.path.join(tmp_folder, "depths")
mask_folder = os.path.join(tmp_folder, "masks")
os.makedirs(rgb_folder, exist_ok=True)
os.makedirs(depth_folder, exist_ok=True)
os.makedirs(mask_folder, exist_ok=True)

if save_intermidiate:
    for i, rgb in enumerate(rgb_seq):
        Image.fromarray(rgb).save(os.path.join(rgb_folder, f"{i}.jpg"))
    for i, depth in enumerate(depth_seq):
        np.save(os.path.join(depth_folder, f"{i}.npy"), depth) # H, W, 1
        
if napari:
    # 展示初始的 pointcloud sequence
    T = depth_seq.shape[0]
    raw_pc_list = []
    for i in range(T):
        pc_i = depth_image_to_pointcloud(depth_seq[i].squeeze(), None, K) # N, 3
        indices = np.random.choice(pc_i.shape[0], size=10000, replace=False)
        # 使用这些索引获取对应的 100 x 3 数组
        pc_i = pc_i[indices]
        raw_pc_list.append(pc_i)
    raw_pc_list = np.stack(raw_pc_list)
    # vis_tracks_3d_napari(raw_pc_list)
    
pc_0 = depth_image_to_pointcloud(depth_seq[0].squeeze(), object_mask_0, K) # N, 3
rgb_0 = rgb_seq[0][object_mask_0] / 255. # N,3
if visualize:
    visualize_pc(pc_0, rgb_0)

# 1) 对初始帧中的物体点云聚类得到 num_clusters 个点簇, 并计算 num_clusters 个簇中心在图像上的投影
# TODO: 可以在一开始聚类的时候也考虑视觉特征
num_clusters = 300
feat_for_kmeans = np.concatenate([pc_0, rgb_0], axis=-1)
centroids, labels, _ = cluster.k_means(feat_for_kmeans, init="k-means++", n_clusters=num_clusters)

color_map = np.random.rand(num_clusters, 3)
colors = color_map[labels]

if visualize:
    visualize_pc(pc_0, colors)

# 2) 然后对这 k 个点进行 rgb tracking
centroids_camera = centroids[:, :3] # num_clusters, 3
centroids_image, _ = camera_to_image(centroids_camera, K) # num_clusters, 2

if visualize:
    draw_points_on_image(Image.fromarray(rgb_seq[0]), centroids_image).show()

# [T, M, 2], [T, M]
pred_tracks_2d, pred_visibility = track_any_points(rgb_seq, centroids_image, visiualize=visualize)

# 3) 对跟踪的点进行筛选
# 3.1）要是一直能跟踪到的点
pred_tracks_2d = filter_by_visibility(pred_tracks_2d, pred_visibility)

if visualize:
    visualize_tracks_on_video(rgb_seq, pred_tracks_2d, "after_vis_filter")
    
# 3.2）要是一直在物体区域内的点(物体区域由sam2得到)
pred_tracks_2d = filter_by_sam2(pred_tracks_2d, sam2_mask=None)

if visualize:
    visualize_tracks_on_video(rgb_seq, pred_tracks_2d, "after_sam2_filter")
    
# 3.3）这些点的深度不应该有突然的变化
# TODO: 改为线性模型的回归
depth_tracks = extract_tracked_depths(depth_seq, pred_tracks_2d) # T, M
pred_tracks_2d, depth_tracks = filter_by_depthSeq(pred_tracks_2d, depth_tracks) # [T, M, 2], [T, M]
    
if visualize:
    visualize_tracks_on_video(rgb_seq, pred_tracks_2d, "after_depth_filter")

# 4) 根据这 m 个点的轨迹将其聚类为 static 和 moving 两类
# 得到追踪点的三维轨迹
T, M, _ = pred_tracks_2d.shape
pred_tracks_3d = image_to_camera(pred_tracks_2d.reshape(T*M, -1), depth_tracks.reshape(-1), K) # T * M, 3
pred_tracks_3d = pred_tracks_3d.reshape(T, M, 3) # T, M, 3

# if napari:
#     vis_tracks_3d_napari(pred_tracks_3d)
    
motion_for_kmeans = pred_tracks_3d.permute(1, 0, 2).reshape(M, -1) # M T*3

_, moving_labels, _ = cluster.k_means(motion_for_kmeans.cpu().numpy(), init="k-means++", n_clusters=2)
red_and_green = np.array([[1, 0, 0], [0, 1, 0]])
rigid_part_colors = red_and_green[moving_labels] # M, 3

if napari:
    vis_tracks_3d_napari(pred_tracks_3d, rigid_part_colors)
    
if visualize:
    visualize_pc(pred_tracks_3d[:M, :].cpu().numpy(), rigid_part_colors)
    
# 在这里跑 sam2, 得到 moving_part 和 static_part 的分割
from embodied_analogy.perception.sam2_masking import run_sam2_whole
from embodied_analogy.visualization.vis_sam2_mask import visualize_sam2_mask
moving_tracks_2d = pred_tracks_2d[:, moving_labels == 1, :] # T, N, 2
static_tracks_2d = pred_tracks_2d[:, moving_labels == 0, :] # T, N, 2
point_prompt = torch.stack([moving_tracks_2d[0][0], static_tracks_2d[0][0]]).cpu() #2, 2
video_masks = run_sam2_whole(rgb_folder, initial_point_prompt=point_prompt)
if visualize:
    visualize_sam2_mask(rgb_seq, video_masks)

if save_intermidiate:
    for i, mask in enumerate(video_masks):
        mask_255 = (mask * 255).astype(np.uint8)
        Image.fromarray(mask_255).save(os.path.join(mask_folder, f"{i}.png"))
        # np.save(os.path.join(mask_folder, f"{i}.npy"), mask)

# 5) 初步估计出 joint parameters
from embodied_analogy.estimation.joint_estimate_w_corr import estimate_translation_from_tracks
    
translation_c, scales = estimate_translation_from_tracks(pred_tracks_3d)
Rc2w = Tw2c[:3, :3].T # 3, 3
translation_w = Rc2w @ translation_c

# 将 joint states 保存到 tmp_folder
if save_intermidiate:
    np.savez(
        os.path.join(tmp_folder, "joint_state.npz"), 
        translation_c=translation_c, 
        translation_w=translation_w, 
        scales=scales
    )

# 6) 对于第一帧剩下的 k-m 个点簇进行分类验证, 分为 static, moving 和 unknown 三类
# static 和 moving 里又分为一直能看到的, 和随着运动新发现的 


# 7) 用所有 known 的点簇重新估计 joint parameters